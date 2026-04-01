import json
import math
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from datasets import load_dataset
from scipy.stats import chi2
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.dataset import CodeDataset
from utils.model import MARGINLossHead, MARGINModel
from utils.math import (
    compute_geometric_median,
    compute_metrics,
    compute_pairwise_margin,
    compute_vmf_kappa,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== 常量配置区 ====================
# 数据配置
DATASET_NAME = "codemetic/MARGIN"
DATASET_SUBSET = "debug"  # 可选其他subset
MAX_LENGTH = 512

# 模型配置
MODEL_NAME = "microsoft/graphcodebert-base"
EMBEDDING_DIM = 768  # graphcodebert-base的维度

# 训练配置
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = MAX_EPOCHS
SCHEDULER_PATIENCE = 3

# ArcFace & 球面配置
BASE_SCALE = 30.0  # s
CONFIDENCE_ALPHA = 0.95  # α
MIN_KAPPA = 1.0  # 防止kappa过小导致数值不稳定
SEED = 42

# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIME_PREFIX = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# 输出配置
OUTPUT_DIR = f"./output/{MODEL_NAME.split('/')[1]}-{TIME_PREFIX}"
UMAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "umap")
PROTOTYPE_ALIGNMENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prototype-alignment")
PROTOTYPE_DISPERSION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prototype-dispersion")
REPORT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "report")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UMAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_ALIGNMENT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_DISPERSION_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# UMAP配置
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

# 几何中位数配置
GEOMEDIAN_MAX_ITER = 100
GEOMEDIAN_TOL = 1e-5

set_seed(SEED)


# ==================== 训练器 ====================
class Trainer:
    def __init__(
        self, model: MARGINModel, train_dataset, val_dataset, label2id, id2label
    ):
        self.model = model.to(DEVICE)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label2id = label2id
        self.id2label = id2label
        self.num_classes = len(label2id)

        # 找出Non-vul的索引（假设为0或名称为'Non-vul'）
        self.non_vul_idx = 0
        if "Non-vul" in label2id:
            self.non_vul_idx = label2id["Non-vul"]

        self.criterion: MARGINLossHead = MARGINLossHead(
            self.num_classes, BASE_SCALE
        ).to(DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scaler = GradScaler()

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

        # 存储每轮的原型
        self.geometric_medians = None  # [C, D]

    def compute_epoch_kappas(self, dataloader):
        """计算每个类别的kappa值（基于训练集）"""
        self.model.eval()
        # 收集每个类别的特征
        class_features = {i: [] for i in range(self.num_classes)}
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing kappas", leave=False):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                _, features = self.model(
                    input_ids, attention_mask, return_features=True
                )

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_features[label].append(features[i].cpu())
        # 计算每个类别的均值结果向量长度
        kappas = {}
        for class_idx in range(self.num_classes):
            if len(class_features[class_idx]) > 0:
                feats = torch.stack(class_features[class_idx])
                feats = F.normalize(feats, dim=1)
                # 计算均值向量
                mean_vec = torch.mean(feats, dim=0)
                r_bar = torch.norm(mean_vec).item()
                # 计算kappa
                kappa = compute_vmf_kappa(torch.tensor(r_bar), EMBEDDING_DIM)
                kappas[class_idx] = kappa
            else:
                kappas[class_idx] = MIN_KAPPA
        return kappas

    def compute_adaptive_margins(self, kappas):
        """基于kappas计算自适应margins"""
        margins = {}
        for i in range(self.num_classes):
            max_margin = 0.0
            for j in range(self.num_classes):
                if i != j:
                    kappa_i = max(kappas[i], MIN_KAPPA)
                    kappa_j = max(kappas[j], MIN_KAPPA)

                    delta_m = compute_pairwise_margin(
                        kappa_i, kappa_j, EMBEDDING_DIM, CONFIDENCE_ALPHA
                    )
                    max_margin = max(max_margin, delta_m)
            margin_rad = max_margin
            margin_rad = min(margin_rad, math.pi)
            margins[i] = margin_rad
        return margins

    def compute_adaptive_scales(self, kappas: dict):
        """
        使用 scale[i] = BASE_SCALE * mean(kappas) / kappa[i]
        kappas: dict {class_idx: kappa_value}
        """
        scales = {}
        if not kappas:
            return scales

        base_scale = getattr(self, "BASE_SCALE", 1.0)

        # 计算平均值
        kappa_mean = sum(kappas.values()) / len(kappas)

        for i in range(self.num_classes):
            kappa = kappas.get(i, kappa_mean)  # 如果缺失，则使用平均值

            # 避免除以 0
            if kappa <= 1e-6:
                kappa = 1e-6

            scale = base_scale * (1 - (kappa / kappa_mean))
            scales[i] = float(scale)

        return scales

    def compute_geometric_median_prototypes(self, dataloader):
        """计算几何中位数原型（基于训练集）"""
        self.model.eval()

        # 收集每个类别的特征
        class_features = {i: [] for i in range(self.num_classes)}

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc="Computing geometric medians", leave=False
            ):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                _, features = self.model(
                    input_ids, attention_mask, return_features=True
                )

                for i in range(len(labels)):
                    label = labels[i].item()
                    class_features[label].append(features[i])

        # 计算几何中位数
        geometric_medians = torch.zeros(self.num_classes, EMBEDDING_DIM, device=DEVICE)

        for class_idx in range(self.num_classes):
            if len(class_features[class_idx]) > 0:
                feats = torch.stack(class_features[class_idx])
                median = compute_geometric_median(
                    feats, GEOMEDIAN_MAX_ITER, GEOMEDIAN_TOL
                )
                geometric_medians[class_idx] = median
            else:
                # 如果没有样本，使用随机初始化
                geometric_medians[class_idx] = F.normalize(
                    torch.randn(EMBEDDING_DIM, device=DEVICE), p=2, dim=0
                )

        self.geometric_medians = geometric_medians
        return geometric_medians

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            self.optimizer.zero_grad()

            # 混合精度训练
            with autocast(DEVICE):
                cos_theta = self.model(input_ids, attention_mask)
                loss = self.criterion(cos_theta, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def evaluate(self, dataloader, epoch, save_prefix="val"):
        """评估模型"""
        self.model.eval()

        all_pred_label_idx = []
        all_truth_label_idx = []
        all_features = []
        all_raw_labels = []

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch} Evaluating", leave=False)
            for batch in pbar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                # 使用几何中位数原型进行分类
                _, features = self.model(
                    input_ids, attention_mask, return_features=True
                )

                # 计算与几何中位数原型的余弦相似度
                if self.geometric_medians is not None:
                    sim = torch.matmul(features, self.geometric_medians.t())
                    preds = torch.argmax(sim, dim=1)
                else:
                    # 如果没有几何中位数，使用weight prototypes
                    cos_theta = self.model(input_ids, attention_mask)
                    preds = torch.argmax(cos_theta, dim=1)

                # 计算loss（用于早停）
                with autocast(DEVICE):
                    cos_theta = self.model(input_ids, attention_mask)
                    loss = self.criterion(cos_theta, labels)

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                all_pred_label_idx.extend(preds.cpu().numpy())
                all_truth_label_idx.extend(labels.cpu().numpy())
                all_features.append(features.cpu())
                all_raw_labels.extend(batch["raw_label"])

        avg_loss = total_loss / num_batches
        all_features = torch.cat(all_features, dim=0).numpy()

        # 计算指标
        metrics = compute_metrics(
            all_truth_label_idx, all_pred_label_idx, self.id2label
        )
        metrics["val_loss"] = avg_loss

        # 保存到JSON
        json_path = os.path.join(
            REPORT_OUTPUT_DIR, f"{save_prefix}_metrics_epoch_{epoch}.json"
        )
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # 打印指标
        print(f"\nEpoch {epoch} Evaluation Results:")
        print(
            f"🐱 Binary - MCC: {metrics['binary']['mcc']:.4f}, F1: {metrics['binary']['f1']:.4f}, "
            f"Prec: {metrics['binary']['precision']:.4f}, Rec: {metrics['binary']['recall']:.4f}"
        )

        if "positive_macro" in metrics:
            print(
                f"🐒 Positive-Macro - MCC: {metrics['positive_macro']['mcc']:.4f}, "
                f"F1: {metrics['positive_macro']['f1']:.4f}"
            )

        print(
            f"🌏 Global-Macro - MCC: {metrics['global_macro']['mcc']:.4f}, "
            f"F1: {metrics['global_macro']['f1']:.4f}, "
            f"FNR: {metrics['global_macro']['fnr']:.4f}, "
            f"FPR: {metrics['global_macro']['fpr']:.4f}"
        )

        # 绘制可视化
        self.visualize_epoch(all_features, all_truth_label_idx, epoch, metrics)

        return avg_loss, metrics

    def visualize_epoch(self, features, labels, epoch, metrics):
        """绘制热力图和UMAP"""
        import seaborn as sns

        sns.set_style("whitegrid")

        # 1. 几何中位数原型相似度热力图
        if self.geometric_medians is not None:
            geo_medians = self.geometric_medians.cpu().numpy()
            sim_matrix = np.matmul(geo_medians, geo_medians.T)
            # 转换为百分比
            sim_matrix = (sim_matrix + 1) / 2 * 100  # 从[-1,1]映射到[0,100]
            mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                sim_matrix,
                annot=True,
                mask=mask,
                fmt=".0f",
                cmap="YlOrRd",
                vmin=0,
                vmax=100,
                xticklabels=[self.id2label[i] for i in range(self.num_classes)],
                yticklabels=[self.id2label[i] for i in range(self.num_classes)],
            )
            plt.title(f"Geometric Median Prototype Similarity (%) - Epoch {epoch}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    PROTOTYPE_DISPERSION_OUTPUT_DIR, f"geo_median_sim_epoch_{epoch}.svg"
                )
            )
            plt.close()

        # 2. Weight prototype与Geometric median prototype相似度热力图
        weight_protos = self.model.get_weight_prototypes().detach()
        if self.geometric_medians.detach() is not None:
            sim_matrix = torch.matmul(self.geometric_medians.detach(), weight_protos.T)
            sim_matrix = ((sim_matrix + 1) / 2 * 100).cpu().numpy()
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                sim_matrix,
                annot=True,
                fmt=".0f",
                cmap="coolwarm",
                vmin=0,
                vmax=100,
                xticklabels=[f"W-{self.id2label[i]}" for i in range(self.num_classes)],
                yticklabels=[f"G-{self.id2label[i]}" for i in range(self.num_classes)],
            )
            plt.title(
                f"Weight vs Geometric Median Prototype Similarity (%) - Epoch {epoch}"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    PROTOTYPE_ALIGNMENT_OUTPUT_DIR, f"weight_geo_sim_epoch_{epoch}.svg"
                )
            )
            plt.close()

        # 3. UMAP可视化
        reducer = umap.UMAP(
            n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=SEED
        )
        embedding = reducer.fit_transform(features)

        plt.figure(figsize=(6, 5))

        # 先画正样本（Non-vul为灰色，其他为有颜色）
        unique_labels = sorted(set(labels))

        # 定义颜色：Non-vul为灰色，其他为tab10
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels) - 1))
        color_map = {}
        idx = 0
        for label in unique_labels:
            if label == self.non_vul_idx:
                color_map[label] = "gray"
            else:
                color_map[label] = colors[idx % len(colors)]
                idx += 1

        # 后画负样本（Non-vul）
        mask = np.array(labels) == self.non_vul_idx
        plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c="gray",
            label=self.id2label.get(self.non_vul_idx, "Non-vul"),
            alpha=0.3,
            s=30,
            edgecolors="none",
        )

        # 先画正样本（非Non-vul）
        for label in unique_labels:
            if label != self.non_vul_idx:
                mask = np.array(labels) == label
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[color_map[label]],
                    label=self.id2label[label],
                    alpha=0.9,
                    s=20,
                    edgecolors="none",
                )

        plt.legend(loc="best", fontsize=8)
        plt.title(f"UMAP Visualization - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(os.path.join(UMAP_OUTPUT_DIR, f"umap_epoch_{epoch}.svg"))
        plt.close()

    def train(self):
        """主训练循环"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        for epoch in range(1, MAX_EPOCHS + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{MAX_EPOCHS}")
            print(f"{'='*50}")

            # 1. 计算当前epoch的kappas（基于训练集）
            kappas = self.compute_epoch_kappas(train_loader)
            self.criterion.update_kappas(kappas)

            # 2. 计算adaptive margins
            margins = self.compute_adaptive_margins(kappas)
            self.criterion.update_margins(margins)

            # 3. 计算adaptive scales
            scales = self.compute_adaptive_scales(kappas)
            self.criterion.update_scales(scales)

            # 打印当前margin和kappa
            print("\nClass-wise Kappa and Margin:")
            for i in range(self.num_classes):
                print(
                    f"  {self.id2label[i]}: κ={kappas[i]:.2f}, m={margins[i]}, s={scales[i]}"
                )

            # 3. 训练
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"\nTrain Loss: {train_loss:.4f}")

            # 4. 计算几何中位数原型（基于训练集）
            self.compute_geometric_median_prototypes(train_loader)

            # 5. 验证
            val_loss, metrics = self.evaluate(val_loader, epoch)
            print(f"Val Loss: {val_loss:.4f}")

            # 6. 早停检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型状态
                self.best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "kappas": kappas,
                    "margins": margins,
                }
                # 保存到文件
                # torch.save(
                #     self.best_model_state, os.path.join(OUTPUT_DIR, "best_model.pt")
                # )
                print("Model improved, saved checkpoint.")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

        # 加载最佳模型
        if self.best_model_state is not None:
            print(f"\nLoading best model from epoch {self.best_model_state['epoch']}")
            self.model.load_state_dict(self.best_model_state["model_state_dict"])

        return self.model


# ==================== 主函数 ====================
def main():
    print("Loading dataset...")
    # 加载HuggingFace数据集
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)

    train_hf = dataset["train"]
    val_hf = dataset["val"]
    test_hf = dataset["test"]

    print(
        f"Train size: {len(train_hf)}, Val size: {len(val_hf)}, Test size: {len(test_hf)}"
    )

    # 初始化tokenizer
    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # 创建数据集
    train_dataset = CodeDataset(train_hf, tokenizer, MAX_LENGTH)
    val_dataset = CodeDataset(val_hf, tokenizer, MAX_LENGTH)

    # 构建标签映射（确保所有数据集使用相同映射）
    label2id = train_dataset.label2id
    id2label = train_dataset.id2label

    print(f"Number of classes: {len(label2id)}")
    print(f"Label mapping: {label2id}")

    # 初始化模型
    model = MARGINModel(
        num_classes=len(label2id),
        backbone=MODEL_NAME,
        embedding_dim=EMBEDDING_DIM,
    )

    # 训练
    trainer = Trainer(model, train_dataset, val_dataset, label2id, id2label)
    trained_model = trainer.train()

    # 最终测试
    print("\n" + "=" * 50)
    print("Final Testing")
    print("=" * 50)
    test_dataset = CodeDataset(test_hf, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 使用训练集重新计算几何中位数用于测试
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    trainer.compute_geometric_median_prototypes(train_loader)

    _, test_metrics = trainer.evaluate(test_loader, epoch="final", save_prefix="test")

    print("\nFinal Test Results:")
    print(f"Binary MCC: {test_metrics['binary']['mcc']:.4f}")
    print(f"Global Macro F1: {test_metrics['global_macro']['f1']:.4f}")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
