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

        # 找出Non-vul的索引
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

        # ========== 新增：运行时统计量 ==========
        self.running_feature_sums = torch.zeros(
            self.num_classes, EMBEDDING_DIM, device=DEVICE
        )
        self.class_counts = torch.zeros(self.num_classes, device=DEVICE)
        # 初始化 geometric_medians 为 weight prototypes（归一化）
        with torch.no_grad():
            weight_protos = self.model.get_weight_prototypes().detach()
            self.geometric_medians = F.normalize(weight_protos, dim=1).clone()
        # 初始化 kappas 为 1.0
        self.kappas = {i: 1.0 for i in range(self.num_classes)}

    def reset_running_stats(self):
        """每个 epoch 开始前重置统计量"""
        self.running_feature_sums.zero_()
        self.class_counts.zero_()

    def update_running_stats(self, features: torch.Tensor, labels: torch.Tensor):
        """在训练 batch 中更新 running feature sums 和 counts"""
        features = F.normalize(features.detach(), dim=1)  # 球面特征
        for i in range(len(labels)):
            label = labels[i].item()
            self.running_feature_sums[label] += features[i]
            self.class_counts[label] += 1

    def finalize_epoch_stats(self):
        """在 epoch 结束时，用 running stats 计算 kappas 和 geometric medians"""
        # 更新 geometric medians ≈ normalized class means
        new_geometric_medians = torch.zeros_like(self.geometric_medians)
        new_kappas = {}

        for class_idx in range(self.num_classes):
            count = self.class_counts[class_idx].item()
            if count > 0:
                mean_vec = self.running_feature_sums[class_idx] / count
                norm_mean = torch.norm(mean_vec).item()
                # 几何中位数近似为归一化均值方向
                new_geometric_medians[class_idx] = F.normalize(mean_vec, dim=0)
                # 计算 kappa
                kappa = compute_vmf_kappa(torch.tensor(norm_mean), EMBEDDING_DIM)
                kappa = max(kappa, MIN_KAPPA)
                new_kappas[class_idx] = kappa
            else:
                # 如果该类本轮无样本，保留上一轮值（或初始化值）
                new_geometric_medians[class_idx] = self.geometric_medians[class_idx]
                new_kappas[class_idx] = self.kappas.get(class_idx, 1.0)

        self.geometric_medians = new_geometric_medians
        self.kappas = new_kappas

        # 更新 criterion
        self.criterion.update_kappas(self.kappas)
        margins = self.compute_adaptive_margins(self.kappas)
        scales = self.compute_adaptive_scales(self.kappas)
        self.criterion.update_margins(margins)
        self.criterion.update_scales(scales)

    def compute_adaptive_margins(self, kappas):
        """保持不变"""
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
            margin_rad = min(max_margin, math.pi)
            margins[i] = margin_rad
        return margins

    def compute_adaptive_scales(self, kappas: dict):
        """保持不变"""
        scales = {}
        if not kappas:
            return scales
        kappa_mean = sum(kappas.values()) / len(kappas)
        for i in range(self.num_classes):
            kappa = kappas.get(i, kappa_mean)
            if kappa <= 1e-6:
                kappa = 1e-6
            scale = BASE_SCALE * (kappa_mean / kappa)
            scales[i] = float(scale)
        return scales

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch，并在线更新统计量"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 重置 running stats
        self.reset_running_stats()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            self.optimizer.zero_grad()

            with autocast(DEVICE):
                cos_theta, features = self.model(
                    input_ids, attention_mask, return_features=True
                )
                loss = self.criterion(cos_theta, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ========== 关键：更新 running stats ==========
            self.update_running_stats(features, labels)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ========== 在 epoch 结束时 finalize stats ==========
        self.finalize_epoch_stats()

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
        self.visualize_epoch(all_features, all_truth_label_idx, epoch)

        return avg_loss, metrics

    def visualize_epoch(self, features, labels, epoch):
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
            g = torch.Generator()
            g.manual_seed(SEED)
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=g,
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

            # 打印当前 margin 和 kappa（来自上一轮 finalize 的结果，首轮是初始值）
            print("\nClass-wise Kappa and Margin:")
            for i in range(self.num_classes):
                margin = (
                    self.criterion.margins[i]
                    if hasattr(self.criterion, "margins")
                    else 0.0
                )
                scale = (
                    self.criterion.scales[i]
                    if hasattr(self.criterion, "scales")
                    else BASE_SCALE
                )
                print(
                    f"  {self.id2label[i]}: κ={self.kappas[i]:.2f}, m={margin:.4f}, s={scale:.2f}"
                )

            # 训练（内部已更新 stats）
            train_loss = self.train_epoch(train_loader, epoch)
            print(f"\nTrain Loss: {train_loss:.4f}")

            # 注意：geometric_medians 已在 train_epoch 结尾更新，可直接用于 evaluate
            val_loss, metrics = self.evaluate(val_loader, epoch)
            print(f"Val Loss: {val_loss:.4f}")

            # 早停逻辑不变
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "kappas": self.kappas.copy(),
                    "margins": (
                        {i: self.criterion.margins[i] for i in range(self.num_classes)}
                        if hasattr(self.criterion, "margins")
                        else {}
                    ),
                }
                print("Model improved, saved checkpoint.")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )
                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    break

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
    trainer.train()


if __name__ == "__main__":
    main()
