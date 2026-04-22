import json
import os
import warnings
from datetime import datetime
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.seed import set_seed
from utils.dataset import CodeDataset
from utils.model import MARGINModel
from utils.math import (
    compute_geometric_median,
    compute_vmf_kappa,
)
from utils.string import print_dict_pipe
from utils.visualize import (
    draw_prototype_dispersion,
    draw_prototype_alignment,
    draw_umap,
)
from utils.evaluation import evaluate_model
from utils.logger import log

warnings.filterwarnings("ignore", category=UserWarning)

# ==================== 常量配置区 ====================
# 数据配置
DATASET_NAME = "codemetic/MARGIN"
DATASET_SUBSET = "megavul"  # 可选其他 subset
MAX_LENGTH = 512

# 模型配置
MODEL_NAME = "Salesforce/codet5-base"  # 可选其他 backbone
# microsoft/graphcodebert-base
# microsoft/unixcoder-base
# microsoft/unixcoder-base-nine
# Salesforce/codet5-base
EMBEDDING_DIM = 768  # graphcodebert-base 的维度

# 训练配置
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 200
EARLY_STOPPING_PATIENCE = MAX_EPOCHS

# ArcFace & 球面配置
BASE_SCALE = 20  # s
CONFIDENCE_ALPHA = 0.95  # α
SEED = 42

# 设备配置
DEVICE = "cuda:2"
TIME_PREFIX = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# 输出配置
OUTPUT_DIR = f"./output/{DATASET_SUBSET}-{MODEL_NAME.split('/')[1]}-{TIME_PREFIX}"
UMAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "umap")
PROTOTYPE_ALIGNMENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prototype-alignment")
PROTOTYPE_DISPERSION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prototype-dispersion")
REPORT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "report")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UMAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_ALIGNMENT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_DISPERSION_OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

# UMAP 配置
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1

set_seed(SEED)
log.set_log_file(os.path.join(OUTPUT_DIR, "train.log"))


# ==================== 训练器 ====================
class Trainer:
    def __init__(self, model: MARGINModel):
        self.model = model.to(DEVICE)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.scaler = GradScaler()

        self.best_global_f1 = float("-inf")
        self.patience_counter = 0
        self.best_model_state = None

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 或者，如果显存允许，可以直接存 Tensor 列表
        feature_accumulator = {label: [] for label in range(self.model.num_classes)}

        # --- 记录开始时间 ---
        start_time = time.time()
        log.print(
            f"⏱️ Epoch {epoch} Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            label_idx = batch["label_idx"].to(DEVICE)

            self.optimizer.zero_grad()
            with autocast(DEVICE):
                cos_theta, features = self.model(
                    input_ids, attention_mask, return_features=True
                )
                loss = self.model.loss_head(cos_theta, label_idx)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            features_cpu = features.detach().cpu()
            labels_cpu = label_idx.detach().cpu()
            for f, l in zip(features_cpu, labels_cpu):
                feature_accumulator[int(l)].append(f)

        # --- 统计阶段优化 ---

        D = features.shape[1]
        C = self.model.num_classes

        mean_prototypes_list = []
        geom_median_prototypes_list = []
        kappas_list = []
        class_counts_list = []

        with torch.no_grad():
            # 预分配张量列表，避免动态扩容
            for label_idx in range(C):
                feats = feature_accumulator[label_idx]

                # 1. 一次性堆叠，效率更高
                feats_tensor = torch.stack(feats, dim=0)  # [N, D]
                class_counts_list.append(len(feats))  # 记录实际参与训练的样本数

                # 2. 只进行一次归一化，复用结果
                # 注意：geometric median 和 kappa 计算通常都需要单位向量
                feats_tensor_norm = F.normalize(feats_tensor, p=2, dim=1)

                # Mean Prototype (在归一化前或后计算取决于你的定义，通常 Mean of Normals 是标准做法)
                mean_proto = F.normalize(feats_tensor_norm.mean(dim=0), dim=0)
                mean_prototypes_list.append(mean_proto)

                # Geometric Median Prototype
                geom_median_proto = compute_geometric_median(feats_tensor_norm)
                geom_median_proto = F.normalize(geom_median_proto, dim=0)
                geom_median_prototypes_list.append(geom_median_proto)

                # Kappa (复用上面的 feats_tensor_norm)
                kappa = compute_vmf_kappa(feats_tensor_norm, mean_proto)
                kappas_list.append(kappa)

            # 拼成 [C, D] 张量 (注意维度顺序，通常类别在前更方便索引)
            self.model.current_mean_prototypes = torch.stack(
                mean_prototypes_list, dim=0
            ).to(DEVICE)

            self.model.current_geometric_median_prototypes = torch.stack(
                geom_median_prototypes_list, dim=0
            ).to(DEVICE)

            self.model.class_counts = torch.tensor(class_counts_list).to(DEVICE)  # [C]
            self.model.current_kappas = torch.tensor(kappas_list).to(DEVICE)  # [C]

            # 我明明在这里面更新了 params 自适应参数
            self.model.loss_head.update_adaptive_params(
                self.model.current_kappas,
                self.model.class_counts,
                self.model.current_mean_prototypes,
            )

        # --- 记录结束时间和耗时 ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.print(
            f"✅ Epoch {epoch} Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        log.print(f"⏳ Epoch {epoch} Training costs {elapsed_time:.2f} seconds.")

        return total_loss / num_batches

    def evaluate_epoch(self, dataloader, epoch, save_prefix="val"):
        """评估模型"""
        self.model.eval()

        # --- 记录开始时间 ---
        start_time = time.time()
        log.print(
            f"⏱️  Epoch {epoch} Evaluation started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )

        (
            metrics,
            all_features,
            all_truth_label_idx,
            all_pred_label_idx,
            all_raw_labels,
            avg_loss,
        ) = evaluate_model(self.model, dataloader, f"Epoch {epoch} Evaluating", DEVICE)

        classification_metrics = metrics["classification_metrics"]
        clustering_metrics = metrics["clustering_metrics"]
        etf_metrics = metrics["etf_metrics"]
        statistics_metrics = metrics["statistics_metrics"]

        # 保存到 JSON
        json_path = os.path.join(
            REPORT_OUTPUT_DIR, f"{save_prefix}_metrics_epoch_{epoch}.json"
        )
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # 打印指标
        log.print(f"Epoch {epoch} Evaluation Results:")
        log.print("Classification Metrics: ---------------------------------")
        log.print(
            f"🐱 Binary - MCC: {classification_metrics['binary']['mcc']:.4f}, F1: {classification_metrics['binary']['f1']:.4f}, "
            f"Prec: {classification_metrics['binary']['precision']:.4f}, Rec: {classification_metrics['binary']['recall']:.4f}"
        )
        log.print(
            f"🐒 Positive-Macro - MCC: {classification_metrics['positive_macro']['mcc']:.4f}, "
            f"F1: {classification_metrics['positive_macro']['f1']:.4f}"
        )
        log.print(
            f"🌏 Global-Macro - MCC: {classification_metrics['global_macro']['mcc']:.4f}, "
            f"F1: {classification_metrics['global_macro']['f1']:.4f}, "
            f"FNR: {classification_metrics['global_macro']['fnr']:.4f}, "
            f"FPR: {classification_metrics['global_macro']['fpr']:.4f}"
        )

        log.print("Clustering Metrics: ---------------------------------")
        log.print(print_dict_pipe(clustering_metrics))

        log.print("ETF Metrics: ---------------------------------")
        log.print(print_dict_pipe(etf_metrics))

        log.print("Statistics Metrics: ---------------------------------")
        log.print(
            f"Margin - Mean: {statistics_metrics['summary']['margin_mean']:.4f}, Std: {statistics_metrics['summary']['margin_std']:.4f}"
        )
        log.print(
            f"Scale - Mean: {statistics_metrics['summary']['scale_mean']:.4f}, Std: {statistics_metrics['summary']['scale_std']:.4f}"
        )
        log.print(
            f"Kappa - Mean: {statistics_metrics['summary']['kappa_mean']:.4f}, Std: {statistics_metrics['summary']['kappa_std']:.4f}"
        )

        # 绘制可视化
        self.visualize_epoch(all_features, all_truth_label_idx, epoch)

        # --- 记录结束时间和耗时 ---
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.print(
            f"✅ Epoch {epoch} Evaluation finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        log.print(f"⏳ Epoch {epoch} Evaluation costs: {elapsed_time:.2f} seconds")

        return avg_loss, metrics

    def visualize_epoch(self, features, truth_label_idx, epoch):
        """绘制热力图和 UMAP"""

        # 1. 几何中位数原型相似度热力图
        draw_prototype_dispersion(
            self.model.current_geometric_median_prototypes,
            self.model.id2label,
            f"Epoch {epoch}",
            os.path.join(
                PROTOTYPE_DISPERSION_OUTPUT_DIR, f"geo_median_sim_epoch_{epoch}.svg"
            ),
        )

        # 2. Weight prototype 与 Geometric median prototype 相似度热力图
        draw_prototype_alignment(
            self.model.current_geometric_median_prototypes,
            self.model.get_norm_weight_prototypes(),
            self.model.id2label,
            f"Epoch {epoch}",
            os.path.join(
                PROTOTYPE_ALIGNMENT_OUTPUT_DIR, f"weight_geo_sim_epoch_{epoch}.svg"
            ),
        )

        # 3. UMAP 可视化
        draw_umap(
            features,
            truth_label_idx,
            self.model.id2label,
            f"UMAP Visualization - Epoch {epoch}",
            os.path.join(UMAP_OUTPUT_DIR, f"umap_epoch_{epoch}.svg"),
            UMAP_N_NEIGHBORS,
            UMAP_MIN_DIST,
            SEED,
        )

    def train(self):
        for epoch in range(0, MAX_EPOCHS + 1):
            log.print(f"\n{'='*50}")
            log.print(f"Epoch {epoch}/{MAX_EPOCHS}")
            log.print(f"{'='*50}")
            g = torch.Generator()
            g.manual_seed(SEED)
            train_loader = DataLoader(
                self.model.train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=g,
            )
            val_loader = DataLoader(
                self.model.val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
            )

            # 训练（内部已更新 stats）
            train_loss = self.train_epoch(train_loader, epoch)
            log.print(f"Train Loss: {train_loss:.4f}")

            # 注意：geometric_medians 已在 train_epoch 结尾更新，可直接用于 evaluate
            avg_val_loss, val_metrics = self.evaluate_epoch(val_loader, epoch)
            val_global_f1 = val_metrics["classification_metrics"]["global_macro"]["f1"]
            log.print(f"Val Loss: {avg_val_loss:.4f}")

            # 但是，现在我想改一下早停逻辑。
            # 第一，用 val_metrics["classification_metrics"]["positive_macro"]["mcc"]
            # 和 val_metrics["classification_metrics"]["binary"]["mcc"] 这两个指标的帕累托最优点作为监听依据。

            # 早停逻辑
            if val_global_f1 > self.best_global_f1:
                self.best_global_f1 = val_global_f1
                self.patience_counter = 0
                self.best_model_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_global_f1": val_global_f1,
                }
                log.print("Model improved, saved checkpoint.")
            else:
                self.patience_counter += 1
                log.print(
                    f"No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )
                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    log.print(f"Early stopping triggered at epoch {epoch}")
                    break
            if self.best_model_state is not None:
                best_epoch = self.best_model_state["epoch"]
                best_global_f1 = self.best_model_state["val_global_f1"]
                log.print(f"🏆 Current Best: Epoch {best_epoch} | Global F1 {best_global_f1:.4f}")
            else:
                log.print(
                    f"🏆 Current Best: Epoch {epoch} | Global F1 {val_global_f1:.4f}"
                )  # 第一轮的情况

        if self.best_model_state is not None:
            log.print(f"Loading best model from epoch {self.best_model_state['epoch']}")
            self.model.load_state_dict(self.best_model_state["model_state_dict"])

        return self.model


# ==================== 主函数 ====================
def main():
    log.print("Loading dataset...")
    # 加载 HuggingFace 数据集
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)

    train_hf = dataset["train"]
    val_hf = dataset["val"]
    test_hf = dataset["test"]

    log.print(
        f"Train size: {len(train_hf)}, Val size: {len(val_hf)}, Test size: {len(test_hf)}"
    )

    # 初始化 tokenizer
    log.print(f"Loading tokenizer and model: {MODEL_NAME}")

    # 创建数据集
    train_dataset = CodeDataset(MODEL_NAME, train_hf, MAX_LENGTH)
    # val_dataset = CodeDataset(MODEL_NAME, val_hf, MAX_LENGTH)
    test_dataset = CodeDataset(MODEL_NAME, test_hf, MAX_LENGTH)

    # 构建标签映射（确保所有数据集使用相同映射）
    label2id = train_dataset.label2idx

    log.print(f"Number of classes: {len(label2id)}")
    log.print(f"Label mapping: {label2id}")

    # 初始化模型
    model = MARGINModel(
        backbone=MODEL_NAME,
        base_scale=BASE_SCALE,
        alpha=CONFIDENCE_ALPHA,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # 训练
    trainer = Trainer(model)
    trainer.train()


if __name__ == "__main__":
    main()
