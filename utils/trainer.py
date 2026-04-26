import json
import os
import warnings
from datetime import datetime
import time
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
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


class TrainerConfig:
    """训练器配置类"""

    def __init__(
        self,
        batch_size: int,
        learning_rate: int,
        weight_decay: int,
        max_epochs: int,
        early_stopping_patience: int,
        output_dir: str,
        device: str,
        umap_n_neighbors: int,
        umap_min_dist: float,
        # 其他配置
        seed=int,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.output_dir = output_dir
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.seed = seed


class Trainer:
    def __init__(self, model: MARGINModel, config: TrainerConfig):
        self.config = config
        self.model = model.to(config.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler()

        self.best_global_f1 = float("-inf")
        self.patience_counter = 0
        self.best_model_state = None

        # 时间戳
        self.time_prefix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    def setup_output_dirs(self):
        """设置输出目录"""
        self.umap_output_dir = os.path.join(self.config.output_dir, "umap")
        self.prototype_alignment_output_dir = os.path.join(
            self.config.output_dir, "prototype-alignment"
        )
        self.prototype_dispersion_output_dir = os.path.join(
            self.config.output_dir, "prototype-dispersion"
        )
        self.report_output_dir = os.path.join(self.config.output_dir, "report")

        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.umap_output_dir, exist_ok=True)
        os.makedirs(self.prototype_alignment_output_dir, exist_ok=True)
        os.makedirs(self.prototype_dispersion_output_dir, exist_ok=True)
        os.makedirs(self.report_output_dir, exist_ok=True)

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
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            label_idx = batch["label_idx"].to(self.config.device)

            self.optimizer.zero_grad()
            with autocast(self.config.device):
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
            ).to(self.config.device)

            self.model.current_geometric_median_prototypes = torch.stack(
                geom_median_prototypes_list, dim=0
            ).to(self.config.device)

            self.model.class_counts = torch.tensor(class_counts_list).to(
                self.config.device
            )  # [C]
            self.model.current_kappas = torch.tensor(kappas_list).to(
                self.config.device
            )  # [C]

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
        ) = evaluate_model(
            self.model, dataloader, f"Epoch {epoch} Evaluating", self.config.device
        )

        classification_metrics = metrics["classification_metrics"]
        clustering_metrics = metrics["clustering_metrics"]
        etf_metrics = metrics["etf_metrics"]
        statistics_metrics = metrics["statistics_metrics"]

        # 保存到 JSON
        json_path = os.path.join(
            self.report_output_dir, f"{save_prefix}_metrics_epoch_{epoch}.json"
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
            f"F1: {classification_metrics['positive_macro']['f1']:.4f}, "
            f"Prec: {classification_metrics['positive_macro']['precision']:.4f}, "
            f"Rec: {classification_metrics['positive_macro']['recall']:.4f}, "
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
                self.prototype_dispersion_output_dir,
                f"geo_median_sim_epoch_{epoch}.svg",
            ),
        )

        # 2. Weight prototype 与 Geometric median prototype 相似度热力图
        draw_prototype_alignment(
            self.model.current_geometric_median_prototypes,
            self.model.get_norm_weight_prototypes(),
            self.model.id2label,
            f"Epoch {epoch}",
            os.path.join(
                self.prototype_alignment_output_dir, f"weight_geo_sim_epoch_{epoch}.svg"
            ),
        )

        # 3. UMAP 可视化
        draw_umap(
            features,
            truth_label_idx,
            self.model.id2label,
            f"UMAP Visualization - Epoch {epoch}",
            os.path.join(self.umap_output_dir, f"umap_epoch_{epoch}.svg"),
            self.config.umap_n_neighbors,
            self.config.umap_min_dist,
            self.config.seed,
        )

    def train(self, dataset_subset, model_name):
        """训练主流程"""
        self.setup_output_dirs()
        log.set_log_file(os.path.join(self.config.output_dir, "train.log"))

        for epoch in range(0, self.config.max_epochs + 1):
            log.print(f"\n{'='*50}")
            log.print(f"Epoch {epoch}/{self.config.max_epochs}")
            log.print(f"{'='*50}")
            g = torch.Generator()
            g.manual_seed(self.config.seed)
            train_loader = DataLoader(
                self.model.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=g,
            )
            val_loader = DataLoader(
                self.model.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
            )

            # 训练（内部已更新 stats）
            train_loss = self.train_epoch(train_loader, epoch)
            log.print(f"Train Loss: {train_loss:.4f}")

            # 注意：geometric_medians 已在 train_epoch 结尾更新，可直接用于 evaluate
            avg_val_loss, val_metrics = self.evaluate_epoch(val_loader, epoch)
            val_global_f1 = val_metrics["classification_metrics"]["global_macro"]["f1"]
            log.print(f"Val Loss: {avg_val_loss:.4f}")

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
                    f"No improvement. Patience: {self.patience_counter}/{self.config.early_stopping_patience}"
                )
                if self.patience_counter >= self.config.early_stopping_patience:
                    log.print(f"Early stopping triggered at epoch {epoch}")
                    break
            if self.best_model_state is not None:
                best_epoch = self.best_model_state["epoch"]
                best_global_f1 = self.best_model_state["val_global_f1"]
                log.print(
                    f"🏆 Current Best: Epoch {best_epoch} | Global F1 {best_global_f1:.4f}"
                )
            else:
                log.print(
                    f"🏆 Current Best: Epoch {epoch} | Global F1 {val_global_f1:.4f}"
                )  # 第一轮的情况

        if self.best_model_state is not None:
            log.print(f"Loading best model from epoch {self.best_model_state['epoch']}")
            self.model.load_state_dict(self.best_model_state["model_state_dict"])

        return self.model

    def get_best_model_state(self):
        """返回最佳模型状态"""
        return self.best_model_state

    def save_checkpoint(self, filepath):
        """保存检查点"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_global_f1": self.best_global_f1,
            "patience_counter": self.patience_counter,
            "best_model_state": self.best_model_state,
            "config": self.config.__dict__,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_global_f1 = checkpoint.get("best_global_f1", float("-inf"))
        self.patience_counter = checkpoint.get("patience_counter", 0)
        self.best_model_state = checkpoint.get("best_model_state", None)
        # 可以选择恢复配置信息
