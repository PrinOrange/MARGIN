from collections import Counter
import os
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.manifold import TSNE
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import warnings
from utils.seed import set_seed

warnings.filterwarnings("ignore")

# ==================== 配置常量 ====================
# 数据集配置
DATASET_NAME = "codemetic/MARGIN"
DATASET_SUBSET = "debug"  # 可选: diversevul, bigvul, megavul
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 64
NUM_WORKERS = 4

# 模型配置
MODEL_NAME = "microsoft/graphcodebert-base"  # 或 "roberta-base"
HIDDEN_DIM = 768
SCALE_S = 30.0
ALPHA_CONFIDENCE = 0.9

# 训练配置
MAX_EPOCHS = 200
EARLY_STOP_PATIENCE = MAX_EPOCHS
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 输出配置
OUTPUT_DIR = f"./outputs/{MODEL_NAME.split('/')[1]}-{DATASET_SUBSET}"
REPORT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "report")
PROTOTYPE_ALIGNMENT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prototype-alignment")
PROTOTYPE_DISPERSION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prototype-dispersion")
UMAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "prototype-umap")
MARGIN_HEATMAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "margin_heatmap")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_ALIGNMENT_OUTPUT_DIR, exist_ok=True)
os.makedirs(PROTOTYPE_DISPERSION_OUTPUT_DIR, exist_ok=True)
os.makedirs(UMAP_OUTPUT_DIR, exist_ok=True)
os.makedirs(MARGIN_HEATMAP_OUTPUT_DIR, exist_ok=True)


# ==================== 数据集类 ====================
class VulnerabilityDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source = str(item["source"])
        label = item["label"]
        sample_id = item["id"]
        label_idx = item["label_idx"]

        encoding = self.tokenizer(
            source,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": label,
            "label_idx": label_idx,
            "id": sample_id,
        }


# ==================== 模型定义 ====================
class AdaptiveSphereModel(nn.Module):
    def __init__(self, model_name, num_classes, hidden_dim=768):
        super().__init__()
        self.config = RobertaConfig.from_pretrained(model_name)
        self.backbone = RobertaModel.from_pretrained(model_name, config=self.config)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 类别权重原型 (在超球面上)
        self.weight_prototypes = nn.Parameter(torch.randn(num_classes, hidden_dim))
        self.margin_matrix = torch.zeros(num_classes, num_classes, device=DEVICE)

        # 初始化权重原型
        self._init_prototypes()

    def _init_prototypes(self):
        """初始化权重原型并归一化到单位超球面"""
        with torch.no_grad():
            self.weight_prototypes.data = F.normalize(
                self.weight_prototypes.data, p=2, dim=1
            )

    def forward(self, input_ids, attention_mask):
        """提取归一化的CLS特征"""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # 获取CLS token的特征
        cls_features = outputs.last_hidden_state[:, 0, :]

        # L2归一化到单位超球面
        normalized_features = F.normalize(cls_features, p=2, dim=1)

        # 获取归一化的权重原型
        normalized_prototypes = F.normalize(self.weight_prototypes, p=2, dim=1)

        return normalized_features, normalized_prototypes

    def compute_kappa(self, prototypes):
        """计算每个类别的vMF分布Kappa值"""
        # prototypes: [num_classes, hidden_dim], 已归一化
        # 使用闭式近似计算kappa
        proto_norms = torch.norm(prototypes, p=2, dim=1)  # 应该接近1

        # kappa估计公式
        d = self.hidden_dim
        kappa = (proto_norms * (d - proto_norms**2)) / (1 - proto_norms**2 + 1e-8)

        # 确保kappa为正
        kappa = torch.clamp(kappa, min=1.0)

        return kappa

    def compute_adaptive_margin(self, kappa_i, kappa_j):
        """计算两个类别之间的自适应margin"""
        d = self.hidden_dim
        alpha = ALPHA_CONFIDENCE

        # margin公式
        margin = (
            0.5
            * math.sqrt(2 * math.log(1 / (1 - alpha)))
            * (math.sqrt((d - 1) / kappa_i) + math.sqrt((d - 1) / kappa_j))
        )

        margin = max(0.0, min(math.pi, margin))

        return margin

    def compute_loss(self, features, prototypes, labels, kappa_values):
        """计算自适应球面损失"""
        batch_size = features.size(0)
        num_classes = self.num_classes

        # 计算余弦相似度矩阵
        # features: [batch, dim], prototypes: [num_classes, dim]
        cosine_sim = torch.matmul(features, prototypes.T)  # [batch, num_classes]

        # 获取每个样本的标签索引
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=features.device)

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j:
                    margin = self.compute_adaptive_margin(
                        kappa_values[i].item(), kappa_values[j].item()
                    )
                    self.margin_matrix[i, j] = margin

        # 对正类应用margin
        theta_with_margin = cosine_sim.clone()
        for idx in range(batch_size):
            true_label = labels_tensor[idx]
            for j in range(num_classes):
                if j != true_label:
                    theta_with_margin[idx, true_label] -= self.margin_matrix[
                        true_label, j
                    ]

        # 应用scale
        logits = SCALE_S * theta_with_margin

        # 交叉熵损失
        loss = F.cross_entropy(logits, labels_tensor)

        return loss


# ==================== 评估函数 ====================
def compute_geometric_median(points: torch.Tensor) -> torch.Tensor:
    """计算几何中位原型，并进行L2归一化"""
    if len(points) == 0:
        return None
    device = points.device
    points_np = points.cpu().numpy()
    # 初始值
    median = np.median(points_np, axis=0)
    # Weiszfeld iteration
    for _ in range(100):
        distances = np.linalg.norm(points_np - median, axis=1, keepdims=True) + 1e-8
        weights = 1.0 / distances
        new_median = np.sum(points_np * weights, axis=0) / np.sum(weights)

        if np.linalg.norm(new_median - median) < 1e-6:
            break
        median = new_median
    # 转回 tensor
    median = torch.tensor(median, dtype=torch.float32, device=device)
    # L2 normalize
    median = F.normalize(median, p=2, dim=0)
    return median


def compute_metrics_binary(y_true, y_pred):
    """计算binary指标 (Non-vul为负, 其他为正)"""
    y_true_binary = [0 if y == "Non-vul" else 1 for y in y_true]
    y_pred_binary = [0 if y == "Non-vul" else 1 for y in y_pred]

    metrics = {
        "mcc": matthews_corrcoef(y_true_binary, y_pred_binary),
        "f1": f1_score(y_true_binary, y_pred_binary),
        "precision": precision_score(y_true_binary, y_pred_binary),
        "recall": recall_score(y_true_binary, y_pred_binary),
    }

    return metrics


def compute_metrics_positive_macro(y_true, y_pred, all_classes):
    """计算positive-macro指标 (只对CWE-*正例)"""
    positive_classes = [c for c in all_classes if c != "Non-vul"]

    y_true_pos = []
    y_pred_pos = []

    for yt, yp in zip(y_true, y_pred):
        if yt != "Non-vul":
            y_true_pos.append(yt)
            y_pred_pos.append(yp)

    if len(y_true_pos) == 0:
        return {
            c: {"mcc": 0, "f1": 0, "precision": 0, "recall": 0}
            for c in positive_classes
        }

    class_metrics = {}
    support = Counter(y_true_pos)
    for cls in positive_classes:
        y_true_binary = [1 if y == cls else 0 for y in y_true_pos]
        y_pred_binary = [1 if y == cls else 0 for y in y_pred_pos]

        try:
            mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        except:
            mcc = 0

        class_metrics[cls] = {
            "mcc": mcc,
            "support": support[cls],
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
        }

    macro_mcc = sum(class_metrics[c]["mcc"] for c in positive_classes) / len(
            positive_classes
        )
    macro_f1 = sum(class_metrics[c]["f1"] for c in positive_classes) / len(
        positive_classes
    )
    macro_precision = sum(
        class_metrics[c]["precision"] for c in positive_classes
    ) / len(positive_classes)
    macro_recall = sum(class_metrics[c]["recall"] for c in positive_classes) / len(
        positive_classes
    )

    return {
        "mcc": macro_mcc,
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "per_class": class_metrics,
    }


def compute_metrics_global_macro(y_true, y_pred, all_classes):
    """计算global-macro指标 (所有类别一视同仁)"""

    class_metrics = {}

    support = Counter(y_true)

    for cls in all_classes:
        y_true_binary = [1 if y == cls else 0 for y in y_true]
        y_pred_binary = [1 if y == cls else 0 for y in y_pred]

        try:
            mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
        except:
            mcc = 0

        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

        # FPR
        tn = sum(
            1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0
        )
        fp = sum(
            1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1
        )

        fpr = fp / (fp + tn + 1e-8)

        class_metrics[cls] = {
            "mcc": mcc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "support": support[cls],
            "FNR": 1 - recall,
            "FPR": fpr,
        }

    macro_mcc = sum(class_metrics[c]["mcc"] for c in all_classes) / len(all_classes)
    macro_f1 = sum(class_metrics[c]["f1"] for c in all_classes) / len(all_classes)
    macro_precision = sum(class_metrics[c]["precision"] for c in all_classes) / len(
        all_classes
    )
    macro_recall = sum(class_metrics[c]["recall"] for c in all_classes) / len(
        all_classes
    )
    macro_fnr = sum(class_metrics[c]["FNR"] for c in all_classes) / len(all_classes)
    macro_fpr = sum(class_metrics[c]["FPR"] for c in all_classes) / len(all_classes)

    return {
        "mcc": macro_mcc,
        "f1": macro_f1,
        "precision": macro_precision,
        "recall": macro_recall,
        "FNR": macro_fnr,
        "FPR": macro_fpr,
        "per_class": class_metrics,
    }


# ==================== 可视化函数 ====================
def plot_similarity_heatmap(
    similarity_matrix, labels, title, x_title, y_title, save_path
):
    """绘制相似度热力图 - 修复版"""
    n = len(labels)
    # 动态调整画布大小，确保标签可读
    size = max(8, n * 0.6)

    plt.figure(figsize=(size, size))

    # 决定是否显示数值标注 (类别太多时关闭避免拥挤)
    fmt = ".0f"
    annot_kws = {"size": 12}

    # 绘制热力图
    ax = sns.heatmap(
        similarity_matrix * 100,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        square=True,
        annot=True,
        fmt=fmt,
        annot_kws=annot_kws,
        vmin=-100,
        vmax=100,
        center=0.0,
    )

    # 美化标签
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    # 标题和轴标签
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_title, fontsize=11)
    plt.ylabel(y_title, fontsize=11)

    # 自动调整布局避免标签被截断
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_margin_heatmap(margin_matrix, labels, title, x_title, y_title, save_path):
    margin_matrix = margin_matrix.detach().cpu().numpy()
    n = len(labels)
    size = max(8, n * 0.6)
    plt.figure(figsize=(size, size))
    fmt = ".2f"
    annot_kws = {"size": 12}
    ax = sns.heatmap(
        margin_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        square=True,
        annot=True,
        fmt=fmt,
        annot_kws=annot_kws,
        vmin=0,
        vmax=np.pi,
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(x_title, fontsize=11)
    plt.ylabel(y_title, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_umap(features, labels, all_classes, save_path):
    """绘制UMAP降维图"""
    # 使用t-SNE替代UMAP (避免额外依赖)
    features_np = features.cpu().numpy()

    # 降维到2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    features_2d = tsne.fit_transform(features_np)

    plt.figure(figsize=(6, 5))

    # 颜色映射
    color_map = {}
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))
    for i, cls in enumerate(all_classes):
        if cls == "Non-vul":
            color_map[cls] = "gray"
        else:
            color_map[cls] = colors[i % len(colors)]

    # 先绘制正样本点
    for cls in all_classes:
        if cls != "Non-vul":
            mask = [l == cls for l in labels]
            if sum(mask) > 0:
                plt.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=[color_map[cls]],
                    label=cls,
                    alpha=0.8,
                    s=15,
                )

    # 再绘制负样本点
    mask_neg = [l == "Non-vul" for l in labels]
    if sum(mask_neg) > 0:
        plt.scatter(
            features_2d[mask_neg, 0],
            features_2d[mask_neg, 1],
            c="gray",
            label="Non-vul",
            alpha=0.6,
            s=20,
        )

    plt.title("t-SNE Visualization of Features", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==================== 主训练函数 ====================
def main():
    set_seed(42)

    print(f"Using device: {DEVICE}")
    print(f"Dataset: {DATASET_NAME}/{DATASET_SUBSET}")
    print(f"Model: {MODEL_NAME}")

    # 加载数据
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_SUBSET)

    train_data = dataset["train"]
    val_data = dataset["val"]
    test_data = dataset["test"]

    # 获取所有标签
    all_labels = list(set(train_data["label"]))
    all_labels.sort()
    num_classes = len(all_labels)
    label2idx = {label: idx for idx, label in enumerate(all_labels)}
    idx2label = {idx: label for label, idx in label2idx.items()}

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {all_labels}")

    # 标签编码
    train_data_encoded = train_data.map(
        lambda x: {"label_idx": label2idx[x["label"]]}, remove_columns=[]
    )
    val_data_encoded = val_data.map(
        lambda x: {"label_idx": label2idx[x["label"]]}, remove_columns=[]
    )

    # 初始化tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # 创建数据集
    train_dataset = VulnerabilityDataset(train_data_encoded, tokenizer, MAX_SEQ_LENGTH)
    val_dataset = VulnerabilityDataset(val_data_encoded, tokenizer, MAX_SEQ_LENGTH)

    # 初始化模型
    model = AdaptiveSphereModel(MODEL_NAME, num_classes, HIDDEN_DIM)
    model = model.to(DEVICE)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 混合精度训练
    scaler = GradScaler()

    # 早停机制
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # 训练循环
    print("Starting training...")
    for epoch in range(MAX_EPOCHS):
        g = torch.Generator()
        g.manual_seed(SEED)
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=g,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")

        for batch in train_pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label_idx"].to(DEVICE)
            label_names = batch["label"]

            optimizer.zero_grad()

            with autocast():
                features, prototypes = model(input_ids, attention_mask)

                # 计算kappa值
                kappa_values = model.compute_kappa(prototypes)

                # 计算损失
                loss = model.compute_loss(
                    features, prototypes, labels.cpu().numpy().tolist(), kappa_values
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            num_batches += 1

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / num_batches

        # ========== 验证阶段 ==========
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        all_features = []
        all_prototypes = []
        all_labels_val = []
        all_preds = []
        all_label_names = []

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Val]")

        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label_idx"].to(DEVICE)
                label_names = batch["label"]

                with autocast():
                    features, prototypes = model(input_ids, attention_mask)

                    kappa_values = model.compute_kappa(prototypes)

                    loss = model.compute_loss(
                        features,
                        prototypes,
                        labels.cpu().numpy().tolist(),
                        kappa_values,
                    )

                val_loss += loss.item()
                num_val_batches += 1

                # 收集特征和预测
                all_features.append(features.cpu())
                all_prototypes.append(prototypes.cpu())

                # 计算预测
                cosine_sim = torch.matmul(features, prototypes.T)
                preds = torch.argmax(cosine_sim, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_label_names.extend(label_names)
                all_labels_val.extend(labels.cpu().numpy().tolist())

                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / num_val_batches

        # ========== 计算几何中位原型 ==========
        # 在训练集上计算每个类别的几何中位原型
        model.eval()
        class_features = {label: [] for label in all_labels}

        with torch.no_grad():
            for batch in tqdm(
                train_loader, desc="Computing geometric median prototypes"
            ):
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                label_names = batch["label"]

                features, _ = model(input_ids, attention_mask)

                for feat, label in zip(features.cpu(), label_names):
                    class_features[label].append(feat)

        geometric_median_prototypes = {}
        for label in all_labels:
            if len(class_features[label]) > 0:
                points = torch.stack(class_features[label])
                geometric_median_prototypes[label] = compute_geometric_median(points)
            else:
                geometric_median_prototypes[label] = None

        # ========== 使用几何中位原型进行预测 ==========
        geo_median_preds = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                features, _ = model(input_ids, attention_mask)

                # 计算到每个几何中位原型的距离
                batch_preds = []
                for feat in features:
                    min_dist = float("inf")
                    pred_label = all_labels[0]

                    for label, geo_proto in geometric_median_prototypes.items():
                        if geo_proto is not None:
                            dist = torch.norm(feat - geo_proto.to(DEVICE))
                            if dist < min_dist:
                                min_dist = dist
                                pred_label = label

                    batch_preds.append(pred_label)

                geo_median_preds.extend(batch_preds)

        # ========== 计算评估指标 ==========
        pred_labels = [idx2label[p] for p in all_preds]

        # Binary指标
        binary_metrics = compute_metrics_binary(all_label_names, pred_labels)

        # Positive-macro指标
        positive_macro_metrics = compute_metrics_positive_macro(
            all_label_names, pred_labels, all_labels
        )

        # Global-macro指标
        global_macro_metrics = compute_metrics_global_macro(
            all_label_names, pred_labels, all_labels
        )

        # ========== 保存评估结果 ==========
        eval_results = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "binary_metrics": binary_metrics,
            "positive_macro_metrics": positive_macro_metrics,
            "global_macro_metrics": global_macro_metrics,
        }

        eval_path = os.path.join(
            REPORT_OUTPUT_DIR, f"eval_results_epoch_{epoch+1}.json"
        )
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)

        # ========== 可视化 ==========
        # 1. 几何中位原型相似度热力图
        geo_proto_labels = []
        for label in all_labels:
            if geometric_median_prototypes[label] is not None:
                geo_proto_labels.append(label)

        if len(geo_proto_labels) > 0:
            geo_protos = torch.stack(
                [geometric_median_prototypes[l] for l in geo_proto_labels]
            )
            geo_sim = torch.matmul(geo_protos, geo_protos.T).detach().numpy()

            plot_similarity_heatmap(
                geo_sim,
                geo_proto_labels,
                f"Geometric Median Prototype Similarity (Epoch {epoch+1})",
                "Classes",
                "Classes",
                os.path.join(
                    PROTOTYPE_DISPERSION_OUTPUT_DIR,
                    f"geo_median_heatmap_epoch_{epoch+1}.svg",
                ),
            )

        # 2. Weight prototype 和 geometric median prototype 相似度热力图
        if len(geo_proto_labels) > 0:
            weight_protos = F.normalize(model.weight_prototypes, p=2, dim=1).cpu()
            geo_protos_tensor = torch.stack(
                [geometric_median_prototypes[l] for l in geo_proto_labels]
            )
            cross_sim = (
                torch.matmul(weight_protos, geo_protos_tensor.T).detach().numpy()
            )

            plot_similarity_heatmap(
                cross_sim,
                geo_proto_labels,
                "Weight Prototype",
                "Geometric Median Prototype",
                f"Weight vs Geometric Median Prototype Similarity (Epoch {epoch+1})",
                os.path.join(
                    PROTOTYPE_ALIGNMENT_OUTPUT_DIR,
                    f"weight_vs_geo_heatmap_epoch_{epoch+1}.svg",
                ),
            )
        if len(geo_proto_labels) > 0:
            plot_margin_heatmap(
                model.margin_matrix,
                geo_proto_labels,
                f"Adaptive Margin Matrix (Epoch {epoch+1})",
                "Classes",
                "Classes",
                os.path.join(
                    MARGIN_HEATMAP_OUTPUT_DIR, f"margin_heatmap_epoch_{epoch+1}.svg"
                ),
            )

        # 3. UMAP/t-SNE图
        all_features_concat = torch.cat(all_features, dim=0)
        all_label_names_flat = all_label_names

        plot_umap(
            all_features_concat,
            all_label_names_flat,
            all_labels,
            os.path.join(UMAP_OUTPUT_DIR, f"umap_epoch_{epoch+1}.svg"),
        )

        # ========== 早停检查 ==========
        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}"
        )
        print(
            f"Binary F1: {binary_metrics['f1']:.4f}, MCC: {binary_metrics['mcc']:.4f}"
        )
        print(
            f"Positive Macro F1: {positive_macro_metrics['f1']:.4f}, MCC: {positive_macro_metrics['mcc']:.4f}"
        )
        print(
            f"Global Macro F1: {global_macro_metrics['f1']:.4f}, MCC: {global_macro_metrics['mcc']:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # 保存最优模型
        # if best_model_state is not None:
        #     torch.save(best_model_state, os.path.join(OUTPUT_DIR, 'best_model.pt'))

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
