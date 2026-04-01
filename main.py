import json
import math
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap
from datasets import load_dataset
from scipy.optimize import minimize
from scipy.special import erfinv
from scipy.stats import chi2, vonmises_fisher
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

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
SCALE_FACTOR = 30.0  # s
CONFIDENCE_ALPHA = 0.95  # α
MIN_KAPPA = 1.0  # 防止kappa过小导致数值不稳定

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


# ==================== 工具函数 ====================
def compute_vmf_kappa(mean_resultant_length, dim):
    """
    计算vMF分布的kappa参数 (MLE近似)
    mean_resultant_length: ||r_bar||, 必须在[0,1]之间
    dim: 维度d
    """
    r = mean_resultant_length
    # 防止数值问题
    r = torch.clamp(r, 0.0, 0.9999)
    print("mean resultant length:", r)

    numerator = r * (dim - r**2)
    denominator = 1 - r**2

    kappa = numerator / denominator
    return torch.clamp(kappa, min=MIN_KAPPA) 


def compute_pairwise_margin(kappa_i, kappa_j, dim, alpha=0.95):
    """
    计算两个类别之间的自适应margin Δm_{i,j}
    返回弧度值
    """
    term_i = math.sqrt(1 / kappa_i)
    term_j = math.sqrt(1 / kappa_j)
    
    q = chi2.ppf(alpha, df=dim)
    sqrt_q = math.sqrt(q)  # 关键修正！
    
    return 0.5 * sqrt_q * (term_i + term_j)  # 返回弧度


def compute_geometric_median(
    features, weights=None, max_iter=GEOMEDIAN_MAX_ITER, tol=GEOMEDIAN_TOL
):
    """
    在单位超球面上计算几何中位数
    features: [N, D] 已归一化的特征
    weights: [N] 可选权重
    返回: [D] 几何中位数（已归一化）
    """
    if weights is None:
        weights = torch.ones(features.shape[0], device=features.device)

    # 初始化为均值
    median = torch.sum(features * weights.unsqueeze(1), dim=0) / torch.sum(weights)
    median = F.normalize(median.unsqueeze(0), p=2, dim=1).squeeze(0)

    for _ in range(max_iter):
        # 计算球面距离 (余弦相似度转角度)
        cos_sim = torch.matmul(features, median)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        distances = torch.acos(cos_sim) + 1e-8  # 避免除零

        # 球面上的Weiszfeld算法
        w = weights / distances
        new_median = torch.sum(features * w.unsqueeze(1), dim=0) / torch.sum(w)
        new_median = F.normalize(new_median.unsqueeze(0), p=2, dim=1).squeeze(0)

        # 检查收敛
        diff = 1 - torch.dot(median, new_median)  # 余弦距离
        median = new_median

        if diff < tol:
            break

    return median


# ==================== 数据集类 ====================
class CodeDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=MAX_LENGTH):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 构建标签映射
        self.label2id = {}
        self.id2label = {}
        self._build_label_mapping()

    def _build_label_mapping(self):
        labels = sorted(set(self.dataset["label"]))

        # 把 Non-vul 放到最前面
        if "Non-vul" in labels:
            labels.remove("Non-vul")
            labels.insert(0, "Non-vul")

        for idx, label in enumerate(labels):
            self.label2id[label] = idx
            self.id2label[idx] = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        code = item["source"]
        label = item["label"]

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": self.label2id[label],
            "raw_label": label,
        }


# ==================== 模型定义 ====================
class SphereFaceClassifier(nn.Module):
    def __init__(self, num_classes, embedding_dim=EMBEDDING_DIM, scale=SCALE_FACTOR):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        self.scale = scale
        self.num_classes = num_classes

        # Weight prototypes (已归一化)
        self.weight_prototypes = nn.Parameter(
            F.normalize(torch.randn(num_classes, embedding_dim), p=2, dim=1)
        )

        # 冻结roberta的部分层（可选优化）
        # for param in self.roberta.embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # 提取CLS token特征
        features = outputs.last_hidden_state[:, 0, :]  # [B, D]
        # L2归一化到单位超球面
        features = F.normalize(features, p=2, dim=1)  # [B, D]

        # 计算余弦相似度 (球面内积)
        cos_theta = torch.matmul(features, self.weight_prototypes.t())  # [B, C]

        if return_features:
            return cos_theta, features
        return cos_theta

    def get_weight_prototypes(self):
        """返回归一化的weight prototypes"""
        return F.normalize(self.weight_prototypes, p=2, dim=1)


# ==================== ArcFace Loss with Adaptive Margin ====================
class AdaptiveArcFaceLoss(nn.Module):
    def __init__(self, num_classes, scale=SCALE_FACTOR):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.register_buffer("margins", torch.zeros(num_classes))
        self.register_buffer("kappas", torch.ones(num_classes) * MIN_KAPPA)

    def update_margins(self, margins_dict):
        """更新各类别的margin，margins_dict: {class_idx: margin_value}"""
        for idx, margin in margins_dict.items():
            self.margins[idx] = margin

    def update_kappas(self, kappas_dict):
        """更新各类别的kappa，kappas_dict: {class_idx: kappa_value}"""
        for idx, kappa in kappas_dict.items():
            self.kappas[idx] = kappa

    def forward(self, cos_theta, labels):
        """
        cos_theta: [B, C] 余弦相似度
        labels: [B] 类别索引
        """
        batch_size = cos_theta.size(0)

        # 获取对应类别的margin
        margins_batch = self.margins[labels]  # [B]

        # 计算 theta = arccos(cos_theta)，需要数值稳定性
        cos_theta_clamped = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta_clamped)

        # 应用margin: cos(theta + m)
        # 使用cos加法公式: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        cos_m = torch.cos(margins_batch)
        sin_m = torch.sin(margins_batch)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta_clamped, 2))

        # 只对正类应用margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        cos_theta_plus_m = cos_theta_clamped * cos_m.unsqueeze(
            1
        ) - sin_theta * sin_m.unsqueeze(1)

        # 混合：正类用cos(theta+m)，负类保持cos(theta)
        output = cos_theta_clamped * (1.0 - one_hot) + cos_theta_plus_m * one_hot

        # 缩放
        output = output * self.scale

        # 交叉熵
        loss = F.cross_entropy(output, labels)

        return loss


# ==================== 评估指标计算 ====================
def compute_metrics(truth_label_idx, pred_label_idx, idx2label: dict):
    """
    计算各类评估指标
    假设：idx2label 中索引 0 为 'Non-vul' (负例)，其余为正例
    返回：包含 global_macro, positive_macro, binary 及各类别详细 TP/FP/TN/FN 的 metrics 字典
    """
    all_label_idx = list(range(len(idx2label)))
    metrics = {}

    # 1. 计算全局混淆矩阵 (用于后续所有 TP/FP/TN/FN 的计算)
    cm = confusion_matrix(truth_label_idx, pred_label_idx, labels=all_label_idx)

    # 辅助函数：根据多分类混淆矩阵计算特定类别的 TP, FP, TN, FN
    def get_class_confusion_values(cm, class_idx):
        tp = cm[class_idx, class_idx]
        fn = np.sum(cm[class_idx, :]) - tp  # 真实为该类别，但预测为其他
        fp = np.sum(cm[:, class_idx]) - tp  # 预测为该类别，但真实为其他
        tn = np.sum(cm) - tp - fn - fp
        return int(tp), int(fp), int(tn), int(fn)

    # 辅助函数：计算二值化标签的指标
    def compute_binary_metrics(y_true_binary, y_pred_binary):
        return {
            "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "mcc": (
                matthews_corrcoef(y_true_binary, y_pred_binary)
                if sum(y_true_binary) > 0
                else 0.0
            ),
        }

    # =========================================================================
    # 2. Global Macro (所有类别的平均指标 + 每个类别的详细计数)
    # =========================================================================
    metrics["global_macro"] = {
        "mcc": matthews_corrcoef(truth_label_idx, pred_label_idx),
        "f1": f1_score(
            truth_label_idx,
            pred_label_idx,
            average="macro",
            labels=all_label_idx,
            zero_division=0,
        ),
        "precision": precision_score(
            truth_label_idx,
            pred_label_idx,
            average="macro",
            labels=all_label_idx,
            zero_division=0,
        ),
        "recall": recall_score(
            truth_label_idx,
            pred_label_idx,
            average="macro",
            labels=all_label_idx,
            zero_division=0,
        ),
        "accuracy": accuracy_score(truth_label_idx, pred_label_idx),
        "per_class": {},  # 初始化每个类别的详情
    }

    # 计算全局的 FNR/FPR 平均值
    fnr_list = []
    fpr_list = []

    for label_idx in all_label_idx:
        tp, fp, tn, fn = get_class_confusion_values(cm, label_idx)
        label_name = idx2label[label_idx]

        # 计算当前类的 FNR/FPR
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_list.append(fnr)
        fpr_list.append(fpr)

        # 【关键修复】先将标签二值化，再计算指标
        y_true_binary = [1 if y == label_idx else 0 for y in truth_label_idx]
        y_pred_binary = [1 if y == label_idx else 0 for y in pred_label_idx]

        support = tp + fn
        binary_metrics = (
            compute_binary_metrics(y_true_binary, y_pred_binary)
            if support > 0
            else {"f1": 0.0, "precision": 0.0, "recall": 0.0, "mcc": 0.0}
        )

        # 保存每个类别的详细指标和计数
        metrics["global_macro"]["per_class"][label_name] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "support": int(support),
            **binary_metrics,
        }

    metrics["global_macro"]["fnr"] = float(np.mean(fnr_list))
    metrics["global_macro"]["fpr"] = float(np.mean(fpr_list))

    # =========================================================================
    # 3. Positive Macro (排除 Non-vul，关注正样本内部的区分能力)
    # =========================================================================
    positive_label_idx = [l for l in all_label_idx if l != 0]
    metrics["positive_macro"] = {"per_class": {}}

    if len(positive_label_idx) > 0:
        # ---------------------------------------------------------------------
        # 3.1 计算 Positive Macro MCC (修正版)
        # ---------------------------------------------------------------------
        # 逻辑：只在真实为正样本的数据中，计算每个类别的 One-vs-Rest MCC，然后取平均
        
        # 1. 筛选出所有真实为正样本的索引
        pos_indices = [i for i, y in enumerate(truth_label_idx) if y in positive_label_idx]
        
        if len(pos_indices) > 0:
            y_true_pos_subset = np.array([truth_label_idx[i] for i in pos_indices])
            y_pred_pos_subset = np.array([pred_label_idx[i] for i in pos_indices])
            
            pos_mcc_scores = []
            
            # 2. 对每个正样本类别计算 One-vs-Rest MCC
            for label_idx in positive_label_idx:
                # 二值化：当前类为 1，其他正样本类为 0
                y_true_binary = (y_true_pos_subset == label_idx).astype(int)
                y_pred_binary = (y_pred_pos_subset == label_idx).astype(int)
                
                # 如果该类别在测试集中存在，则计算 MCC
                if np.sum(y_true_binary) > 0:
                    # 注意：这里不需要 zero_division 处理，因为 y_true 肯定有值
                    # 但如果 y_pred 全为0或全为1，sklearn 可能会报错，需捕获
                    try:
                        mcc = matthews_corrcoef(y_true_binary, y_pred_binary)
                        pos_mcc_scores.append(mcc)
                    except ValueError:
                        # 处理极端情况（如预测结果只有一类）
                        pos_mcc_scores.append(0.0)
                else:
                    # 如果该类别没有真实样本，跳过或记为0
                    pass
            
            # 3. 取平均
            metrics["positive_macro"]["mcc"] = float(np.mean(pos_mcc_scores)) if pos_mcc_scores else 0.0
        else:
            metrics["positive_macro"]["mcc"] = 0.0

        # ---------------------------------------------------------------------
        # 3.2 计算其他指标 (F1, Precision, Recall) - 保持原有逻辑即可
        # ---------------------------------------------------------------------
        metrics["positive_macro"]["f1"] = f1_score(
            truth_label_idx,
            pred_label_idx,
            average="macro",
            labels=positive_label_idx,
            zero_division=0,
        )
        metrics["positive_macro"]["precision"] = precision_score(
            truth_label_idx,
            pred_label_idx,
            average="macro",
            labels=positive_label_idx,
            zero_division=0,
        )
        metrics["positive_macro"]["recall"] = recall_score(
            truth_label_idx,
            pred_label_idx,
            average="macro",
            labels=positive_label_idx,
            zero_division=0,
        )

        # ---------------------------------------------------------------------
        # 3.3 计算每个正例类别的详细指标
        # ---------------------------------------------------------------------
        for label_idx in positive_label_idx:
            tp, fp, tn, fn = get_class_confusion_values(cm, label_idx)
            label_name = idx2label[label_idx]
            support = tp + fn

            y_true_binary = [1 if y == label_idx else 0 for y in truth_label_idx]
            y_pred_binary = [1 if y == label_idx else 0 for y in pred_label_idx]

            if support > 0:
                binary_metrics = compute_binary_metrics(y_true_binary, y_pred_binary)
                metrics["positive_macro"]["per_class"][label_name] = {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "support": int(support),
                    **binary_metrics,
                }
    else:
        metrics["positive_macro"].update(
            {"mcc": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
        )

    # =========================================================================
    # 4. Binary 指标 (Non-vul (0) vs CWE-* (Rest))
    # =========================================================================
    # 0 为 Negative, 其他为 Positive
    binary_truth_label_idx = [0 if y == 0 else 1 for y in truth_label_idx]
    binary_pred_label_idx = [0 if y == 0 else 1 for y in pred_label_idx]

    # 计算二分类的 TP, FP, TN, FN
    tp_bin = sum(
        1
        for t, p in zip(binary_truth_label_idx, binary_pred_label_idx)
        if t == 1 and p == 1
    )
    tn_bin = sum(
        1
        for t, p in zip(binary_truth_label_idx, binary_pred_label_idx)
        if t == 0 and p == 0
    )
    fp_bin = sum(
        1
        for t, p in zip(binary_truth_label_idx, binary_pred_label_idx)
        if t == 0 and p == 1
    )
    fn_bin = sum(
        1
        for t, p in zip(binary_truth_label_idx, binary_pred_label_idx)
        if t == 1 and p == 0
    )

    metrics["binary"] = {
        "tp": int(tp_bin),
        "fp": int(fp_bin),
        "tn": int(tn_bin),
        "fn": int(fn_bin),
        "support_positive": int(tp_bin + fn_bin),
        "support_negative": int(tn_bin + fp_bin),
        "mcc": matthews_corrcoef(binary_truth_label_idx, binary_pred_label_idx),
        "f1": f1_score(binary_truth_label_idx, binary_pred_label_idx, zero_division=0),
        "precision": precision_score(
            binary_truth_label_idx, binary_pred_label_idx, zero_division=0
        ),
        "recall": recall_score(
            binary_truth_label_idx, binary_pred_label_idx, zero_division=0
        ),
        "accuracy": accuracy_score(binary_truth_label_idx, binary_pred_label_idx),
    }

    return metrics


# ==================== 训练器 ====================
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, label2id, id2label):
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

        self.criterion = AdaptiveArcFaceLoss(self.num_classes).to(DEVICE)
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

            # 转换角度为弧度（因为ArcFace使用cos，内部使用弧度）
            # 但compute_adaptive_margin返回的已经是角度值，需要转换为弧度
            margin_rad = max_margin

            # 限制margin范围，避免过大
            margin_rad = min(margin_rad, math.pi)  # 最大45度

            margins[i] = margin_rad

        return margins

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
                median = compute_geometric_median(feats)
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
            f"Binary - MCC: {metrics['binary']['mcc']:.4f}, F1: {metrics['binary']['f1']:.4f}, "
            f"Prec: {metrics['binary']['precision']:.4f}, Rec: {metrics['binary']['recall']:.4f}"
        )

        if "positive_macro" in metrics:
            print(
                f"Positive-Macro - MCC: {metrics['positive_macro']['mcc']:.4f}, "
                f"F1: {metrics['positive_macro']['f1']:.4f}"
            )

        print(
            f"Global-Macro - MCC: {metrics['global_macro']['mcc']:.4f}, "
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

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                sim_matrix,
                annot=True,
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
            mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                sim_matrix,
                mask=mask,
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
            n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=42
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

            # 打印当前margin和kappa
            print("\nClass-wise Kappa and Margin:")
            for i in range(self.num_classes):
                print(f"  {self.id2label[i]}: κ={kappas[i]:.2f}, m={margins[i]}")

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
    train_dataset = CodeDataset(train_hf, tokenizer)
    val_dataset = CodeDataset(val_hf, tokenizer)

    # 构建标签映射（确保所有数据集使用相同映射）
    label2id = train_dataset.label2id
    id2label = train_dataset.id2label

    # 更新val和test的标签映射以匹配train
    for label in val_hf["label"]:
        if label not in label2id:
            print(f"Warning: Label {label} in val but not in train")

    print(f"Number of classes: {len(label2id)}")
    print(f"Label mapping: {label2id}")

    # 初始化模型
    model = SphereFaceClassifier(num_classes=len(label2id))

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
