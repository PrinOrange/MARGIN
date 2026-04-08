import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


# ==================== 评估指标计算 ====================
def compute_classification_metrics(truth_label_idx, pred_label_idx, idx2label: dict):
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
        pos_indices = [
            i for i, y in enumerate(truth_label_idx) if y in positive_label_idx
        ]

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
            metrics["positive_macro"]["mcc"] = (
                float(np.mean(pos_mcc_scores)) if pos_mcc_scores else 0.0
            )
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


def compute_clustering_metrics(truth_label_idx, pred_label_idx, features=None):
    truth_label_idx = np.array(truth_label_idx)
    pred_label_idx = np.array(pred_label_idx)

    clustering_metrics = {}

    clustering_metrics["nmi"] = normalized_mutual_info_score(
        truth_label_idx, pred_label_idx
    )
    clustering_metrics["ari"] = adjusted_rand_score(truth_label_idx, pred_label_idx)
    clustering_metrics["ami"] = adjusted_mutual_info_score(
        truth_label_idx, pred_label_idx
    )
    clustering_metrics["fmi"] = fowlkes_mallows_score(truth_label_idx, pred_label_idx)
    clustering_metrics["v_measure"] = v_measure_score(truth_label_idx, pred_label_idx)
    features_normalized = normalize(features, norm="l2", axis=1)
    angular_sh = silhouette_score(features_normalized, pred_label_idx, metric="cosine")
    clustering_metrics["angular_silhouette_score"] = angular_sh
    return clustering_metrics


def compute_etf_metrics(prototypes: torch.Tensor):
    P = F.normalize(prototypes, dim=1)  # 确保归一化
    K, d = P.shape

    # Gram matrix
    G = P @ P.T

    # 1. ETF 理想 Gram
    target = torch.full((K, K), -1 / (K - 1), device=P.device)
    target.fill_diagonal_(1)
    etf_error = torch.norm(G - target, p="fro").item()

    # 2. Cosine variance
    mask = ~torch.eye(K, dtype=bool, device=P.device)
    cosines = G[mask]
    cosine_variance = cosines.var().item()

    # 3. Welch bound gap
    max_cos = cosines.abs().max().item()
    welch_bound = math.sqrt((K - d) / (d * (K - 1)))
    welch_gap = max_cos - welch_bound

    # 4. Average cosine deviation
    avg_cosine_deviation = torch.mean(torch.abs(cosines - (-1 / (K - 1)))).item()

    # 5. Gram eigenvalue deviation
    eigvals = torch.linalg.eigvalsh(G)
    non_zero_eig = eigvals[eigvals > 1e-6]
    eig_var = non_zero_eig.var().item()

    return {
        "etf_error": etf_error,
        "cosine_variance": cosine_variance,
        "welch_gap": welch_gap,
        "avg_cosine_deviation": avg_cosine_deviation,
        "eig_var": eig_var,
    }


def compute_statistics_metrics(kappas, margins, scales, id2label):

    kappas = kappas.detach().cpu()
    margins = margins.detach().cpu()
    scales = scales.detach().cpu()
    C = kappas.shape[0]

    per_class = {}
    for i in range(C):
        label = id2label[i]
        per_class[label] = {
            "kappa": float(kappas[i]),
            "margin": float(margins[i]),
            "scale": float(scales[i]),
        }

    summary = {
        "kappa_mean": float(kappas.mean()),
        "kappa_std": float(kappas.std()),
        "margin_mean": float(margins.mean()),
        "margin_std": float(margins.std()),
        "scale_mean": float(scales.mean()),
        "scale_std": float(scales.std()),
    }

    return {"per_class": per_class, "summary": summary}
