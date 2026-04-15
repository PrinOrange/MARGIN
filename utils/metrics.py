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
    # 3. Positive Macro (仅在真实正样本子集上评估 CWE 分类能力)
    # =========================================================================
    positive_label_idx = [l for l in all_label_idx if l != 0]
    metrics["positive_macro"] = {"per_class": {}}

    if len(positive_label_idx) > 0:

        # Step1: 仅保留真实 positive 样本
        pos_indices = [
            i for i, y in enumerate(truth_label_idx) if y in positive_label_idx
        ]

        if len(pos_indices) > 0:

            y_true_pos = np.array([truth_label_idx[i] for i in pos_indices])

            y_pred_pos = np.array([pred_label_idx[i] for i in pos_indices])

            # ---------------------------------------------------------
            # 3.1 Macro Precision / Recall / F1
            # sklearn 会自动把预测到 non-vul(0) 当成 miss/FN
            # ---------------------------------------------------------
            metrics["positive_macro"]["f1"] = f1_score(
                y_true_pos,
                y_pred_pos,
                average="macro",
                labels=positive_label_idx,
                zero_division=0,
            )

            metrics["positive_macro"]["precision"] = precision_score(
                y_true_pos,
                y_pred_pos,
                average="macro",
                labels=positive_label_idx,
                zero_division=0,
            )

            metrics["positive_macro"]["recall"] = recall_score(
                y_true_pos,
                y_pred_pos,
                average="macro",
                labels=positive_label_idx,
                zero_division=0,
            )

            # ---------------------------------------------------------
            # 3.2 Macro MCC + Per-class OVA Metrics
            # 严格 One-vs-Rest
            # ---------------------------------------------------------
            pos_mcc_scores = []

            for label_idx in positive_label_idx:

                y_true_binary = (y_true_pos == label_idx).astype(int)

                y_pred_binary = (y_pred_pos == label_idx).astype(int)

                # -------------------------
                # TP / FP / TN / FN
                # -------------------------
                tp = int(np.sum((y_true_binary == 1) & (y_pred_binary == 1)))

                fp = int(np.sum((y_true_binary == 0) & (y_pred_binary == 1)))

                fn = int(np.sum((y_true_binary == 1) & (y_pred_binary == 0)))

                tn = int(np.sum((y_true_binary == 0) & (y_pred_binary == 0)))

                support = tp + fn

                # -------------------------
                # MCC
                # -------------------------
                try:
                    mcc = matthews_corrcoef(
                        y_true_binary,
                        y_pred_binary,
                    )
                    mcc = 0.0 if np.isnan(mcc) else float(mcc)

                except ValueError:
                    mcc = 0.0

                pos_mcc_scores.append(mcc)

                # -------------------------
                # Other binary metrics
                # -------------------------
                binary_metrics = compute_binary_metrics(y_true_binary, y_pred_binary)

                metrics["positive_macro"]["per_class"][idx2label[label_idx]] = {
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "support": support,
                    "mcc": mcc,
                    **binary_metrics,
                }

            metrics["positive_macro"]["mcc"] = float(np.mean(pos_mcc_scores))

        else:
            metrics["positive_macro"].update(
                {
                    "mcc": 0.0,
                    "f1": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                }
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
    etf_error_norm = etf_error / K  # 归一化

    # 2. Off-diagonal cosines
    mask = ~torch.eye(K, dtype=bool, device=P.device)
    cosines = G[mask]

    # 3. Cosine variance / std
    cosine_variance = cosines.var().item()
    cosine_std = cosines.std().item()

    # 4. Average cosine deviation from ETF ideal
    avg_cosine_deviation = torch.mean(torch.abs(cosines - (-1 / (K - 1)))).item()
    max_cosine_deviation = torch.max(torch.abs(cosines - (-1 / (K - 1)))).item()

    # 5. Average angle deviation (rad)
    angles = torch.acos(cosines.clamp(-1, 1))
    etf_angle = math.acos(-1 / (K - 1))
    avg_angle_deviation = torch.mean(torch.abs(angles - etf_angle)).item()
    max_angle_deviation = torch.max(torch.abs(angles - etf_angle)).item()

    # 6. Gram eigenvalue statistics
    eigvals = torch.linalg.eigvalsh(G)
    non_zero_eig = eigvals[eigvals > 1e-6]
    eig_var = non_zero_eig.var().item()
    eig_mean = non_zero_eig.mean().item()

    # 7. Gram condition number
    cond_num = torch.linalg.cond(G).item()

    return {
        "etf_error": etf_error,
        "etf_error_norm": etf_error_norm,
        "cosine_variance": cosine_variance,
        "cosine_std": cosine_std,
        "avg_cosine_deviation": avg_cosine_deviation,
        "max_cosine_deviation": max_cosine_deviation,
        "avg_angle_deviation": avg_angle_deviation,
        "max_angle_deviation": max_angle_deviation,
        "eig_var": eig_var,
        "eig_mean": eig_mean,
        "cond_num": cond_num,
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
