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
    import numpy as np
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        matthews_corrcoef,
        accuracy_score,
    )

    all_label_idx = list(range(len(idx2label)))
    metrics = {}

    # ===============================
    # Helper: OVA confusion (FULL)
    # ===============================
    def ova_confusion(y_true, y_pred, class_idx):
        y_true_bin = (np.array(y_true) == class_idx).astype(int)
        y_pred_bin = (np.array(y_pred) == class_idx).astype(int)

        tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
        fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
        fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))
        tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))

        return tp, fp, tn, fn, y_true_bin, y_pred_bin

    # ===============================
    # GLOBAL MACRO (FULL MULTICLASS)
    # ===============================
    metrics["global_macro"] = {
        "mcc": float(matthews_corrcoef(truth_label_idx, pred_label_idx)),
        "f1": float(f1_score(truth_label_idx, pred_label_idx, average="macro", zero_division=0)),
        "precision": float(precision_score(truth_label_idx, pred_label_idx, average="macro", zero_division=0)),
        "recall": float(recall_score(truth_label_idx, pred_label_idx, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth_label_idx, pred_label_idx)),
        "per_class": {},
    }

    fnr_list, fpr_list = [], []

    for c in all_label_idx:
        tp, fp, tn, fn, y_true_bin, y_pred_bin = ova_confusion(truth_label_idx, pred_label_idx, c)

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        fnr_list.append(fnr)
        fpr_list.append(fpr)

        # binary metrics (consistent OVA)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin) if (tp + fp + fn + tn) > 0 else 0.0

        label_name = idx2label[c]
        metrics["global_macro"]["per_class"][label_name] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "support": int(tp + fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mcc": float(mcc),
        }

    metrics["global_macro"]["fnr"] = float(np.mean(fnr_list))
    metrics["global_macro"]["fpr"] = float(np.mean(fpr_list))

    # ===============================
    # POSITIVE MACRO (FULL OVA)
    # CWE-only evaluation but FULL dataset OVA
    # ===============================
    positive_label_idx = [l for l in all_label_idx if l != 0]
    metrics["positive_macro"] = {"per_class": {}}

    if len(positive_label_idx) > 0:

        pos_mcc_list = []

        for c in positive_label_idx:

            tp, fp, tn, fn, y_true_bin, y_pred_bin = ova_confusion(
                truth_label_idx,
                pred_label_idx,
                c,
            )

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
            mcc = 0.0 if np.isnan(mcc) else float(mcc)

            pos_mcc_list.append(mcc)

            metrics["positive_macro"]["per_class"][idx2label[c]] = {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "support": int(tp + fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mcc": mcc,
            }

        metrics["positive_macro"]["mcc"] = float(np.mean(pos_mcc_list))
        metrics["positive_macro"]["precision"] = float(
            np.mean([metrics["positive_macro"]["per_class"][idx2label[c]]["precision"]
                     for c in positive_label_idx])
        )
        metrics["positive_macro"]["recall"] = float(
            np.mean([metrics["positive_macro"]["per_class"][idx2label[c]]["recall"]
                     for c in positive_label_idx])
        )
        metrics["positive_macro"]["f1"] = float(
            np.mean([metrics["positive_macro"]["per_class"][idx2label[c]]["f1"]
                     for c in positive_label_idx])
        )

    else:
        metrics["positive_macro"] = {
            "mcc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "per_class": {},
        }

    # ===============================
    # BINARY (Non-vul vs Vul)
    # ===============================
    y_true_bin = [0 if y == 0 else 1 for y in truth_label_idx]
    y_pred_bin = [0 if y == 0 else 1 for y in pred_label_idx]

    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics["binary"] = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy_score(y_true_bin, y_pred_bin),
        "mcc": float(matthews_corrcoef(y_true_bin, y_pred_bin)),
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
