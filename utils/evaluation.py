import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import (
    compute_classification_metrics,
    compute_clustering_metrics,
    compute_etf_metrics,
    compute_statistics_metrics,
)
import torch.nn.functional as F
from utils.model import MARGINModel


def evaluate_model(model: MARGINModel, dataloader: DataLoader, title: str, device):
    """评估模型"""
    model.eval()

    all_pred_label_idx = []
    all_truth_label_idx = []
    all_features = []
    all_raw_labels = []

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=title, leave=False)

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_idxs = batch["label_idx"].to(device)

            # ✅ 一次 forward
            with torch.autocast(device):
                cos_theta, features = model(
                    input_ids, attention_mask, return_features=True
                )

            # ✅ 强制单位球
            features = F.normalize(features, dim=1)
            prototypes = F.normalize(model.current_geometric_median_prototypes, dim=1)

            # ✅ prototype 分类
            logits = torch.matmul(features, prototypes.t())
            preds = torch.argmax(logits, dim=1)

            # ✅ loss（监控用）
            loss = model.loss_head(cos_theta, label_idxs)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            all_pred_label_idx.extend(preds.cpu().numpy())
            all_truth_label_idx.extend(label_idxs.cpu().numpy())
            all_features.append(features.cpu())
            all_raw_labels.extend(batch["raw_label"])

    avg_loss = total_loss / num_batches
    all_features = torch.cat(all_features, dim=0)

    # 计算指标
    classification_metrics = compute_classification_metrics(
        all_truth_label_idx, all_pred_label_idx, model.id2label
    )
    clustering_metrics = compute_clustering_metrics(
        all_truth_label_idx, all_pred_label_idx, all_features
    )
    etf_metrics = compute_etf_metrics(model.current_geometric_median_prototypes.cpu())

    statistics_metrics = compute_statistics_metrics(
        model.loss_head.kappas,
        model.loss_head.margins,
        model.loss_head.scales,
        model.id2label,
    )

    metrics = {
        "val_loss": avg_loss,
        "classification_metrics": classification_metrics,
        "clustering_metrics": clustering_metrics,
        "etf_metrics": etf_metrics,
        "statistics_metrics": statistics_metrics,
    }

    return (
        metrics,
        all_features,
        all_truth_label_idx,
        all_pred_label_idx,
        all_raw_labels,
        avg_loss,
    )
