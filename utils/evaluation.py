import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.metrics import compute_metrics
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

            # 使用几何中位数原型进行分类
            _, features = model(input_ids, attention_mask, return_features=True)

            # 计算与几何中位数原型的余弦相似度
            sim = torch.matmul(features, model.current_geometric_median_prototypes.t())
            preds = torch.argmax(sim, dim=1)

            # 计算loss（用于早停）
            with torch.autocast(device):
                cos_theta = model(input_ids, attention_mask)
                loss = model.classification_loss(cos_theta, label_idxs)

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            all_pred_label_idx.extend(preds.cpu().numpy())
            all_truth_label_idx.extend(label_idxs.cpu().numpy())
            all_features.append(features.cpu())
            all_raw_labels.extend(batch["raw_label"])

    avg_loss = total_loss / num_batches
    all_features = torch.cat(all_features, dim=0).numpy()

    # 计算指标
    metrics = compute_metrics(all_truth_label_idx, all_pred_label_idx, model.id2label)
    metrics["val_loss"] = avg_loss
    return (
        metrics,
        all_features,
        all_truth_label_idx,
        all_pred_label_idx,
        all_raw_labels,
        avg_loss,
    )
