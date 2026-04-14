import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForTextEncoding
from utils.dataset import CodeDataset
from utils.math import compute_pairwise_margin, compute_margin
from utils.logger import log

# ==================== 模型定义 ====================
class MARGINModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        base_scale: float,
        alpha: float,
        train_dataset: CodeDataset,
        val_dataset: CodeDataset,
        dropout_rate: float = 0.0,  # 1. 新增参数
    ):
        super().__init__()
        self.roberta = AutoModelForTextEncoding.from_pretrained(backbone)
        self.config = AutoConfig.from_pretrained(backbone)

        self.embedding_dim = self.config.hidden_size
        self.num_classes = len(train_dataset.label2idx)
        self.dropout = nn.Dropout(dropout_rate)
        self.weight_prototypes = nn.Parameter(
            F.normalize(torch.randn(self.num_classes, self.embedding_dim), p=2, dim=1)
        )
        self.class_counts = torch.zeros(self.num_classes)
        self.current_kappas = torch.zeros(self.num_classes)
        self.current_mean_prototypes = torch.zeros(self.num_classes, self.embedding_dim)
        self.current_geometric_median_prototypes = torch.zeros(
            self.num_classes, self.embedding_dim
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label2id = train_dataset.label2idx
        self.id2label = train_dataset.idx2label

        self.loss_head: MARGINLossHead = MARGINLossHead(
            self.num_classes, base_scale, alpha, self.embedding_dim
        )

    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state[:, 0, :]  # [B, D]
        features = F.normalize(features, p=2, dim=1)  # [B, D]
        cos_theta = torch.matmul(features, self.weight_prototypes.t())  # [B, C]
        if return_features:
            return cos_theta, features
        return cos_theta

    def get_weight_prototypes(self):
        """返回归一化的weight prototypes"""
        return F.normalize(self.weight_prototypes, p=2, dim=1)


# ==================== ArcFace Loss with Adaptive Margin ====================
class MARGINLossHead(nn.Module):

    def __init__(self, num_classes, base_scale: int, alpha: float, dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.base_scale = base_scale
        self.dim = dim
        self.alpha = alpha

        self.register_buffer("margins", torch.zeros(num_classes))
        self.register_buffer("kappas", torch.zeros(num_classes))
        self.register_buffer(
            "scales", torch.full((num_classes,), base_scale, dtype=torch.float32)
        )

    def update_adaptive_params(
        self,
        kappas: torch.Tensor,
        class_counts: torch.Tensor,
        mean_prototypes: torch.Tensor,
    ):
        """
        kappas: [C]
        mean_prototypes: [C, D]
        返回:
            margins: [C]
            scales: [C]
        """
        device = kappas.device
        C = self.num_classes
        # --- 防止数值问题 ---
        kappas = torch.clamp(kappas, min=1e-6)
        kappa_min = kappas.min()
        kappa_max = kappas.max()
        kappas_norm = (kappas - kappa_min) / (kappa_max - kappa_min + 1e-8)
        scales_weight = 2 - kappas_norm
        scales = self.base_scale * scales_weight
        # 如果你想固定 scale
        # scales = torch.full_like(scales, 30.0)

        # --- 计算 margins ---
        margins = torch.zeros(C, device=device)

        for i in range(C):
            count_i = class_counts[i]
            kappa_i = torch.clamp(kappas[i], min=1.0)

            margin = compute_margin(
                self.num_classes,
                count_i,
                kappa_i,
                self.dim,
            )
            margins[i] = margin

        self.margins = margins
        self.kappas = kappas
        self.scales = scales

        log.print(f"Updated margins: {margins}")
        log.print(f"Updated scales: {scales}")
        log.print(f"Updated kappas: {kappas}")

        return margins, scales

    # def forward(self, cos_theta, label_idxs):
    #     B, C = cos_theta.shape

    #     # 每个样本对应的 margin
    #     margins = self.margins[label_idxs]  # [B]

    #     cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

    #     # one-hot
    #     one_hot = F.one_hot(label_idxs, C).float()

    #     # CosFace: cos(theta) - m
    #     cos_theta_minus_m = cos_theta - margins.unsqueeze(1)

    #     # 只对 target class 减 margin
    #     output = cos_theta * (1 - one_hot) + cos_theta_minus_m * one_hot

    #     # scale
    #     output = output * self.scales.unsqueeze(0)

    #     loss = F.cross_entropy(output, label_idxs)
    #     return loss

    def forward(self, cos_theta, label_idxs):
        B, C = cos_theta.shape

        # ✅ margins 按标签取（每个样本一个 margin）
        margins_batch = self.margins[label_idxs]  # [B]

        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

        cos_m = torch.cos(margins_batch)
        sin_m = torch.sin(margins_batch)

        sin_theta = torch.sqrt(torch.clamp(1 - cos_theta**2, min=1e-7))

        cos_theta_plus_m = cos_theta * cos_m.unsqueeze(1) - sin_theta * sin_m.unsqueeze(
            1
        )

        one_hot = F.one_hot(label_idxs, C).float()

        output = cos_theta * (1 - one_hot) + cos_theta_plus_m * one_hot

        # self.scales: [C] → unsqueeze(0): [1, C] → 广播到 [B, C]
        output = output * self.scales.unsqueeze(0)

        loss = F.cross_entropy(output, label_idxs)
        return loss
