import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForTextEncoding
from utils.dataset import CodeDataset
from utils.math import (
    compute_pairwise_margin,
    compute_margin,
    compute_convergence_coefficient,
)
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
        self.roberta_config = AutoConfig.from_pretrained(
            backbone, output_hidden_states=True
        )
        self.roberta = AutoModelForTextEncoding.from_pretrained(
            backbone, config=self.roberta_config
        )

        self.embedding_dim = self.roberta_config.hidden_size
        self.num_classes = len(train_dataset.label2idx)
        self.dropout = nn.Dropout(dropout_rate)
        self.weights = nn.Parameter(
            F.normalize(torch.Tensor(self.num_classes, self.embedding_dim), p=2, dim=1)
        )
        self.layer_weights = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.weights)

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
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states
        selected_layers = []
        layers_to_concat = [-1]  # 选择哪些层的输出进行融合
        for layer_idx in layers_to_concat:
            selected_layers.append(hidden_states[layer_idx])  # (B, L, D)
        stacked = torch.stack(selected_layers, dim=0)  # (N, B, L, D)
        norm_weights = self.softmax(self.layer_weights)  # (N,)
        weighted = norm_weights.view(-1, 1, 1, 1) * stacked
        fused = torch.sum(weighted, dim=0)  # (B, L, D)
        features = fused[:, 0, :]  # (B, D)
        cos_thetas = torch.matmul(
            F.normalize(features, p=2, dim=1), F.normalize(self.weights, p=2, dim=1).t()
        )  # [B, C]
        if return_features:
            return cos_thetas, features
        return cos_thetas

    def get_norm_weight_prototypes(self):
        """返回归一化的 weight prototypes"""
        return F.normalize(self.weights.detach(), p=2, dim=1)


# ==================== ArcFace Loss with Adaptive Margin ====================
class MARGINLossHead(nn.Module):

    def __init__(
        self,
        num_classes: int,
        base_scale: int,
        alpha: float,
        dim: int,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.base_scale = base_scale
        self.dim = dim
        self.alpha = alpha

        self.is_initialized = False

        self.margins = torch.zeros(num_classes)
        self.kappas = torch.zeros(num_classes)
        self.scales = torch.full((num_classes,), base_scale, dtype=torch.float32)

    def update_adaptive_params(
        self,
        kappas: torch.Tensor,
        class_counts: torch.Tensor,
        mean_prototypes: torch.Tensor,
    ):
        device = kappas.device

        max_class_count = class_counts.max().item()
        # min-max 归一化
        kappa_min = kappas.min()
        kappa_max = kappas.max()
        kappas_norm = (kappas - kappa_min) / (kappa_max - kappa_min + 1e-8)

        # 反向 softmax（小 κ → 大权重）
        scale_weights = torch.softmax(-kappas_norm, dim=0)

        # 恢复 scale 量级
        new_scales = self.base_scale * scale_weights * len(kappas)
        
        C = self.num_classes
        new_margins = torch.zeros(C, device=device)
        # betas = torch.zeros(C, device=device)

        for i in range(C):
            count_i = class_counts[i]
            kappa_i = torch.clamp(kappas[i], min=1.0)

            margin = compute_margin(
                self.num_classes,
                count_i,
                kappa_i,
                self.dim,
                max_class_count,
                self.alpha,
            )
            new_margins[i] = margin

        # ========================
        # Cosine UPDATE
        # ========================

        self.kappas = kappas
        self.margins = new_margins
        self.scales = new_scales

        log.print(f"Updated margins: {self.margins}")
        log.print(f"Updated scales: {self.scales}")
        log.print(f"Updated kappas: {self.kappas}")

        return self.margins, self.scales

    def forward(self, cos_thetas, label_idxs):
        B, C = cos_thetas.shape
        device = cos_thetas.device

        cos_thetas = torch.clamp(cos_thetas, -1 + 1e-7, 1 - 1e-7)
        margins_batch = self.margins.to(device)[label_idxs].to(cos_thetas.dtype)
        cos_m = torch.cos(margins_batch)
        sin_m = torch.sin(margins_batch)
        target_cos = cos_thetas[torch.arange(B, device=device), label_idxs]
        target_sin = torch.sqrt(torch.clamp(1.0 - target_cos**2, min=1e-7))
        # cos(θ + m)
        target_cos_margin = target_cos * cos_m - target_sin * sin_m
        target_cos_margin = target_cos_margin.to(cos_thetas.dtype)
        logits = cos_thetas.clone()
        logits[torch.arange(B, device=device), label_idxs] = target_cos_margin
        logits = logits * self.scales.to(device).unsqueeze(0)
        loss = F.cross_entropy(logits, label_idxs)
        return loss

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
