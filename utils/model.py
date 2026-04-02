import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForTextEncoding


# ==================== 模型定义 ====================
class MARGINModel(nn.Module):
    def __init__(self, num_classes, backbone: str, embedding_dim: int):
        super().__init__()
        self.roberta = AutoModelForTextEncoding.from_pretrained(backbone)
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
class MARGINLossHead(nn.Module):

    def __init__(self, num_classes, scale):
        super().__init__()

        self.scale = scale
        self.num_classes = num_classes

        self.register_buffer("margins", torch.ones(num_classes))
        self.register_buffer("kappas", torch.ones(num_classes))
        self.register_buffer("scales", torch.ones(num_classes))

    def update_margins(self, margins_dict):
        for idx, margin in margins_dict.items():
            self.margins[idx] = torch.tensor(margin, device=self.margins.device)

    def update_kappas(self, kappas_dict):
        for idx, kappa in kappas_dict.items():
            self.kappas[idx] = torch.tensor(kappa, device=self.kappas.device)

    def update_scales(self, scale_dict):
        for idx, scale in scale_dict.items():
            self.scales[idx] = torch.tensor(scale, device=self.scales.device)

    def forward(self, cos_theta, labels):
        B, C = cos_theta.shape

        # ✅ margins 按标签取（每个样本一个 margin）
        margins_batch = self.margins[labels]  # [B]

        cos_theta = torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7)

        cos_m = torch.cos(margins_batch)
        sin_m = torch.sin(margins_batch)

        sin_theta = torch.sqrt(torch.clamp(1 - cos_theta**2, min=1e-7))

        cos_theta_plus_m = cos_theta * cos_m.unsqueeze(1) - sin_theta * sin_m.unsqueeze(
            1
        )

        one_hot = F.one_hot(labels, C).float()

        output = cos_theta * (1 - one_hot) + cos_theta_plus_m * one_hot

        # self.scales: [C] → unsqueeze(0): [1, C] → 广播到 [B, C]
        output = output * self.scales.unsqueeze(0)

        # global scale
        output = output * self.scale

        loss = F.cross_entropy(output, labels)
        return loss
