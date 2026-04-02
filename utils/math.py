import json
import math
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import chi2
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from utils.seed import set_seed
from utils.metrics import compute_metrics
from utils.dataset import CodeDataset
from utils.model import MARGINLossHead, MARGINModel


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
    return torch.clamp(kappa, min=1)


def compute_pairwise_margin(kappa_i, kappa_j, dim, alpha=0.95):
    """
    计算两个类别之间的自适应margin Δm_{i,j}
    返回弧度值
    """
    q = chi2.ppf(2 * alpha - 1, df=dim - 1)
    term_i = math.sqrt(q / kappa_i)
    term_j = math.sqrt(1 / kappa_j)
    # return 0
    return 0.5 * term_i  # 返回弧度


def compute_geometric_median(features, max_iter, tol):
    """
    在单位超球面上计算几何中位数
    features: [N, D] 已归一化的特征
    weights: [N] 可选权重
    返回: [D] 几何中位数（已归一化）
    """
    initial_prototypes = torch.ones(features.shape[0], device=features.device)

    # 初始化为均值
    median = torch.sum(features * initial_prototypes.unsqueeze(1), dim=0) / torch.sum(
        initial_prototypes
    )
    median = F.normalize(median.unsqueeze(0), p=2, dim=1).squeeze(0)

    for _ in range(max_iter):
        # 计算球面距离 (余弦相似度转角度)
        cos_sim = torch.matmul(features, median)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        distances = torch.acos(cos_sim) + 1e-8  # 避免除零

        # 球面上的Weiszfeld算法
        w = initial_prototypes / distances
        new_median = torch.sum(features * w.unsqueeze(1), dim=0) / torch.sum(w)
        new_median = F.normalize(new_median.unsqueeze(0), p=2, dim=1).squeeze(0)

        # 检查收敛
        diff = 1 - torch.dot(median, new_median)  # 余弦距离
        median = new_median

        if diff < tol:
            break

    return median
