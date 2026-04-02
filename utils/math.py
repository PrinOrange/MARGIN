import math
import torch
import torch.nn.functional as F
from scipy.stats import chi2


def compute_vmf_kappa(features: torch.Tensor, dim: int):
    """
    计算 vMF 分布的 kappa 参数（MLE 近似）
    features: [N, D] 已归一化特征
    dim: 维度 D
    返回: kappa tensor，标量形式
    """
    # mean resultant vector
    r_bar_vec = features.mean(dim=0)  # [D]
    r = r_bar_vec.norm(p=2)  # 标量 ||r_bar||

    # 防止数值问题
    r = torch.clamp(r, 0.0, 0.9999)

    numerator = r * (dim - r**2)
    denominator = 1 - r**2

    kappa = numerator / denominator
    kappa = torch.clamp(kappa, min=1.0)  # 保证最小为1

    return kappa


def compute_pairwise_margin(
    mu_i: torch.Tensor, kappa_i: float, mu_j: torch.Tensor, kappa_j: float, alpha=0.95
):
    """
    计算两个类别之间的自适应 margin Δm_{i,j} (弧度)
    mu_i, mu_j: 未归一化向量 [embedding_dim]
    kappa_i, kappa_j: 对应类别的 κ
    """
    # 归一化向量
    mu_i = F.normalize(mu_i, dim=0)
    mu_j = F.normalize(mu_j, dim=0)

    dim = mu_i.size(0)
    # 球冠半径近似 (chi2 或高维近似)
    q = chi2.ppf(2 * alpha - 1, df=dim - 1)
    term_i = math.sqrt(q / kappa_i)
    term_j = math.sqrt(q / kappa_j)

    # 中心夹角
    cos_theta = torch.dot(mu_i, mu_j).clamp(-1.0, 1.0)
    theta_ij = torch.acos(cos_theta)

    # 最大 overlap 角度
    theta_overlap = max(0.0, term_i + term_j - theta_ij.item())

    return theta_overlap


def compute_geometric_median(features, max_iter, tol=1e-5):
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
