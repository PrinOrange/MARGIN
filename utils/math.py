import math
import torch
import torch.nn.functional as F
from scipy.stats import chi2


def compute_vmf_kappa(features, dim):
    """
    改进的 κ 估计，使用更稳定的数值方法
    """
    r_bar = features.mean(dim=0)
    R = r_bar.norm().item()

    # 防止边界情况
    R = min(R, 0.999999)

    # Sra (2012) 的改进近似，全范围更稳定
    if R < 0.53:
        kappa = R * (dim - R**2) / (1 - R**2)
    elif R < 0.85:
        kappa = R * (dim - 1) / (1 - R**2) * 0.8  # 经验修正
    else:
        # 高集中度：使用级数展开
        kappa = (dim - 1) / (2 * (1 - R))
        # 高阶修正项
        kappa = kappa - (dim - 3) / (4 * kappa)

    return torch.tensor(kappa)


def compute_margin(
    n: int,
    count_i: int,
    kappa_i: float,
    dim: int,
    alpha: float = 0.95,
):
    """
    Geometric margin:
    spherical cap angle - Voronoi cone angle
    """

    # predictive uncertainty
    q = chi2.ppf(alpha, df=dim - 1)

    kappa_i_eff = kappa_i * count_i / (count_i + 1)

    theta_vmf = math.sqrt(q / kappa_i_eff)

    # ETF Voronoi cone angle
    theta_voronoi_cell = 0.5 * math.acos(-1 / (n - 1))

    margin = theta_vmf - theta_voronoi_cell

    return margin


def compute_pairwise_margin(
    n: int,
    mu_i: torch.Tensor,
    count_i: int,
    kappa_i: float,
    mu_j: torch.Tensor,
    count_j: int,
    kappa_j: float,
    dim: int,
    alpha: float = 0.95,
    temperature: float = 0.5,
):
    """
    vMF predictive margin
    """

    q = chi2.ppf(alpha, df=dim - 1)

    # predictive uncertainty
    mu_i = F.normalize(mu_i, dim=0)
    mu_j = F.normalize(mu_j, dim=0)

    cos_theta = torch.dot(mu_i, mu_j).clamp(-1.0, 1.0)
    theta_ij = torch.acos(cos_theta).item()

    q = chi2.ppf(alpha, df=dim - 1)

    theta_i = math.sqrt(q / kappa_i)
    theta_j = math.sqrt(q / kappa_j)

    sigma = math.sqrt(theta_i + theta_j)

    # kappa_i_eff = kappa_i * count_i / (count_i + 1)
    # kappa_j_eff = kappa_j * count_j / (count_j + 1)

    # theta_i = math.sqrt(q / kappa_i_eff)
    # theta_j = math.sqrt(q / kappa_j_eff)

    # margin = max(0.0, theta_i + theta_j - theta_ij) * temperature
    margin = sigma / (sigma + theta_ij)

    return margin


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
        distances = torch.acos(cos_sim) + 1e-8

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
