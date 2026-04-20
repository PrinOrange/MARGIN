import math
import torch
import torch.nn.functional as F
from scipy.stats import chi2


def sigmoid(x):
    """计算 sigmoid 并返回 Python 浮点数"""
    x_tensor = torch.tensor([x], dtype=torch.float32)
    y_tensor = torch.sigmoid(x_tensor)
    return y_tensor.item()


def compute_vmf_kappa(features, prototype):
    if features.size(0) == 0:
        return 0.0

    features = torch.nn.functional.normalize(features, dim=1)
    prototype = torch.nn.functional.normalize(prototype, dim=0)

    cos_sim = torch.matmul(features, prototype)  # (N)
    r = torch.mean(cos_sim).item()
    d = features.size(1)
    if r >= 1.0:
        r = 0.999999
    if r <= 0:
        return 0.0
    kappa = r * (d - r * r) / (1 - r * r)
    return max(kappa, 1e-6)


def compute_margin(
    n: int,
    count_i: int,
    kappa_i: float,
    dim: int,
    max_class_counts: int,
    alpha: float = 0.95,
):
    """
    Geometric margin with frequency-aware lower bound:
    margin = max(0, max(theta_vmf, theta_freq) - theta_voronoi)
    """

    # predictive uncertainty (confidence level)
    q = chi2.ppf(alpha, df=dim - 1)

    # ---- vMF-based uncertainty（类内结构 + 样本数）----
    kappa_i_eff = kappa_i
    theta_vmf = math.sqrt(q / kappa_i_eff)

    # ---- frequency-induced lower bound（采样分辨率极限）----
    # theta_freq = 0.5 * sigmoid(-count_i / math.sqrt(max_class_counts))

    # ---- ETF Voronoi cone angle（类间分离）----
    theta_voronoi_cell = 0.5 * math.acos(-1 / (n - 1))

    # ---- final margin ----
    margin = max(0, theta_vmf - theta_voronoi_cell)

    return margin


def compute_convergence_coefficient(
    n: int, count_i: int, kappa_i: float, dim: int, alpha: float = 0.95
):
    # predictive uncertainty
    q = chi2.ppf(alpha, df=dim - 1)

    kappa_i_eff = kappa_i * count_i / (count_i + 1)

    theta_vmf = math.sqrt(q / kappa_i_eff)

    # ETF Voronoi cone angle
    theta_voronoi_cell = 0.5 * math.acos(-1 / (n - 1))
    convergence_coeff = theta_voronoi_cell / (theta_vmf)
    convergence_coeff = max(0.0, min(1.0, convergence_coeff))
    return convergence_coeff


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


def compute_geometric_median(
    X: torch.Tensor,
    max_iter: int = 100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    球面上的几何中位数（Weiszfeld + 投影）

    X: [N, D]，已 L2 normalize
    """

    # ✅ 初始化也要在球面上
    y = F.normalize(X.mean(dim=0), dim=0)
    for _ in range(max_iter):
        dist = torch.norm(X - y, dim=1)
        # ✅ 防止除零
        dist = torch.clamp(dist, min=eps)
        inv_dist = 1.0 / dist
        y_new = (X * inv_dist[:, None]).sum(dim=0) / inv_dist.sum()
        # ✅ 投影回单位球（关键）
        y_new = F.normalize(y_new, dim=0)  # 这里的 dim 是不是写错方向了
        # ✅ 收敛判断
        if torch.norm(y - y_new) < eps:
            break
        y = y_new
    return y
