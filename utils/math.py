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

    return torch.clamp(torch.tensor(kappa), 1.0, 5000.0)


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
):
    """
    基于预测分布的 margin（新样本的不确定性）
    """
    mu_i = F.normalize(mu_i, dim=0)
    mu_j = F.normalize(mu_j, dim=0)
    
    # 中心夹角
    cos_theta = torch.dot(mu_i, mu_j).clamp(-1.0, 1.0)
    theta_ij = torch.acos(cos_theta).item()
    
    # === 关键：预测不确定性 vs 参数不确定性 ===
    # 参数不确定性：σ²_μ ≈ 1/(n*κ)  —— 你当前用的
    # 预测不确定性：σ²_pred ≈ 1/κ + 1/(n*κ) ≈ 1/κ （主导项）
    
    # 对于 vMF，新样本的分布就是 vMF(μ, κ)
    # 我们关心的是：两个分布的重叠程度
    
    q = chi2.ppf(alpha, df=dim - 1)
    
    # 参数不确定性（估计误差）
    param_uncertainty_i = math.sqrt(q / (count_i * kappa_i))
    param_uncertainty_j = math.sqrt(q / (count_j * kappa_j))
    
    # 预测不确定性：考虑 κ 本身描述的固有分散
    # 对于 vMF，"预测球冠"应该基于 κ 本身
    # 这里用启发式：结合参数不确定性和分布本身的集中度
    
    # 方法1：直接用 1/κ 作为预测方差
    predictive_var_i = 1.0 / kappa_i + 1.0 / (count_i * kappa_i)
    predictive_var_j = 1.0 / kappa_j + 1.0 / (count_j * kappa_j)
    
    theta_i = math.sqrt(q * param_uncertainty_i)
    theta_j = math.sqrt(q * param_uncertainty_j)
    
    # 或者更简单的启发式：降低有效 κ
    # effective_kappa_i = kappa_i * (1 - 1/count_i)  # 小样本修正
    
    # 计算重叠
    # if theta_ij >= theta_i + theta_j:
    #     return 0.0
    # else:
    #     return max(0.0, theta_i + theta_j - theta_ij)
    
    return 0


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
