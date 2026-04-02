from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
import umap


def draw_prototype_dispersion(
    geometric_median_prototypes: torch.Tensor, id2label: dict, title: str, filepath: str
):
    geo_medians = geometric_median_prototypes.cpu().numpy()
    sim_matrix = np.matmul(geo_medians, geo_medians.T)
    sim_matrix = sim_matrix * 100
    mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        mask=mask,
        fmt=".0f",
        cmap="YlOrRd",
        vmin=-100,
        vmax=100,
        xticklabels=[id2label[i] for i in range(len(id2label))],
        yticklabels=[id2label[i] for i in range(len(id2label))],
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def draw_prototype_alignment(
    geometric_median_prototypes: torch.Tensor,
    weight_prototypes: torch.Tensor,
    id2label: dict,
    title: str,
    filepath: str,
):
    weight_protos = weight_prototypes.detach()
    sim_matrix = torch.matmul(geometric_median_prototypes.detach(), weight_protos.t())
    sim_matrix = (sim_matrix * 100).cpu().numpy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".0f",
        cmap="coolwarm",
        vmin=-100,
        vmax=100,
        xticklabels=[f"W-{id2label[i]}" for i in range(len(id2label))],
        yticklabels=[f"G-{id2label[i]}" for i in range(len(id2label))],
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def draw_umap(
    features: torch.Tensor,
    pred_label_idx: list,
    id2label: dict,
    title: str,
    filepath: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state
    )
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(6, 5))

    # 先画正样本（Non-vul为灰色，其他为有颜色）
    unique_labels = sorted(set(pred_label_idx))

    # 定义颜色：Non-vul为灰色，其他为tab10
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels) - 1))
    color_map = {}
    idx = 0
    for label in unique_labels:
        if label == 0:
            color_map[label] = "gray"
        else:
            color_map[label] = colors[idx % len(colors)]
            idx += 1

    # 后画负样本（Non-vul）
    mask = np.array(pred_label_idx) == 0
    plt.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c="gray",
        label=id2label.get(0, "Non-vul"),
        alpha=0.3,
        s=30,
        edgecolors="none",
    )

    # 先画正样本（非Non-vul）
    for label in unique_labels:
        if label != 0:
            mask = np.array(pred_label_idx) == label
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color_map[label]],
                label=id2label[label],
                alpha=0.9,
                s=20,
                edgecolors="none",
            )

    plt.legend(loc="best", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
