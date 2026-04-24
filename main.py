import argparse
from datetime import datetime
import os
from datasets import load_dataset
from utils.trainer import Trainer, TrainerConfig
from utils.seed import set_seed
from utils.dataset import CodeDataset
from utils.model import MARGINModel
from utils.logger import log


def parse_args():
    # 1. 创建解析器对象
    parser = argparse.ArgumentParser(description="MARGIN Model Training Script")

    # --- 数据与模型配置 ---
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="codemetic/MARGIN",
        help="HuggingFace 数据集名称 (默认: codemetic/MARGIN)",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        default="bigvul",
        help="数据集子集名称 (默认: bigvul)",
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="microsoft/unixcoder-base",
        help="预训练模型名称 (默认: microsoft/unixcoder-base)",
    )

    # --- 超参数配置 ---
    parser.add_argument(
        "--base_scale", type=int, default=10, help="Base scale 参数 (默认: 10)"
    )
    parser.add_argument(
        "--confidence_alpha",
        type=float,
        default=0.95,
        help="Confidence Alpha 参数 (默认: 0.95)",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="学习率 (默认: 2e-5)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="权重衰减 (默认: 0.01)"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=200, help="最大训练轮数 (默认: 200)"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=200,
        help="早停耐心值 (默认: 200)",
    )

    # --- 运行环境配置 ---
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="训练设备，如 cuda:0, cpu (默认: cuda:1)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="批次大小 (默认: 16)"
    )

    # UMAP 相关参数
    parser.add_argument("--umap_n_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)

    # 解析参数
    args = parser.parse_args()
    return args


def main():
    # 2. 获取命令行参数
    args = parse_args()

    # 生成时间戳前缀，用于区分不同次运行的输出文件夹
    TIME_PREFIX = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    OUTPUT_DIR = f"./output/{args.dataset_subset}-{args.backbone_name.split('/')[0]}-{TIME_PREFIX}"
    log.set_log_file(os.path.join(OUTPUT_DIR, "train.log"))

    # 3. 使用 args 中的值构建配置
    # 注意：这里我们优先使用命令行传入的参数，如果没有传则使用 default 值
    config = TrainerConfig(
        output_dir=OUTPUT_DIR,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        device=args.device,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        seed=args.seed,
    )

    set_seed(config.seed)

    log.print("Loading dataset...")
    # 使用 args 中的数据集名称
    dataset = load_dataset(args.dataset_name, args.dataset_subset)

    train_hf = dataset["train"]
    val_hf = dataset["val"]
    test_hf = dataset["test"]

    log.print(
        f"Train size: {len(train_hf)}, Val size: {len(val_hf)}, Test size: {len(test_hf)}"
    )

    log.print(f"Loading tokenizer and model: {args.backbone_name}")

    # 使用 args 中的模型名称
    train_dataset = CodeDataset(args.backbone_name, train_hf)
    # val_dataset = CodeDataset(MODEL_NAME, val_hf, config.max_length) # 原代码注释掉的行
    test_dataset = CodeDataset(args.backbone_name, test_hf)

    label2id = train_dataset.label2idx

    log.print(f"Number of classes: {len(label2id)}")
    log.print(f"Label mapping: {label2id}")

    model = MARGINModel(
        backbone=args.backbone_name,
        base_scale=args.base_scale,
        alpha=args.confidence_alpha,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )

    # 训练
    trainer = Trainer(model, config)
    # 这里传入参数名称以便日志记录
    trainer.train(args.dataset_subset, args.backbone_name)


if __name__ == "__main__":
    main()
