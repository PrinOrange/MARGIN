from collections import Counter

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class CodeDataset(Dataset):
    def __init__(self, backbone: str, hf_dataset: str, max_length: int):
        self.dataset = hf_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.max_length = max_length
        self.label2idx = {}
        self.idx2label = {}
        self._build_label_mapping()

    def _build_label_mapping(self):
        label_counts = Counter(self.dataset["label"])
        # 按频率降序排序
        sorted_labels = sorted(
            label_counts.keys(),
            key=lambda x: label_counts[x],
            reverse=True,
        )
        # 强制 Non-vul 放最前
        if "Non-vul" in sorted_labels:
            sorted_labels.remove("Non-vul")
            sorted_labels.insert(0, "Non-vul")

        for idx, label in enumerate(sorted_labels):
            self.label2idx[label] = idx
            self.idx2label[idx] = label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        code = item["source"]
        label = item["label"]

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_idx": self.label2idx[label],
            "raw_label": label,
        }
