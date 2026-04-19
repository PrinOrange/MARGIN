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
        labels = list(set(self.dataset["label"]))
    
        sorted_labels = sorted(
            labels,
            key=lambda x: (x != "Non-vul", x)
        )
    
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
