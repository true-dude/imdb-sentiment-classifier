import os

import torch
from torch.utils.data import Dataset


class IMDBDataset(Dataset):
    def __init__(
        self, positive_path: str, negative_path: str, tokenizer, max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        if not os.path.exists(positive_path):
            raise FileNotFoundError(f"Файл {positive_path} не найден")
        if not os.path.exists(negative_path):
            raise FileNotFoundError(f"Файл {negative_path} не найден")

        with open(positive_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    self.samples.append((text, 1))

        with open(negative_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    self.samples.append((text, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoded = self.tokenizer.encode(text, return_attention_mask=True)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids = (
                input_ids
                + [self.tokenizer.token_to_idx[self.tokenizer.pad_token]] * pad_len
            )
            attention_mask = attention_mask + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }
