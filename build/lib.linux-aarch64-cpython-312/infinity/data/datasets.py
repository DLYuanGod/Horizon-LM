"""Dataset implementations for training."""

import torch
from torch.utils.data import Dataset


class MetaMathDataset(Dataset):
    """MetaMath dataset for mathematical reasoning training.

    Args:
        dataset_path: Path to the dataset (HuggingFace datasets format)
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
    """

    def __init__(self, dataset_path, tokenizer, max_seq_len: int):
        from datasets import load_from_disk
        self.dataset = load_from_disk(dataset_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["response"]}
        ]

        # Apply chat template (already contains special tokens)
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encoded = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False  # chat template already has special tokens
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Tokenize prompt with SAME settings
        prompt_messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": example["query"]},
        ]
        prompt_text = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_encoded = self.tokenizer(
            prompt_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=False  # chat template already has special tokens
        )
        prompt_length = int(prompt_encoded["attention_mask"].sum().item())

        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": prompt_length
        }


def collate_fn(batch):
    """Collate function for batching dataset samples.

    Args:
        batch: List of dataset samples

    Returns:
        Dictionary with batched tensors
    """
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "prompt_length": torch.tensor([x["prompt_length"] for x in batch])
    }
