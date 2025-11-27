"""
Data loading utilities for CommonsenseQA and related datasets.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple


class MultipleChoiceDataset(torch.utils.data.Dataset):
    """Dataset for multiple choice questions."""

    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


def load_commonsenseqa_data(tokenizer, split='train', max_length=256):
    """
    Load and preprocess CommonsenseQA dataset.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train', 'validation', 'test')
        max_length: Maximum sequence length

    Returns:
        Processed dataset
    """
    dataset = load_dataset('commonsense_qa', split=split)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for example in dataset:
        question = example['question']
        choices = example['choices']['text']

        # Tokenize each choice with the question
        encoded = tokenizer(
            [question] * len(choices),
            choices,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])

        # Get label index
        if 'answerKey' in example and example['answerKey']:
            label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
            label = label_map.get(example['answerKey'], 0)
            labels_list.append(label)
        else:
            labels_list.append(-1)  # For test set without labels

    encodings = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.tensor(labels_list)
    }

    return MultipleChoiceDataset(encodings)


def load_swag_data(tokenizer, split='train', max_length=256):
    """
    Load and preprocess SWAG dataset (related commonsense reasoning dataset).

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train', 'validation')
        max_length: Maximum sequence length

    Returns:
        Processed dataset
    """
    dataset = load_dataset('swag', 'regular', split=split)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for example in dataset:
        # SWAG provides context (sent1 + sent2) and 4 endings
        context = example['sent1'] + ' ' + example['sent2']
        endings = [
            example['ending0'],
            example['ending1'],
            example['ending2'],
            example['ending3']
        ]

        # Tokenize each ending with context
        encoded = tokenizer(
            [context] * 4,
            endings,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])
        labels_list.append(int(example['label']))

    encodings = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.tensor(labels_list)
    }

    return MultipleChoiceDataset(encodings)


def load_hellaswag_data(tokenizer, split='train', max_length=256):
    """
    Load and preprocess HellaSwag dataset (another commonsense reasoning dataset).

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train', 'validation')
        max_length: Maximum sequence length

    Returns:
        Processed dataset
    """
    dataset = load_dataset('Rowan/hellaswag', split=split)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for example in dataset:
        context = example['ctx']
        endings = example['endings']

        # Tokenize each ending with context
        encoded = tokenizer(
            [context] * len(endings),
            endings,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids_list.append(encoded['input_ids'])
        attention_mask_list.append(encoded['attention_mask'])

        # Get label
        label = int(example['label']) if isinstance(example['label'], (int, str)) and str(example['label']).isdigit() else 0
        labels_list.append(label)

    encodings = {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list),
        'labels': torch.tensor(labels_list)
    }

    return MultipleChoiceDataset(encodings)


def get_dataloader(dataset_name: str, tokenizer, split: str, batch_size: int = 8,
                   max_length: int = 256, shuffle: bool = True):
    """
    Get dataloader for a specific dataset.

    Args:
        dataset_name: Name of dataset ('commonsenseqa', 'swag', 'hellaswag')
        tokenizer: HuggingFace tokenizer
        split: Dataset split
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data

    Returns:
        DataLoader
    """
    if dataset_name.lower() == 'commonsenseqa':
        dataset = load_commonsenseqa_data(tokenizer, split, max_length)
    elif dataset_name.lower() == 'swag':
        dataset = load_swag_data(tokenizer, split, max_length)
    elif dataset_name.lower() == 'hellaswag':
        dataset = load_hellaswag_data(tokenizer, split, max_length)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
