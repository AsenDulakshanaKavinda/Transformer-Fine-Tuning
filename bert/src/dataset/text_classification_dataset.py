# dataset class
import sys


import torch
import torch.nn as nn
from torch.utils.data import Dataset

from bert.src.logger import logging as log
from bert.src.exception import ProjectException

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)


    def __getitem__(self, index: int) -> dict:
        text = str(self.texts[index])
        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation = True,
            padding = 'max_length',
            max_length = self.max_length,
            return_tensors = 'pt'
        )

        try:
            log.info(f"Creating encoder...")
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            log.info(f"Error while encoding data: {str(e)}")
            raise ProjectException("Error while encoding data", sys)


        """
        {
            'input_ids': tensor([101, 2023, 2003, 1037, ...]),
            'attention_mask': tensor([1, 1, 1, 0, 0, ...]),
            'labels': tensor(1)
        }

        """


