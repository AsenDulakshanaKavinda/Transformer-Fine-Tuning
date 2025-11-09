# dataset class
import torch
from torch.utils.data import Dataset

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

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        """
        {
            'input_ids': tensor([101, 2023, 2003, 1037, ...]),
            'attention_mask': tensor([1, 1, 1, 0, 0, ...]),
            'labels': tensor(1)
        }

        """


