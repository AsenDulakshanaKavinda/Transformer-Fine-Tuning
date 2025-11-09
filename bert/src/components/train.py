import sys

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW #
from transformers import get_linear_schedule_with_warmup

from bert.src.logger import logging as log
from bert.src.exception import ProjectException
from bert.src.dataset.text_classification_dataset import TextClassificationDataset
from bert.src.utils.device import Device





class Train:
    def __init__(self):
        self.epochs = 1
        self.batch_size = 8
        self.learning_rate = 2e-5
        self.tokenizer = None
        self.model = None

    def train(self, train_texts, train_labels) -> None:

        log.info(f"loading dataset for train")
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, max_length=5000
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        log.info(f"loading optimizer (adam) for train")
        optimizer = AdamW(self.model.parameters(), 
                          lr=self.learning_rate, 
                          weight_decay=0.01)
        
        log.info(f"setting up scheduler")
        total_steps = len(train_loader) * self.epochs # total number of training steps (batches) across all epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps
        ) # Adjusts learning rate as training progresses

        log.info(f"setting up the device to train")
        device = Device().get_device()

        self.model.train()

        for epoch in range(self.epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}')


            for batch in progress_bar: # batch in train_loader
                optimizer.zero_grad() # clears out previous gradients

                # loading to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # forward pass
                outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # backward pass
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # This prevents exploding gradients, especially in large models. If any gradient exceeds 1.0, it’s scaled down proportionally.

                optimizer.step() # update model weight - weight = weight - learning_rate * gradient

                scheduler.step() #

                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = total_loss/len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')






