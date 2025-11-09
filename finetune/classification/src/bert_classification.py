# BERT text classifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup # Gradually warms up then decays learning rate for stable BERT training.
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW # Adam Optimizer with Weight Decay
from .text_classification import TextClassificationDataset

class BERTTextClassifier:
    def __init__(self, model_name='bert-base-uncased', num_classes=2, max_length=512):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.isavailable() else "cpu")

        self.tokenizer = BertTokenizerFast.from_pretrained(         
            model_name
        )

        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )

        self.model.to(self.device)



    def load_imdb_data(self, sample_size=5000):
        print("Loading IMDB dataset ...")

        dataset = load_dataset("imdb")

        # sample data from faster training
        train_indices = np.random.choice(
            len(dataset['train']), # n, pick random int from 0 - (n-1)
            min(sample_size, len(dataset['train'])), # num of sample need (this logic - pick only what in dataset )
            replace=False # pick unique samples only
        )

        test_indices = np.random.choice(
            len(dataset['test']),
            min(sample_size//4, len(dataset['test'])),
            replace=False
        )

        # convert numpy.int64 -> int for indexing
        train_texts = [dataset['train'][int(i)]['text'] for i in train_indices]
        train_labels = [dataset['train'][int(i)]['label'] for i in train_indices]

        test_texts = [dataset['test'][int(i)]['text'] for i in test_indices]
        test_labels = [dataset['test'][int(i)]['label'] for i in test_indices]

        print(f"train samples: {len(train_texts)}")
        print(f"test samples: {len(test_texts)}")

        return train_texts, train_labels, test_texts, test_labels


    def train(self, train_texts, train_labels, epochs=1, batch_size=8, learning_rate=2e-5) -> None:
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        ) # get dataset, each get tokenized(truned into token id, attention maske)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # make batches

        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01) # update model weights based on gradients

        total_steps = len(train_loader) * epochs # total number of training steps (batches) across all epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps
        ) # Adjusts learning rate as training progresses

        self.model.train()


        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

            for batch in progress_bar: # batch in train_loader
                optimizer.zero_grad() # clears out previous gradients

                # load everything to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # forward pass
                outputs = self.model(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = labels
                )

                # get the loss
                loss = outputs.loss
                total_loss += loss.item() # add to total loss

                # backward pass -
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # This prevents exploding gradients, especially in large models. If any gradient exceeds 1.0, it’s scaled down proportionally.

                optimizer.step() # update model weight - weight = weight - learning_rate * gradient

                scheduler.step() #

                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

            avg_loss = total_loss/len(train_loader)
            print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')


    def evaluate(self, test_texts, test_labels, batch_size=8):

        test_dataset = TextClassificationDataset(
            test_texts, test_labels, self.tokenizer, self.max_length
        )

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()

        predictions = []

        true_labels = []

        with torch.no_grad(): # prevents PyTorch from building a computational graph (no need backpropagration)
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # - The model predicts logits, which are raw, unnormalized scores (not yet probabilities).
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1).cpu().numpy() # # preds = torch.argmax(logits, dim=1).cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')
            report = classification_report(true_labels, predictions, target_names=['Nagative', 'Positive'])

            return accuracy, f1, report


    def predict(self, texts):
        predictions = [] # what model predict
        probabilities = [] # true values

        self.model.eval()

        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            with torch.no_grad():
                # - The model predicts logits, which are raw, unnormalized scores (not yet probabilities).
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits


                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]


                predictions.append(pred)
                probabilities.append(probs)

        return predictions, probabilities
