

import sys

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW 

from bert.src.logger import logging as log
from bert.src.exception import ProjectException
from bert.src.dataset.text_classification_dataset import TextClassificationDataset
from bert.src.utils.device import Device

from sklearn.metrics import accuracy_score, classification_report, f1_score



class Evaluate:
    def __init__(self):
        pass


    def evaluate(self, test_texts, test_labels, batch_size=8):
        log.info(f"Loading dataset for evaluation")
        test_dataset = TextClassificationDataset(
            test_texts, test_labels, self.tokenizer, max_length=5000
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        self.model.eval()

        predictions = []

        true_labels = []

        log.info(f"setting up the device to train")
        device = Device().get_device()

        log.info(f"staring evaluation process")
        with torch.no_grad(): # prevents PyTorch from building a computational graph (no need backpropagration)
            progress_bar = tqdm(test_loader, desc="Evaluating")
            for batch in progress_bar:
                # loading to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # the model predicts logists, which are raw, unnormalized scores (not yet probabilities).
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1).cpu().numpy()

                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

            accuracy = accuracy_score(true_labels, predictions)
            f1 = f1_score(true_labels, predictions, average='weighted')
            report = classification_report(true_labels, predictions, target_names=['Nagative', 'Positive'])

            return accuracy, f1, report
