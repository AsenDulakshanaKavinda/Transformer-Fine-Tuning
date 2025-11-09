

from bert.src.models.base_model import BaseTextClassifier  

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from bert.src.components.train import Train
from bert.src.components.evaluate import Evaluate
from bert.src.components.data_loader import DataLoader

from bert.src.logger import logging as log
from bert.src.exception import ProjectException


class BERTTextClassifier(BaseTextClassifier):
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

        self.data_loader = DataLoader()
        self.trainer = Train()
        self.evaluator = Evaluate()

        # share tokenizer, model references with train and eval classes
        self.trainer.tokenizer = self.tokenizer
        self.trainer.model = self.model
        self.evaluator.tokenizer = self.tokenizer
        self.evaluate.model = self.model

    def load_dataset(self, dataset_name = "imdb", use_sample = True, sample_size=5000):
        log.info("Loading dataset...")
        return self.load_dataset(dataset_name, use_sample, sample_size)

    def train(self, train_texts, train_labels):
        log.info("Staring training...")
        return self.trainer.train(train_texts, train_labels)
    
    def evaluate(self, test_texts, test_labels):
        log.info("Starting evaluation...")
        return self.evaluator.evaluate(test_texts, test_labels)









