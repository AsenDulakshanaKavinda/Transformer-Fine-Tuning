

from bert.src.models.base_model import BaseTextClassifier  

import torch
from transformers import BertTokenizer, BertForSequenceClassification


class BERTTextClassifier(BaseTextClassifier):
    def __init__(self):
        


        self.preprocess = None
        self.trainer = None
        self.evaluator = None










