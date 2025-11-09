
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from bert.src.logger import logging as log
from bert.src.exception import ProjectException


class DataLoader:
    def __init__(self):
        self.dataset_name = "imdb"
        self.use_sample = True
        self.sample_size = 5000
        
        

    def load_dataset(self, dataset_name: str="imdb" ,use_sample: bool=True, sample_size=5000):
        
        try:
            dataset = load_dataset(dataset_name)
            if use_sample:
                return self._get_sample_dataset()
            else:
                log.info(f"creating sample dataset...")
                train_texts = dataset['train']['text']
                train_labels = dataset['train']['label']

                test_texts = dataset['test']['text']
                test_labels = dataset['test']['label']
                log.info(f"dataset created, train size: {len(train_texts)}, test size{len(test_texts)}")
                return train_texts, train_labels, test_texts, test_labels 
            

        except Exception as e:
            log.error(f"Error while creating dataset.")
            raise Exception(f"Error while creating dataset: {str(e)}", sys)

    def _get_sample_indices(self, dataset, sample_size):
        train_indices = np.random.choice(
            len(dataset['train']),
            min(sample_size, len(dataset['train'])),
            replace=False
        ).tolist()

        test_indices = np.random.choice(
            len(dataset['test']),
            min(sample_size, len(dataset['test'])),
            replace=False
        ).tolist()

        return train_indices, test_indices
    
    def _get_sample_dataset(self, dataset):

        try:
            log.info(f"creating sample dataset...")
            train_indices, test_indices = self._get_sample_indices(dataset=dataset)

            train_texts = [dataset['train'][int(i)]['text'] for i in train_indices]
            train_labels = [dataset['train'][int(i)]['label'] for i in train_indices]

            test_texts = [dataset['test'][int(i)]['text'] for i in test_indices]
            test_labels = [dataset['test'][int(i)]['label'] for i in test_indices]

            log.info(f"sample dataset created, train size: {len(train_texts)}, test size{len(test_texts)}")

            return train_texts, train_labels, test_texts, test_labels
        
        except Exception as e:
            log.error(f"Error while creating sample dataset.")
            raise Exception(f"Error while creating sample dataset: {str(e)}", sys)