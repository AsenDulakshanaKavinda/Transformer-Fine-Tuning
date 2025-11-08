from abc import ABC, abstractmethod

class BaseTextClassifier(ABC):

    @abstractmethod
    def load_dataset(self, dataset_name: str="imdb" ,use_sample: bool=True, sample_size=5000):
        """ creating dataset need to train model, sample or full data"""
        pass

    @abstractmethod
    def get_device(self):
        """ setting device available """
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass
