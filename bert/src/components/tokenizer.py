import sys

from transformers import BertTokenizer

from bert.src.logger import logging as log
from bert.src.exception import ProjectException

class Tokenizer:
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased'
        )

    def get_tokenizer(self):
        try:
            log.info(f"Using tokenizer '{self._tokenizer}'")
            return self._tokenizer
        except Exception as e:
            log.error(f"Exception while setting up tokenizer: {str(e)}")
            raise ProjectException(f"Exception while setting up tokenizer: {str(e)}", sys)

