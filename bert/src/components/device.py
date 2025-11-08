import torch
import sys

from bert.src.logger import logging as log
from bert.src.exception import ProjectException

class Device:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_device(self):
        try:
            log.info(f"Using device '{self._device}'")
            return self._device
        except Exception as e:
            log.error(f"Exception while setting up device: {str(e)}")
            raise ProjectException(f"Exception while setting up device: {str(e)}", sys)

