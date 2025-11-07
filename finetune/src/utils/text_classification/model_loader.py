
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from finetune.src.logger import logging as log
from finetune.src.exception import ProjectException

from finetune.src.utils.config_loader import load_config

from datasets import load_dataset
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification

load_dotenv()

class ApiKeyManager:
    REQUIRED_KEYS = ["TEST_KEY"]
    log.info(f"{REQUIRED_KEYS}")

    def __init__(self):
        # if env has individual API key
        self.api_keys = {}
        raw = os.getenv("apikeys")

        # if .env has -> dict of api keys
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ProjectException("API_KEYS is not a valid JSON object.")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secrets")
            except Exception as e:
                log.warning(f"Failed to parse API_KEYS as JSON error: {str(e)}")


        # if .env has individual api keys
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env variables")

        # check for missing keys
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error(f"Missing required API keys, missing_keys={missing}")
            raise ProjectException("Missing API keys", sys)
        

class Model_Loader:

    def __init__(self):

        env = os.getenv("ENV", "local").lower()
        
        if env != "production":
            log.info("RUNNING IN LOCAL MODE: .env loaded")
        else:
            log.info("RUNNING IN PRODUCTION MODE!!!")

        self.api_key_manager = ApiKeyManager()
        self.config = load_config()
        log.info(f"YAML CONFIG LOADED, config_keys={list(self.config.keys())}")

    def _get_model_config(self):
        model_block = self.config["model"]
        model_type = os.getenv("MODEL_TYPE", "encoder_only")


        if model_type not in model_block:
            log.error(f"Model type not found in config model = {model_type}")
            raise ProjectException(f"Model type '{model_type}' not found in the config")
        
        return model_type, model_block[model_type]


    def load_model(self):

        model_type, model_config = self._get_model_config()
        model_name = model_config.get("model_name")
        num_labels = model_config.get("num_labels", 2)
        model_dir = Path(os.getenv("MODEL_DIR", model_config.get("model_dir", "./artifacts/model")))

        if model_type != "encoder_only":
            log.error(f"Unsupported model type, model type = {model_type}")
            raise ProjectException(f"Unsupported model type, {model_type}", sys)
        
        model_dir.mkdir(parents=True, exist_ok=True)

        pretrained_model_path = model_dir

        # If directory is empty, download and save
        if not os.listdir(model_dir):
            log.info(f"Local model directory empty, Downloading {model_name}...")
            model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            model.save_pretrained(pretrained_model_path)
            log.info(f"Model saved to {model_dir}")
        else:
            log.info(F"Loading model from lacal directory: {pretrained_model_path}")
            model = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_labels)

        return model

    def load_tokenizer(self):
        model_type, model_config = self._get_model_config()
        tokenizer_name = model_config.get("tokenizer_name")
        tokenizer_dir = Path(os.getenv("TOKENIZER_DIR", model_config.get("model_dir", "./artifacts/tokenizer")))

        if model_type != "encoder_only":
            log.error(f"Unsupported model type, model type = {model_type}")
            raise ProjectException(f"Unsupported model type, {model_type}", sys)
        
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        pretrained_tokenizer_path = tokenizer_dir

        # If directory is empty, download and save
        if not os.listdir(tokenizer_dir):
            log.info(f"Local model directory empty, Downloading {tokenizer_name}...")
            tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
            tokenizer.save_pretrained(pretrained_tokenizer_path)
            log.info(f"Model saved to {pretrained_tokenizer_path}")
        else:
            log.info(F"Loading model from lacal directory: {pretrained_tokenizer_path}")
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_tokenizer_path)

        return tokenizer
# fine-tune and want to reuse that model later
# model.save_pretrained("./artifacts/model/")
# tokenizer.save_pretrained("./artifacts/tokenizer/")
