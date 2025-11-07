
import os
import sys
import json
from dotenv import load_dotenv

from finetune.src.logger import logging as log
from finetune.src.exception import ProjectException

from finetune.src.utils.config_loader import load_config

from datasets import load_dataset
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification

load_dotenv()

class ApiKeyManager:
    REQUIRED_KEYS = [""]

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
        
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("RUNNING IN LOCAL MODE: .env loaded")
        else:
            log.info("RUNNING IN PRODUCTION MODE!!!")

        self.api_key_manager = ApiKeyManager()
        self.config = load_config()
        log.info(f"YAML CONFIG LOADED, config_keys={list(self.config.keys())}")



    def load_llm(self):

        model_block = self.config["model"]
        model_type = os.getenv("MODEL_TYPE", "encoder_only")

        if model_type not in model_block:
            log.error(f"Model type not found in config model = {model_type}")
            raise ProjectException(f"Model type '{model_type}' not found in the config")

        model_config = model_block[model_type]
        # model = model_config.get("model")
        model_name = model_config.get("model_name")
        num_labels = model_config.get("num_labels", 2)

        if model_type == "encoder_only":
            return BertForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path = model_name,
                num_labels = num_labels
            )
        else:
            log.error(f"Unsupported model type, model type = {model_type}")
            ProjectException(f"Unsupported model type, {model_type}", sys)

