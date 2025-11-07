





# from finetune.src.utils.config_loader import test
from finetune.src.utils.text_classification.model_loader import Model_Loader, ApiKeyManager

def test():
    api = ApiKeyManager()
    ml = Model_Loader()
    ml.load_tokenizer()


test()

