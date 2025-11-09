
import torch
from bert.src.logger import logging as log
from bert.src.exception import ProjectException


class Predict:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def predict(self, text: str):
        try:
            log.info("Strting prediction process...")

            self.model.eval()

            # Tokenize the input text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )

            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()

            label_map = {0: "Negative", 1: "Positive"}
            prediction = label_map.get(pred_label, "Unknown")

            log.info(f"Prediction: {prediction}, Confidence: {probs.max().item():.4f}")
            return {
                "text": text,
                "prediction": prediction,
                "confidence": probs.max().item(),
                "probabilities": probs.cpu().numpy()
            }

        except Exception as e:
            log.error(f"Error during prediction: {str(e)}")
            raise ProjectException("Prediction failed", e)
