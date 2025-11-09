# main.py
from bert.src.models.bert_text_classifier import BERTTextClassifier

if __name__ == "__main__":
    model = BERTTextClassifier()

    # 1️⃣ Load dataset
    train_texts, train_labels, test_texts, test_labels = model.load_dataset(
        dataset_name="imdb",
        use_sample=True,
        sample_size=2000
    )

    # 2️⃣ Train the model
    model.train(train_texts, train_labels)

    # 3️⃣ Evaluate
    accuracy, f1, report = model.evaluate(test_texts, test_labels)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    # 4️⃣ Predict a single example
    pred, probs = model.predict("This movie was awesome!")
    print("Prediction:", "Positive" if pred == 1 else "Negative")
    print("Probabilities:", probs)
