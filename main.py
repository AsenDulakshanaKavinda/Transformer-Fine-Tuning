# main.py

from bert.src.models.bert_text_classifier import BERTTextClassifier

if __name__ == "__main__":
    model = BERTTextClassifier()

    # Load dataset
    train_texts, train_labels, test_texts, test_labels = model.load_dataset(
        use_sample=True, sample_size=2000
    )

    # Train model
    model.train(train_texts, train_labels)

    # Evaluate
    accuracy, f1, report = model.evaluate(test_texts, test_labels)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(report)

    # Predict
    sample_text = "The movie was absolutely fantastic!"
    result = model.predict(sample_text)
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
