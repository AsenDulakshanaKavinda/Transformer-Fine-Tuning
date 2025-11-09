from finetune.classification.src.bert_classification import BERTTextClassifier

def run_text_classification_demo():

    """Demo for text classification"""

    print("\n" + "="*60)
    print("TEXT CLASSIFICATION (Sentiment Analysis) DEMO")
    print("="*60)

    classifier = BERTTextClassifier()

    # Load data
    train_texts, train_labels, test_texts, test_labels = classifier.load_imdb_data(sample_size=1000)

    # Show sample
    print(f"\nSample Review: {train_texts[0][:200]}...")
    print(f"Label: {'Positive' if train_labels[0] == 1 else 'Negative'}")

    # Train for 2 epochs (small for demo)
    classifier.train(train_texts, train_labels, epochs=1, batch_size=8)

    # Evaluate
    accuracy, f1, report = classifier.evaluate(test_texts, test_labels, batch_size=8)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Test custom examples
    custom_reviews = [
        "This movie was fantastic! Amazing acting and great plot.",
        "Boring and terrible. Waste of time.",
        "Not bad, could be better though."
    ]

    predictions, probabilities = classifier.predict(custom_reviews)

    print(f"\nCustom Predictions:")

    for text, pred, prob in zip(custom_reviews, predictions, probabilities):

        sentiment = "Positive" if pred == 1 else "Negative"

        confidence = prob[pred] * 100

        print(f"'{text[:50]}...' -> {sentiment} ({confidence:.1f}%)")
