"""
Sentiment Analysis (Positive/Negative) using scikit-learn
Author: Muhammad Shahzaib Shehzad

This script trains a simple ML pipeline using:
- TF-IDF vectorization for text
- Logistic Regression classifier

It uses a small built-in dataset by default.
You can replace it with your own CSV dataset later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


@dataclass
class Example:
    text: str
    label: str  # "positive" or "negative"


def load_sample_data() -> List[Example]:
    """Small starter dataset. Replace with real data later."""
    positive = [
        "Amazing service and very friendly staff",
        "The food was delicious and arrived quickly",
        "Great experience, I would recommend it",
        "Really happy with the quality, excellent value",
        "The app is easy to use and works perfectly",
        "Fast delivery and the packaging was great",
        "Customer support solved my issue quickly",
        "I enjoyed using this, it is very useful",
    ]
    negative = [
        "Terrible experience, I will not come back",
        "The food was cold and tasted bad",
        "Very disappointing, not worth the money",
        "The app keeps crashing and is slow",
        "Delivery was late and the order was wrong",
        "Poor quality and rude service",
        "I had problems and nobody helped me",
        "This was a waste of time",
    ]

    data: List[Example] = []
    data += [Example(t, "positive") for t in positive]
    data += [Example(t, "negative") for t in negative]
    return data


def build_pipeline() -> Pipeline:
    """Create ML pipeline: TF-IDF -> Logistic Regression."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def train_and_evaluate(texts: List[str], labels: List[str]) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\n✅ Evaluation Results")
    print("Accuracy:", round(accuracy_score(y_test, preds), 3))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds))

    # Quick demo predictions
    demo = [
        "The service was great and the staff were helpful",
        "This is awful, everything was broken and slow",
        "Nice quality and good value for money",
        "Late delivery and the food was cold",
    ]
    print("\n✅ Demo Predictions:")
    for s in demo:
        print(f"- {s}  -->  {model.predict([s])[0]}")


def main() -> None:
    data = load_sample_data()
    texts = [d.text for d in data]
    labels = [d.label for d in data]
    train_and_evaluate(texts, labels)


if __name__ == "__main__":
    main()
