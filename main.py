import os
import pandas as pd
import numpy as np
import re
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    words = text.split()
    cleaned = [w for w in words if len(w) > 2]
    if not cleaned:
        return "news"
    return " ".join(cleaned)


def load_dataset():
    fake_path = "Fake.csv"
    true_path = "True.csv"
    if not os.path.isfile(fake_path) or not os.path.isfile(true_path):
        raise FileNotFoundError(
            f"Missing required dataset file(s): {fake_path}, {true_path}"
        )

    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake["label"] = 0
    true["label"] = 1
    data = pd.concat([fake, true], axis=0, ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    if "title" in data.columns:
        data["title"] = data["title"].fillna("")
    else:
        data["title"] = ""

    data["text"] = data["text"].fillna("")
    data["text"] = data["title"] + " " + data["text"]
    data = data[["text", "label"]]
    data = data.dropna()
    data["text"] = data["text"].apply(clean_text)
    return data


def split_data(data):
    X = data["text"]
    y = data["label"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )


def vectorize_data(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("Model Accuracy:", acc)
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)
    print("Unique predictions:", np.unique(predictions))
    return cm


def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.tight_layout()
    backend = matplotlib.get_backend().lower()
    if backend == "agg" or not plt.isinteractive():
        fig_path = "confusion_matrix.png"
        plt.savefig(fig_path)
        print(f"Saved confusion matrix to {fig_path}")
    else:
        plt.show()
    plt.close()


def predict_input(model, vectorizer):
    try:
        while True:
            text = input("Enter news text (or type exit): ")
            if text.strip().lower() == "exit":
                break
            text_clean = clean_text(text)
            vec = vectorizer.transform([text_clean])
            pred = model.predict(vec)[0]
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(vec)[0]
                print(f"Fake probability: {prob[0]:.4f}")
                print(f"Real probability: {prob[1]:.4f}")
            if pred == 1:
                print("Prediction: Real News")
            else:
                print("Prediction: Fake News")
            print("-" * 50)
    except (EOFError, KeyboardInterrupt):
        print("\nPrediction loop ended.")


def main():
    print("Loading data...")
    data = load_dataset()
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(data)
    print("Vectorizing...")
    X_train_vec, X_test_vec, vectorizer = vectorize_data(X_train, X_test)
    print("Training model...")
    model = train_model(X_train_vec, y_train)
    print("Evaluating model...")
    cm = evaluate_model(model, X_test_vec, y_test)
    print("Plotting confusion matrix...")
    plot_confusion_matrix(cm)
    print("Ready for predictions")
    predict_input(model, vectorizer)


if __name__ == "__main__":
    main()
