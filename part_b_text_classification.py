"""
Part B: SVM for Text Classification
-------------------------------------
Loads 20newsgroups dataset (4 categories), builds a TF-IDF + LinearSVC pipeline,
trains and evaluates it, then compares with Logistic Regression.
"""

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Categories to use
CATEGORIES = [
    'sci.med',
    'sci.space',
    'talk.politics.guns',
    'rec.sport.hockey'
]


def load_news_data(categories):
    """Load 20newsgroups train/test splits, removing headers/footers/quotes."""
    train = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'))
    test  = fetch_20newsgroups(subset='test',  categories=categories,
                                remove=('headers', 'footers', 'quotes'))
    return train, test


def build_svm_pipeline():
    """TF-IDF + LinearSVC pipeline. LinearSVC is the classic SVM for text."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, sublinear_tf=True, ngram_range=(1, 2))),
        ('clf',   LinearSVC(C=1.0, max_iter=2000))
    ])


def build_lr_pipeline():
    """TF-IDF + Logistic Regression pipeline for comparison."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, sublinear_tf=True, ngram_range=(1, 2))),
        ('clf',   LogisticRegression(C=1.0, max_iter=2000, solver='saga'))
    ])


def evaluate_pipeline(pipeline, X_train, y_train, X_test, y_test, target_names, label):
    """Fits pipeline and prints accuracy + classification report."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n=== {label} ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=target_names))
    return acc


if __name__ == "__main__":
    # Load data
    train_data, test_data = load_news_data(CATEGORIES)
    X_train, y_train = train_data.data, train_data.target
    X_test,  y_test  = test_data.data,  test_data.target

    print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    print(f"Categories: {train_data.target_names}")

    # SVM
    svm_acc = evaluate_pipeline(
        build_svm_pipeline(),
        X_train, y_train, X_test, y_test,
        train_data.target_names, "TF-IDF + LinearSVC"
    )

    # Logistic Regression
    lr_acc = evaluate_pipeline(
        build_lr_pipeline(),
        X_train, y_train, X_test, y_test,
        train_data.target_names, "TF-IDF + Logistic Regression"
    )

    # Summary
    print("\n=== Comparison Summary ===")
    print(f"LinearSVC          : {svm_acc:.4f}")
    print(f"Logistic Regression: {lr_acc:.4f}")
    winner = "LinearSVC" if svm_acc >= lr_acc else "Logistic Regression"
    print(f"\nWinner: {winner}")
    print("Note: LinearSVC is typically faster and slightly better on text classification.")
    print("LR is preferred when you need probability outputs (e.g., ranking articles).")
