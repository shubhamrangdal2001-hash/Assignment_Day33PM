"""
Part A: 8-Algorithm Comparison
--------------------------------
Loads breast cancer dataset, runs all 8 Week 6 algorithms with proper
5-fold stratified CV, and ranks them.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier


def build_pipelines():
    """Returns a dict of model name -> sklearn Pipeline (with StandardScaler)."""
    return {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(C=1.0, kernel='rbf', gamma='scale'))
        ]),
        "KNN (K=5)": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ]),
        "Naive Bayes": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ]),
        "XGBoost": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(n_estimators=100, learning_rate=0.1,
                                   random_state=42, eval_metric='logloss', verbosity=0))
        ])
    }


def run_comparison(X, y, models, n_splits=5):
    """
    Runs stratified CV for each model and returns a ranked DataFrame.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for name, pipeline in models.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        rows.append({
            'Algorithm': name,
            'Mean CV Accuracy': round(scores.mean(), 4),
            'Std Dev': round(scores.std(), 4)
        })
        print(f"{name:25s} | Acc: {scores.mean():.4f} ± {scores.std():.4f}")

    result_df = (pd.DataFrame(rows)
                   .sort_values('Mean CV Accuracy', ascending=False)
                   .reset_index(drop=True))
    result_df.index += 1
    result_df.index.name = 'Rank'
    return result_df


def print_recommendation(ranked_df):
    best = ranked_df.iloc[0]
    print(f"\nRecommended Model: {best['Algorithm']}")
    print(f"  Mean CV Accuracy: {best['Mean CV Accuracy']}")
    print("  Reasoning: Best balance of accuracy and generalization on this dataset.")
    print("  Proper scaling + CV ensures a fair comparison across all 8 algorithms.")


if __name__ == "__main__":
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset: Breast Cancer | Samples: {X.shape[0]} | Features: {X.shape[1]}\n")

    # Build and run
    models = build_pipelines()
    print("Running 5-fold Stratified CV for all 8 algorithms...")
    ranked = run_comparison(X, y, models)

    print("\n=== Final Rankings ===")
    print(ranked.to_string())
    print_recommendation(ranked)
