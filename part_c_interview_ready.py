"""
Part C: Interview Ready
------------------------
Q2: model_selection_report() function
Q3: Demonstrate SVM overfitting and 3 fixes
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────
# Q2: model_selection_report
# ─────────────────────────────────────────────

def model_selection_report(X, y, models_dict, n_splits=5, random_state=42):
    """
    Runs stratified CV for each model in models_dict, returns a formatted
    DataFrame of results, and identifies the statistically best model
    using paired t-test on fold scores.

    Parameters:
        X           : numpy array, feature matrix
        y           : numpy array, target labels
        models_dict : dict {model_name: sklearn estimator/pipeline}
        n_splits    : number of CV folds (default 5)
        random_state: seed for reproducibility

    Returns:
        summary_df  : pd.DataFrame with columns [Model, Mean Accuracy, Std Dev, Min, Max]
    """
    # Scale features for models that need it
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Collect fold scores for each model
    fold_scores_map = {}
    for name, model in models_dict.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv_strategy, scoring='accuracy')
        fold_scores_map[name] = scores

    # Build summary DataFrame
    rows = []
    for name, scores in fold_scores_map.items():
        rows.append({
            'Model':          name,
            'Mean Accuracy':  round(scores.mean(), 4),
            'Std Dev':        round(scores.std(),  4),
            'Min Fold':       round(scores.min(),  4),
            'Max Fold':       round(scores.max(),  4)
        })

    summary_df = (pd.DataFrame(rows)
                    .sort_values('Mean Accuracy', ascending=False)
                    .reset_index(drop=True))

    print("=== Model Selection Report ===")
    print(summary_df.to_string(index=False))

    # Best model
    best_name   = summary_df.iloc[0]['Model']
    best_scores = fold_scores_map[best_name]
    print(f"\nBest model (highest mean CV accuracy): {best_name}")

    # Paired t-test: best vs all others
    print("\n--- Paired t-test (best vs each other model) ---")
    for name, scores in fold_scores_map.items():
        if name == best_name:
            continue
        t_stat, p_val = ttest_rel(best_scores, scores)
        tag = "SIGNIFICANT" if p_val < 0.05 else "not significant"
        print(f"  {best_name} vs {name:25s} | t={t_stat:+.3f} | p={p_val:.4f} ({tag})")

    return summary_df


# ─────────────────────────────────────────────
# Q3: SVM Overfitting — demonstrate and fix
# ─────────────────────────────────────────────

def demonstrate_overfitting():
    """Shows SVM(RBF) overfitting and 3 fixes."""

    # Small dataset to make overfitting obvious
    X, y = make_classification(n_samples=200, n_features=20,
                                n_informative=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n=== Q3: SVM Overfitting Demo ===")

    # Overfit model: very large C + large gamma
    overfit_svm = SVC(C=1000, kernel='rbf', gamma=1.0)
    overfit_svm.fit(X_train_s, y_train)
    train_acc = overfit_svm.score(X_train_s, y_train)
    test_acc  = overfit_svm.score(X_test_s,  y_test)
    print(f"Overfit SVM  | Train: {train_acc:.4f} | Test: {test_acc:.4f}  <-- BAD GAP")

    # Fix 1: Reduce C
    fix1_svm = SVC(C=0.1, kernel='rbf', gamma=1.0)
    fix1_svm.fit(X_train_s, y_train)
    print(f"Fix 1 (C=0.1)| Train: {fix1_svm.score(X_train_s, y_train):.4f} | "
          f"Test: {fix1_svm.score(X_test_s, y_test):.4f}")

    # Fix 2: Use gamma='scale' instead of large gamma
    fix2_svm = SVC(C=1000, kernel='rbf', gamma='scale')
    fix2_svm.fit(X_train_s, y_train)
    print(f"Fix 2 (g=scl)| Train: {fix2_svm.score(X_train_s, y_train):.4f} | "
          f"Test: {fix2_svm.score(X_test_s, y_test):.4f}")

    # Fix 3: GridSearchCV to tune both C and gamma
    param_grid = {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 'auto', 0.01]}
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=cv_strategy,
                                scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_s, y_train)
    best_svm = grid_search.best_estimator_
    print(f"Fix 3 (GridCV)| Train: {best_svm.score(X_train_s, y_train):.4f} | "
          f"Test: {best_svm.score(X_test_s, y_test):.4f} | "
          f"Best params: {grid_search.best_params_}")

    print("\nSummary of Fixes:")
    print("  Fix 1: Lower C -> softer margin -> less memorization")
    print("  Fix 2: Use gamma='scale' -> avoids overly local decision boundary")
    print("  Fix 3: GridSearchCV -> principled C/gamma tuning via CV, not guessing")


if __name__ == "__main__":
    # Run Q2
    print("─" * 50)
    print("Q2: model_selection_report() Demo")
    print("─" * 50)

    X_bc, y_bc = load_breast_cancer(return_X_y=True)

    models_to_compare = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)':           SVC(kernel='rbf', gamma='scale')
    }

    report = model_selection_report(X_bc, y_bc, models_to_compare)

    # Run Q3
    print("\n" + "─" * 50)
    print("Q3: SVM Overfitting Demo & Fixes")
    print("─" * 50)
    demonstrate_overfitting()
