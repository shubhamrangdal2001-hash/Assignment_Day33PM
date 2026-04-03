"""
Part D: AI-Augmented — Algorithm Selection Guide
--------------------------------------------------
I used AI to generate a decision framework for algorithm selection,
then verified each recommendation and added edge cases from experience.
This script prints the guide and also runs a small interactive demo.
"""

import textwrap


# ─────────────────────────────────────────────────────────────
# Algorithm Selection Guide (AI-generated + human-verified)
# ─────────────────────────────────────────────────────────────

ALGORITHM_GUIDE = {
    "text_data": {
        "recommendation": "TF-IDF + LinearSVC",
        "alternative": "TF-IDF + Logistic Regression (when probabilities needed)",
        "reason": "SVM with linear kernel is the classic text classifier. "
                  "High-dim TF-IDF space is linearly separable — linear models thrive here."
    },
    "small_dataset_high_dim": {
        "condition": "n_samples < 500 AND n_features > n_samples",
        "recommendation": "Logistic Regression (L1) or SVM Linear",
        "alternative": "Naive Bayes as a fast baseline",
        "reason": "L1 LR does automatic feature selection. "
                  "SVM linear only relies on support vectors, not all samples. "
                  "KNN fails here — curse of dimensionality destroys neighbor distance."
    },
    "small_dataset_low_dim": {
        "condition": "n_samples < 500 AND n_features <= n_samples",
        "recommendation": "Naive Bayes, KNN, or Logistic Regression",
        "alternative": "SVM with careful regularization",
        "reason": "Simple, low-parameter models generalize better with small data."
    },
    "medium_dataset_interpretable": {
        "condition": "500 <= n_samples <= 50000 AND need interpretability",
        "recommendation": "Logistic Regression or Decision Tree",
        "alternative": "Logistic Regression preferred — DT overfits without pruning",
        "reason": "LR gives direct coefficient interpretation. "
                  "DT gives visual rules but needs max_depth control."
    },
    "medium_dataset_accuracy": {
        "condition": "500 <= n_samples <= 50000 AND maximize accuracy",
        "recommendation": "SVM (RBF) or XGBoost",
        "alternative": "Random Forest as robust backup",
        "reason": "SVM RBF maps to higher dim for better separation. "
                  "XGBoost wins most Kaggle-style tabular benchmarks."
    },
    "large_dataset": {
        "condition": "n_samples > 50000",
        "recommendation": "XGBoost or Random Forest",
        "alternative": "Logistic Regression if speed at inference matters",
        "reason": "SVM is O(n^2) to O(n^3) in training — too slow at this scale. "
                  "Tree ensembles scale well with parallelism."
    },
    "linear_boundary": {
        "condition": "Data appears linearly separable",
        "recommendation": "Logistic Regression, LinearSVC",
        "reason": "No need for complex kernels. Simpler model = faster + less overfitting."
    },
    "nonlinear_boundary": {
        "condition": "Complex non-linear decision boundary expected",
        "recommendation": "SVM (RBF), Random Forest, or XGBoost",
        "reason": "Kernel trick (SVM) or ensemble splits (RF/XGB) handle non-linearity."
    },
    "need_feature_importance": {
        "condition": "Need to know which features matter",
        "recommendation": "Random Forest or XGBoost",
        "reason": "Both expose .feature_importances_ natively. "
                  "LR coefficients work too but only for linear relationships."
    }
}

# Edge cases the AI missed — added manually after verification
EDGE_CASES_AI_MISSED = [
    {
        "case": "Imbalanced classes",
        "fix": "Set class_weight='balanced' in LR/SVC/RF, or use SMOTE oversampling before training. "
               "AI did not mention this — it makes a huge difference in practice."
    },
    {
        "case": "Time series data",
        "fix": "None of these algorithms apply directly. Need lag features, rolling stats, "
               "or a proper time-series model (ARIMA, LSTM). AI assumed i.i.d. data."
    },
    {
        "case": "Noisy labels",
        "fix": "SVM with hard margin is very sensitive to noise — use small C. "
               "Random Forest and XGBoost are more robust due to averaging over many trees."
    },
    {
        "case": "Deployment latency matters",
        "fix": "KNN is O(n) at predict time — bad for real-time APIs. "
               "Logistic Regression and Naive Bayes are O(1) at inference. "
               "AI only considered training speed, not serving speed."
    },
    {
        "case": "Need calibrated probabilities",
        "fix": "SVM probabilities (via Platt scaling) are poorly calibrated. "
               "Use Logistic Regression instead if confidence scores drive business decisions."
    },
    {
        "case": "Lots of irrelevant features",
        "fix": "LR with L1 does implicit feature selection. "
               "XGBoost also handles this well. KNN is hurt badly by irrelevant features."
    }
]


def print_selection_guide():
    """Prints the full algorithm selection guide in a readable format."""
    print("=" * 65)
    print("  ALGORITHM SELECTION GUIDE — Week 6 ML Algorithms")
    print("  (AI-generated + verified + extended)")
    print("=" * 65)

    for key, entry in ALGORITHM_GUIDE.items():
        print(f"\n[{key.upper().replace('_', ' ')}]")
        if "condition" in entry:
            print(f"  When     : {entry['condition']}")
        print(f"  Use      : {entry['recommendation']}")
        if "alternative" in entry:
            print(f"  Alt      : {entry['alternative']}")
        wrapped = textwrap.fill(entry["reason"], width=60, initial_indent="  Why     : ", subsequent_indent="           ")
        print(wrapped)

    print("\n" + "=" * 65)
    print("  EDGE CASES THE AI MISSED")
    print("=" * 65)
    for item in EDGE_CASES_AI_MISSED:
        print(f"\n  Case: {item['case']}")
        wrapped = textwrap.fill(item["fix"], width=60, initial_indent="  Fix : ", subsequent_indent="        ")
        print(wrapped)


def recommend_algorithm(n_samples, n_features, is_text=False,
                        need_interpretability=False, need_probabilities=False):
    """
    Simple rule-based recommender using the guide above.
    Returns a recommendation string.
    """
    if is_text:
        if need_probabilities:
            return "TF-IDF + Logistic Regression (probabilities needed)"
        return "TF-IDF + LinearSVC"

    if n_samples < 500:
        if n_features > n_samples:
            return "Logistic Regression (L1 penalty) — high-dim, small data"
        return "Naive Bayes or KNN — simple models for small data"

    if n_samples > 50000:
        return "XGBoost or Random Forest — SVM too slow at this scale"

    # Medium dataset
    if need_interpretability:
        return "Logistic Regression — interpretable + generalizes well"
    return "SVM (RBF) or XGBoost — best accuracy on medium tabular data"


if __name__ == "__main__":
    print_selection_guide()

    print("\n\n" + "=" * 65)
    print("  INTERACTIVE DEMO — recommend_algorithm()")
    print("=" * 65)

    test_cases = [
        {"n_samples": 200,   "n_features": 300, "is_text": False, "need_interpretability": False},
        {"n_samples": 5000,  "n_features": 25,  "is_text": False, "need_interpretability": True},
        {"n_samples": 5000,  "n_features": 25,  "is_text": False, "need_interpretability": False},
        {"n_samples": 100000,"n_features": 50,  "is_text": False, "need_interpretability": False},
        {"n_samples": 3000,  "n_features": 10000,"is_text": True,  "need_probabilities": False},
        {"n_samples": 3000,  "n_features": 10000,"is_text": True,  "need_probabilities": True},
    ]

    for i, case in enumerate(test_cases, 1):
        rec = recommend_algorithm(**case)
        print(f"\nCase {i}: n={case['n_samples']}, p={case['n_features']}, "
              f"text={case.get('is_text')}, interpret={case.get('need_interpretability')}, "
              f"proba={case.get('need_probabilities', False)}")
        print(f"  --> {rec}")
