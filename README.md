# D33 | PM | Take-Home Assignment
## SVM, KNN & Full Week Comparison
**Day 33 | Week 6 | Machine Learning & AI | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## What This Assignment Covers

This take-home covers all 8 ML algorithms from Week 6:
- Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- SVM (kernel selection, C/gamma tuning), KNN (optimal K)
- Naive Bayes, XGBoost

Tasks span algorithm comparison, text classification, interview prep, and an AI-augmented algorithm selection guide.

---

## Folder Structure

```
d33-pm-svm-knn/
├── notebooks/
│   └── D33_PM_SVM_KNN_CS.ipynb       # Main Jupyter notebook (all 4 parts)
├── scripts/
│   ├── part_a_algo_comparison.py      # Part A: 8-algorithm CV comparison
│   ├── part_b_text_classification.py  # Part B: TF-IDF + LinearSVC vs LR
│   ├── part_c_interview_ready.py      # Part C: model_selection_report + overfitting
│   └── part_d_algo_selection_guide.py # Part D: AI-augmented algorithm guide
├── solution/
│   └── D33_PM_SVM_KNN_CS_Solution.docx  # Written solution document (all parts)
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install scikit-learn xgboost pandas numpy scipy
```

### 2. Run each part individually

```bash
# Part A — 8 algorithm comparison on Breast Cancer dataset
python scripts/part_a_algo_comparison.py

# Part B — TF-IDF + SVM text classification (downloads 20newsgroups on first run)
python scripts/part_b_text_classification.py

# Part C — model_selection_report() function + SVM overfitting demo
python scripts/part_c_interview_ready.py

# Part D — Algorithm selection guide + interactive recommender
python scripts/part_d_algo_selection_guide.py
```

### 3. Run the full notebook

Open `notebooks/D33_PM_SVM_KNN_CS.ipynb` in Jupyter and run all cells top to bottom.

```bash
jupyter notebook notebooks/D33_PM_SVM_KNN_CS.ipynb
```

---

## Part-wise Summary

### Part A — 8-Algorithm Comparison (40%)
- Dataset: **Breast Cancer** (sklearn), 569 samples, 30 features, binary classification
- All 8 models wrapped in `StandardScaler + Classifier` pipelines
- **5-fold Stratified CV** used for fair comparison (same folds for all models)

| Rank | Algorithm           | Mean CV Accuracy |
|------|---------------------|-----------------|
| 1    | SVM (RBF)           | 0.9772          |
| 2    | Logistic Regression | 0.9737          |
| 3    | KNN (K=5)           | 0.9631          |
| 4    | XGBoost             | 0.9596          |
| 5    | Random Forest       | 0.9561          |
| 6    | Gradient Boosting   | 0.9491          |
| 7    | Naive Bayes         | 0.9297          |
| 8    | Decision Tree       | 0.9280          |

**Recommendation:** SVM (RBF) — best generalization on medium-sized continuous-feature datasets.

---

### Part B — SVM for Text Classification (30%)
- Dataset: **20newsgroups** (4 categories: sci.med, sci.space, talk.politics.guns, rec.sport.hockey)
- Pipeline: `TfidfVectorizer(max_features=20000, sublinear_tf=True, ngram_range=(1,2))` → `LinearSVC`
- Compared against Logistic Regression on the same data
- LinearSVC is marginally better and faster; LR preferred when probabilities are needed

---

### Part C — Interview Ready (20%)

**Q1:** With 100 features and 50 samples, use **Logistic Regression (L1)** or **SVM Linear**.
Avoid KNN (curse of dimensionality), tree ensembles (too few samples), and XGBoost.

**Q2:** `model_selection_report(X, y, models_dict)` — runs stratified CV for any model dict,
returns a ranked DataFrame, and identifies the best model using **paired t-test** on fold scores.

**Q3:** SVM(RBF) train=1.0, test=0.52 is overfitting.
Three fixes: reduce C, use `gamma='scale'`, use GridSearchCV for tuning.

---

### Part D — AI-Augmented Algorithm Guide (10%)
- Used AI to generate a decision framework for algorithm selection
- Verified every recommendation against week's learning
- Identified **6 edge cases the AI missed**: imbalanced classes, time series, noisy labels,
  deployment latency, probability calibration, irrelevant features
- Saved as a reusable `recommend_algorithm()` function in `part_d_algo_selection_guide.py`

---

## Notes
- All random seeds set to `42` for reproducibility
- No test set used for model selection — only CV scores drive decisions
- Part B requires internet access on first run to download the 20newsgroups dataset
