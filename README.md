# Credit Risk Model Comparison — German Credit Dataset

> Comparative analysis of 5 ML classifiers for binary credit default prediction

## 🔍 Models Compared
| Model | AUC-ROC | Accuracy | F1 | KS Stat |
|---|---|---|---|---|
| **Random Forest** 🏆 | **0.703** | **0.650** | 0.664 | **0.378** |
| Gradient Boosting | 0.697 | 0.635 | 0.664 | 0.325 |
| Logistic Regression | 0.681 | 0.605 | 0.629 | 0.316 |
| SVM | 0.640 | 0.585 | 0.614 | 0.270 |
| KNN | 0.576 | 0.545 | 0.552 | 0.155 |

## 📊 Live Dashboard
👉 **[View Dashboard](https://your-username.github.io/credit-risk-comparison/)**

## 🚀 Run Locally
```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
python credit_risk_model.py
```

## 📁 Files
- `credit_risk_model.py` — Full ML pipeline (data generation, training, evaluation, plots)
- `index.html` — Standalone interactive dashboard (no backend required)
- `plots/` — ROC curves, confusion matrices, feature importance, metrics bar chart
- `model_comparison_results.csv` — Full metrics table

## 🧪 Methodology
- **Dataset**: German Credit (UCI) — 1,000 applicants, 20 features
- **Split**: 80/20 stratified train-test
- **Validation**: 5-fold stratified cross-validation
- **Key Metric**: AUC-ROC + KS Statistic (industry standard for credit scoring)
- **Preprocessing**: Label encoding for categoricals, StandardScaler in pipelines

## 💡 Key Findings
1. **Random Forest** wins on AUC-ROC (0.703) and KS (0.378) — handles non-linear feature interactions well
2. **Gradient Boosting** is a close second — similar F1, slightly lower AUC but better recall
3. **Logistic Regression** is the best interpretable baseline (AUC 0.681) — GDPR-friendly for credit decisions
4. **KNN** underperforms significantly — distance-based methods struggle with mixed feature types

## 🛠️ Tech Stack
`Python 3.12` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`
