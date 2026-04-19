"""
Credit Risk Model Comparison — German Credit Dataset
======================================================
Models: Logistic Regression, Random Forest, XGBoost, SVM, KNN
Metrics: Accuracy, AUC-ROC, F1, Precision, Recall, KS Statistic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os

# ── 1. Generate German Credit-style Dataset ──────────────────────────────────
print("Generating German Credit Dataset (mirroring UCI structure)...")
np.random.seed(42)
N = 1000

# German Credit features (20 attributes mirroring UCI spec)
duration        = np.random.choice([6,12,18,24,36,48], N)
credit_amount   = np.random.lognormal(7.5, 0.9, N).astype(int)
installment_rate= np.random.choice([1,2,3,4], N)
residence_since = np.random.choice([1,2,3,4], N)
age             = np.random.randint(19, 75, N)
existing_credits= np.random.choice([1,2,3,4], N)
num_dependents  = np.random.choice([1,2], N, p=[0.7,0.3])
checking_acct   = np.random.choice([0,1,2,3], N, p=[0.27,0.27,0.27,0.19])
credit_history  = np.random.choice([0,1,2,3,4], N)
purpose         = np.random.choice(range(10), N)
savings_acct    = np.random.choice([0,1,2,3,4], N, p=[0.10,0.10,0.10,0.10,0.60])
employment      = np.random.choice([0,1,2,3,4], N, p=[0.07,0.22,0.20,0.29,0.22])
personal_status = np.random.choice([0,1,2,3], N)
other_debtors   = np.random.choice([0,1,2], N, p=[0.67,0.04,0.29])
property        = np.random.choice([0,1,2,3], N)
other_install   = np.random.choice([0,1,2], N, p=[0.81,0.06,0.13])
housing         = np.random.choice([0,1,2], N, p=[0.11,0.71,0.18])
job             = np.random.choice([0,1,2,3], N, p=[0.02,0.20,0.63,0.15])
telephone       = np.random.choice([0,1], N, p=[0.40,0.60])
foreign_worker  = np.random.choice([0,1], N, p=[0.04,0.96])

X = pd.DataFrame({
    "checking_account": checking_acct,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "credit_amount": credit_amount,
    "savings_account": savings_acct,
    "employment": employment,
    "installment_rate": installment_rate,
    "personal_status": personal_status,
    "other_debtors": other_debtors,
    "residence_since": residence_since,
    "property": property,
    "age": age,
    "other_installment": other_install,
    "housing": housing,
    "existing_credits": existing_credits,
    "job": job,
    "num_dependents": num_dependents,
    "telephone": telephone,
    "foreign_worker": foreign_worker,
})

# Build target with realistic credit logic (70/30 good/bad split)
score = (
    - 0.3  * (checking_acct < 2)
    - 0.25 * (duration > 24)
    - 0.2  * (credit_history < 2)
    - 0.25 * (np.log1p(credit_amount) - 7.5)
    + 0.2  * (savings_acct > 2)
    + 0.15 * (employment > 2)
    + 0.1  * (age > 35)
    + np.random.normal(0, 0.6, N)
)
y = pd.Series((score < 0).astype(int))
print(f"Dataset shape: {X.shape}")
print(f"Class distribution:\n{y.value_counts()}\n")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

# ── 3. Define Models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.1, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42
        ))
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=11, metric="minkowski"))
    ]),
}

# ── 4. Train & Evaluate ───────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=" * 60)
print(f"{'Model':<22} {'Acc':>6} {'AUC':>6} {'F1':>6} {'Prec':>6} {'Rec':>6}")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    f1    = f1_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred)
    rec   = recall_score(y_test, y_pred)

    # KS Statistic
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ks = np.max(tpr - fpr)

    # CV AUC
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

    results[name] = {
        "Accuracy": acc, "AUC-ROC": auc, "F1": f1,
        "Precision": prec, "Recall": rec,
        "KS Statistic": ks, "CV AUC": cv_auc,
        "y_prob": y_prob, "fpr": fpr, "tpr": tpr
    }

    print(f"{name:<22} {acc:>6.3f} {auc:>6.3f} {f1:>6.3f} {prec:>6.3f} {rec:>6.3f}")

print("=" * 60)

# ── 5. Plots ──────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# -- ROC Curves
plt.figure(figsize=(8, 6))
colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261"]
for (name, res), color in zip(results.items(), colors):
    plt.plot(res["fpr"], res["tpr"],
             label=f"{name} (AUC={res['AUC-ROC']:.3f})", color=color, lw=2)
plt.plot([0,1],[0,1],"k--", alpha=0.4)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Credit Risk Models")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=150)
plt.close()
print("Saved: plots/roc_curves.png")

# -- Metrics Bar Chart
metrics_to_plot = ["Accuracy", "AUC-ROC", "F1", "Precision", "Recall"]
df_metrics = pd.DataFrame(
    {name: {m: res[m] for m in metrics_to_plot} for name, res in results.items()}
).T

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(metrics_to_plot))
width = 0.15
for i, (name, row) in enumerate(df_metrics.iterrows()):
    ax.bar(x + i * width, row.values, width, label=name, color=colors[i])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metrics_to_plot)
ax.set_ylim(0, 1.1)
ax.set_title("Model Performance Comparison")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("plots/metrics_comparison.png", dpi=150)
plt.close()
print("Saved: plots/metrics_comparison.png")

# -- Feature Importance (Random Forest)
rf_model = models["Random Forest"].named_steps["clf"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top15 = importances.nlargest(15)

plt.figure(figsize=(8, 5))
top15.sort_values().plot(kind="barh", color="#457b9d")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()
print("Saved: plots/feature_importance.png")

# -- Confusion Matrices
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
for ax, (name, res) in zip(axes, results.items()):
    y_pred = models[name].predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Good","Bad"], yticklabels=["Good","Bad"])
    ax.set_title(name, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150)
plt.close()
print("Saved: plots/confusion_matrices.png\n")

# ── 6. Summary Table ──────────────────────────────────────────────────────────
summary_cols = ["Accuracy", "AUC-ROC", "F1", "Precision", "Recall", "KS Statistic", "CV AUC"]
summary = pd.DataFrame(
    {name: {m: round(res[m], 4) for m in summary_cols} for name, res in results.items()}
).T
summary.index.name = "Model"
print("\nFull Summary Table:")
print(summary.to_string())
summary.to_csv("model_comparison_results.csv")
print("\nSaved: model_comparison_results.csv")

# Save best model
best = summary["AUC-ROC"].idxmax()
joblib.dump(models[best], "best_model.pkl")
print(f"\nBest Model by AUC-ROC: {best}")
print(f"Saved: best_model.pkl")
