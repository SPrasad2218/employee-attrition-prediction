"""
Employee Attrition Prediction
-----------------------------
Run:  python attrition_project.py
Needs: data/WA_Fn-UseC_-HR-Employee-Attrition.csv  (IBM HR Analytics dataset from Kaggle)
       or your own CSV: python attrition_project.py path/to/file.csv  (target column must be 'Attrition' with Yes/No)

Sections follow the 10-day plan:
  1. Load + inspect      2. EDA        3. Preprocessing + split
  4. Baseline + Random Forest (class imbalance handled)
  5. Metrics             6. Cross-validation + tuning
  7. Feature importance  8. Save model
"""
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save plots to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------- 1. LOAD + INSPECT
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))
df = df.drop_duplicates()

# Drop columns that carry no information (constants / IDs) if they exist
useless = [c for c in ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"] if c in df.columns]
df = df.drop(columns=useless)

y = (df["Attrition"] == "Yes").astype(int)
X = df.drop(columns="Attrition")
print(f"Attrition rate: {y.mean():.1%}  (imbalanced -> accuracy alone is misleading)")

# ---------------------------------------------------------------- 2. EDA
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, col in zip(axes.ravel(), ["OverTime", "Department", "JobSatisfaction", "WorkLifeBalance"]):
    if col in df.columns:
        rate = df.groupby(col)["Attrition"].apply(lambda s: (s == "Yes").mean()).sort_values()
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, color="#4C72B0")
        ax.set_title(f"Attrition rate by {col}")
        ax.set_ylabel("Attrition rate")
        ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(OUT / "01_eda_attrition_rates.png", dpi=150)
plt.close()

num_cols_all = X.select_dtypes(include="number").columns
plt.figure(figsize=(11, 9))
sns.heatmap(X[num_cols_all].corr(), cmap="coolwarm", center=0)
plt.title("Correlation of numeric features")
plt.tight_layout()
plt.savefig(OUT / "02_correlation_heatmap.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- 3. PREPROCESS + SPLIT
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# stratify=y keeps the same attrition ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------- 4. MODELS
# class_weight="balanced" handles the imbalance by penalising mistakes on the minority class more.
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    ),
}
pipes = {name: Pipeline([("prep", preprocess), ("model", m)]) for name, m in models.items()}


# ---------------------------------------------------------------- 5. EVALUATE
def evaluate(name, pipe, tag):
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    print(f"\n===== {name} ({tag}) =====")
    print(classification_report(y_test, pred, target_names=["Stayed", "Left"]))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stayed", "Left"], cmap="Blues")
    plt.title(f"Confusion matrix - {name} ({tag})")
    plt.savefig(OUT / f"cm_{name.replace(' ', '_')}_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    return {
        "Model": f"{name} ({tag})",
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }


results = [evaluate(n, p, "baseline") for n, p in pipes.items()]

# ---------------------------------------------------------------- 6. CROSS-VALIDATION + TUNING
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("\n===== 5-fold cross-validation (ROC-AUC, training data only) =====")
for name, pipe in pipes.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"{name}: {scores.mean():.3f} +/- {scores.std():.3f}")

param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [5, 10, None],
    "model__min_samples_leaf": [1, 3, 5],
}
grid = GridSearchCV(pipes["Random Forest"], param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
print("\nBest params:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 3))
best = grid.best_estimator_
results.append(evaluate("Random Forest", best, "tuned"))

# Compare all models on the same test set
summary = pd.DataFrame(results).round(3)
print("\n===== Model comparison (test set) =====")
print(summary.to_string(index=False))
summary.to_csv(OUT / "model_comparison.csv", index=False)

# ROC curves
plt.figure(figsize=(6, 5))
ax = plt.gca()
for name, pipe in {"Logistic Regression": pipes["Logistic Regression"], "Random Forest (tuned)": best}.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name, ax=ax)
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.title("ROC curves")
plt.savefig(OUT / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- 7. FEATURE IMPORTANCE
# Permutation importance on the ORIGINAL columns -> easy to explain to non-technical people.
perm = permutation_importance(
    best, X_test, y_test, scoring="roc_auc", n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1
)
imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("\n===== Top 10 attrition drivers (permutation importance) =====")
print(imp.head(10).round(4).to_string())

plt.figure(figsize=(8, 5))
sns.barplot(x=imp.head(10).values, y=imp.head(10).index, color="#C44E52")
plt.title("Top 10 drivers of attrition (drop in ROC-AUC when shuffled)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()

# Optional: SHAP (pip install shap). Skipped automatically if not installed.
try:
    import shap

    Xt = best.named_steps["prep"].transform(X_test)
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    names = best.named_steps["prep"].get_feature_names_out()
    explainer = shap.TreeExplainer(best.named_steps["model"])
    sv = explainer.shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved SHAP summary plot.")
except ImportError:
    print("(shap not installed - skipping SHAP plot. Optional: pip install shap)")

# ---------------------------------------------------------------- 8. SAVE MODEL (used by app.py)
defaults = {c: float(X_train[c].median()) for c in num_cols}
defaults.update({c: X_train[c].mode()[0] for c in cat_cols})
joblib.dump(
    {
        "model": best,
        "defaults": defaults,
        "columns": X.columns.tolist(),
        "choices": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
    },
    OUT / "attrition_model.joblib",
)
print(f"\nDone. Plots, metrics and model saved in ./{OUT}/")
