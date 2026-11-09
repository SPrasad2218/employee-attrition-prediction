# Employee Attrition Prediction

Predicting which employees are likely to leave, and the factors behind it, using classification models in Python.

## Problem
Employee turnover is costly. This project builds and compares ML models that predict attrition and identifies its main drivers, so HR teams can act early.

## Data
IBM HR Analytics Employee Attrition dataset ([N] employees, [N] features). Attrition rate: [X]% (imbalanced).

## Approach
1. Cleaning and EDA (attrition by overtime, department, satisfaction, income, ...)
2. Preprocessing: scaling numeric features, one-hot encoding categoricals (sklearn Pipeline, no data leakage)
3. Models: Logistic Regression (baseline), Random Forest
4. Imbalance handled with class weights
5. 5-fold stratified cross-validation and GridSearchCV tuning
6. Evaluation with precision, recall, F1 and ROC-AUC (accuracy alone is misleading with imbalanced classes)
7. Interpretation with permutation importance [and SHAP]

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | [ ] | [ ] | [ ] | [ ] |
| Random Forest (tuned) | [ ] | [ ] | [ ] | [ ] |

**Top drivers of attrition:** [e.g. overtime, monthly income, ...]

**Key insight:** [one or two sentences in plain English]

## Run it
```bash
pip install -r requirements.txt
python attrition_project.py        # trains, evaluates, saves plots + model in outputs/
streamlit run app.py               # optional demo app
```

## Tech
Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit
