# Fraud Detection ML System — Project Report
> Generated: 2026-03-29 18:45:43

---

## 1. Executive Summary
End-to-end machine learning pipeline to detect fraudulent financial transactions from a highly imbalanced dataset. Four classifiers were trained, evaluated, and compared using industry-standard metrics.

## 2. Dataset Overview
| Attribute | Value |
|-|-|
| Total transactions | 50,000 |
| Fraudulent | 1,553 (3.11%) |
| Legitimate | 48,447 |
| Features | 19 (10 anonymised V-features + 9 engineered) |

## 3. Methodology
### 3.1 Preprocessing
- Missing value check (none detected)
- LabelEncoder on `MerchantCategory`
- StandardScaler on continuous features
- Stratified 80 / 20 train-test split

### 3.2 Class-Imbalance Handling
- SMOTE-style interpolative oversampling of minority class to ~15 % ratio
- `class_weight='balanced'` passed to every scikit-learn estimator

### 3.3 Engineered Features
| Feature | Description |
|-|-|
| `IsNightTxn` | 1 if transaction hour is 10 PM – 4 AM |
| `AmountZScore` | Standardised transaction amount |
| `HighAmountFlag` | 1 if Amount > 95th percentile |
| `TransactionCount_24h` | Rolling 24-hour transaction count |
| `FrequencyFlag` | 1 if TransactionCount_24h > 8 |
| `RiskScore` | Composite weighted indicator (0–1) |

## 4. Models
1. Logistic Regression — baseline
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## 5. Results
| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-|-|-|-|-|-|
| Logistic Regression | 1.0000 | 1.0000 | 0.9984 | 1.0000 | 0.9968 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 6. Best Model: Logistic Regression
```
ROC-AUC  : 1.0000
PR-AUC   : 1.0000
F1-Score : 0.9984
Precision: 1.0000
Recall   : 0.9968
```

## 7. Threshold Guidance
Default threshold = **0.5**. Lower it (e.g. 0.3) to catch more fraud at the cost of more false positives. Tune based on business cost of false negatives vs false positives.

## 8. Recommendations
- Monthly model retraining to combat concept drift
- Add velocity features (amount last 1 h / 7 d)
- Integrate SHAP for auditor-facing explanations
- A/B test threshold values in staging before production

---
_End of Report_