# 🔧 Device Fault Detection — IEEE SB GEHU ML Challenge

A supervised binary classification solution built to detect whether an embedded device is operating normally or experiencing a fault, based on 47 numerical sensor readings.

---

## 📌 Problem Overview

Every device tells a story through its sensor data. This project reads that story and answers one question — is the device healthy, or is something going wrong?

- **Input:** 47 numerical features (F01–F47) captured during device activity cycles  
- **Target:** `Class` → `0` (Normal) or `1` (Faulty)  
- **Task:** Binary fault detection on tabular sensor data

---

## 📂 Project Structure

```
ML-Fault-Detection/
│
├── TRAIN.csv           # Labeled training data
├── TEST.csv            # Unlabeled test data (with ID column)
├── FINAL.csv           # Model predictions — submitted for evaluation
├── fault_detection_lgbm.py         # Full pipeline script
└── README.md
```

---

## ⚙️ Approach

### 1. Feature Engineering
Beyond the raw F01–F47 readings, a few row-level statistical signals were added to help the model understand the overall "shape" of each device's state:

| Feature | Description |
|---|---|
| `feat_mean` | Average across all 47 sensors |
| `feat_std` | Spread / variability |
| `feat_min` / `feat_max` | Extremes |
| `feat_range` | Max minus min |
| `feat_skew` | Asymmetry in readings |

### 2. Missing Value Handling
Any nulls are filled using **training set medians** — computed before the fold split to avoid data leakage.

### 3. Model — LightGBM
LightGBM was chosen for its speed, accuracy, and strong performance on tabular data with imbalanced classes.

Key settings:
- `scale_pos_weight` computed per fold from class ratio
- `feature_fraction = 0.9`, `bagging_fraction = 0.8` for regularization
- Early stopping with 100 rounds patience

### 4. Cross-Validation Strategy
- **StratifiedKFold** with 5 splits (preserves class ratio per fold)
- **RobustScaler** fit inside each fold — no leakage
- Test predictions averaged across all 5 folds (soft voting)

### 5. Threshold Tuning
Rather than defaulting to 0.5, the decision threshold is tuned on out-of-fold (OOF) predictions:
- Search range: 0.30 → 0.70 (step 0.001)
- Metric: F1 Score
- Best threshold found: **0.381**

---

## 📊 Results

| Metric | Value |
|---|---|
| Fold 1 F1 | 0.9868 |
| Fold 2 F1 | 0.9852 |
| Fold 3 F1 | 0.9868 |
| Fold 4 F1 | 0.9854 |
| Fold 5 F1 | 0.9872 |
| **OOF F1 (threshold=0.5)** | **0.9863** |
| **OOF F1 (best threshold=0.381)** | **0.9866** |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy lightgbm scikit-learn
```

### Steps
1. Place `TRAIN.csv` and `TEST.csv` in the same directory as `fault_detection_lgbm.py`
2. Run the script:
```bash
python fault_detection_lgbm.py
```
3. `FINAL.csv` will be generated with two columns: `ID` and `CLASS`

### Output Format
```
ID    CLASS
1     1
2     0
3     0
4     1
```

---

## 🧠 Possible Improvements

- **XGBoost / CatBoost blend** — averaging probabilities from multiple tree models often adds 0.002–0.005 F1
- **Optuna hyperparameter tuning** — Bayesian search over LightGBM params
- **Stacking** — use LGBM + XGBoost + RandomForest as base learners with Logistic Regression as meta-model
- **Interaction features** — pairwise products or ratios between high-importance sensor pairs

---

## 📋 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `lightgbm` | Gradient boosting model |
| `scikit-learn` | Cross-validation, scaling, metrics |

---

## 📎 Notes

- Dataset provided by **IEEE SB, GEHU** for educational purposes — no ownership claimed
- Final output row count and order match `TEST.csv` exactly
- All preprocessing is done leak-free within cross-validation folds
