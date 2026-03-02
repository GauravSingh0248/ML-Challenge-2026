import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler


warnings.filterwarnings("ignore")


def add_row_stats(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add simple row-wise stats over the main feature columns."""
    df = df.copy()
    feats = df[feature_cols]

    df["feat_mean"] = feats.mean(axis=1)
    df["feat_std"] = feats.std(axis=1)
    df["feat_min"] = feats.min(axis=1)
    df["feat_max"] = feats.max(axis=1)
    df["feat_range"] = df["feat_max"] - df["feat_min"]
    df["feat_skew"] = feats.skew(axis=1)

    return df


def median_impute(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]):
    """Median-fill selected columns using train medians."""
    medians = train_df[cols].median()
    train_df[cols] = train_df[cols].fillna(medians)
    test_df[cols] = test_df[cols].fillna(medians)
    return train_df, test_df


def tune_threshold_f1(y_true: np.ndarray, y_proba: np.ndarray, low: float = 0.3, high: float = 0.7, step: float = 0.001):
    """Brute-force search for best F1 threshold on OOF predictions."""
    best_thr = 0.5
    best_f1 = -1.0

    thr = low
    while thr <= high + 1e-9:
        preds = (y_proba >= thr).astype(int)
        score = f1_score(y_true, preds)
        if score > best_f1:
            best_f1 = score
            best_thr = thr
        thr += step

    return best_thr, best_f1


def run_lightgbm_pipeline(
    train_path: str = "TRAIN.csv",
    test_path: str = "TEST.csv",
    output_path: str = "FINAL.csv",
    n_splits: int = 5,
    random_state: int = 42,
):
    base_dir = Path(__file__).resolve().parent
    train_path = base_dir / train_path
    test_path = base_dir / test_path
    output_path = base_dir / output_path

    print("Loading data...")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Expect F01..F47 and Class / ID as described
    feature_cols = [c for c in train.columns if c.startswith("F")]
    feature_cols = sorted(feature_cols)  # keep a stable order

    # Basic sanity check
    if len(feature_cols) != 47:
        print(f"Warning: expected 47 feature columns, found {len(feature_cols)}")

    print("Adding row-wise stats...")
    train = add_row_stats(train, feature_cols)
    test = add_row_stats(test, feature_cols)

    # Columns to feed the model (original + engineered). Exclude target and ID.
    extra_cols = ["feat_mean", "feat_std", "feat_min", "feat_max", "feat_range", "feat_skew"]
    all_features = feature_cols + extra_cols

    print("Handling missing values with median imputation...")
    train, test = median_impute(train, test, all_features)

    X = train[all_features]
    y = train["Class"].astype(int)
    X_test_full = test[all_features]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_preds = np.zeros(len(train), dtype=float)
    test_preds = np.zeros(len(test), dtype=float)
    fold_scores = []

    print("Starting LightGBM CV training...")

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold}/{n_splits}")

        X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        # Robust scaling within the fold (to avoid leakage)
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test_full)

        # Handle imbalance with scale_pos_weight; simple heuristic
        pos = y_tr.sum()
        neg = len(y_tr) - pos
        if pos == 0:
            scale_pos_weight = 1.0
        else:
            scale_pos_weight = neg / pos

        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "scale_pos_weight": scale_pos_weight,
            "verbosity": -1,
            "seed": random_state + fold,
        }

        lgb_train = lgb.Dataset(X_tr_s, label=y_tr)
        lgb_valid = lgb.Dataset(X_val_s, label=y_val)

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=5000,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=200),
            ],
        )

        val_pred = model.predict(X_val_s, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_pred

        # temporary 0.5 threshold for per-fold reporting
        fold_f1 = f1_score(y_val, (val_pred >= 0.5).astype(int))
        fold_scores.append(fold_f1)
        print(f"Fold {fold} F1 (thr=0.5): {fold_f1:.4f}")

        test_preds += model.predict(X_test_s, num_iteration=model.best_iteration) / n_splits

    print("\nCV done.")
    print("Fold-wise F1 scores (thr=0.5):", [f"{s:.4f}" for s in fold_scores])

    # Tune threshold on the full OOF predictions
    print("\nTuning decision threshold for best F1 on OOF...")
    best_thr, best_f1 = tune_threshold_f1(y.values, oof_preds, low=0.3, high=0.7, step=0.001)
    print(f"Best threshold: {best_thr:.3f}")
    print(f"Best OOF F1 at best threshold: {best_f1:.4f}")

    # Also report OOF F1 at default 0.5 for reference
    default_f1 = f1_score(y, (oof_preds >= 0.5).astype(int))
    print(f"OOF F1 at threshold 0.5: {default_f1:.4f}")

    # Apply tuned threshold on averaged test probabilities
    final_test_labels = (test_preds >= best_thr).astype(int)

    print("\nSaving FINAL.csv...")
    submission = pd.DataFrame(
        {
            "ID": test["ID"],
            "CLASS": final_test_labels,
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    run_lightgbm_pipeline()

