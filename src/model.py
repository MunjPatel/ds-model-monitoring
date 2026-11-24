"""Train a frozen PD model for the monitoring simulation."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


FEATURE_COLS = [
    "limit_bal",
    "age",
    "education",
    "marriage",
    "pay_1",
    "pay_2",
    "pay_3",
    "bill_amt1",
    "pay_amt1",
    "utilization",
    "n_late",
    "max_delinquency",
    "pay_to_bill",
    "avg_bill",
    "avg_pay",
]


def available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def train_pd_model(train: pd.DataFrame):
    feats = available_features(train)
    # Hold out last training vintage as internal valid
    vmax = int(train["vintage"].max())
    tr = train[train["vintage"] < vmax]
    va = train[train["vintage"] == vmax]
    if len(va) < 200 or len(tr) < 500:
        tr, va = train.iloc[: int(0.85 * len(train))], train.iloc[int(0.85 * len(train)) :]
    model = lgb.LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=24,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        tr[feats],
        tr["default"],
        eval_set=[(va[feats], va["default"])],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model, feats


def score(model, feats: list[str], df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[feats])[:, 1]


def month_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    if len(y) < 30 or y.sum() == 0 or y.sum() == len(y):
        return {"auc": np.nan, "brier": float(brier_score_loss(y, p)) if len(y) else np.nan, "n": int(len(y))}
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "n": int(len(y)),
        "default_rate": float(y.mean()),
        "mean_pd": float(p.mean()),
    }
