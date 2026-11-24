"""Load UCI credit data and assign synthetic issue vintages for walk-forward monitoring.

Real Lending Club issue_d would be ideal; UCI has no timestamps, so we assign
vintages with a documented random but stratified scheme so the monitoring loop
is demonstrable offline without Kaggle.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_p5_module(name: str):
    path = Path(__file__).resolve().parents[2] / "05_ds_calibrated_credit" / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"p5_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_p5_data = _load_p5_module("data")
_p5_features = _load_p5_module("features")


def load_with_vintages(n_months: int = 18, seed: int = 7) -> pd.DataFrame:
    df = _p5_features.engineer(_p5_data.load_credit_default())
    rng = np.random.default_rng(seed)
    weights = np.linspace(0.8, 1.2, n_months)
    weights = weights / weights.sum()
    months = rng.choice(np.arange(n_months), size=len(df), p=weights)
    df = df.copy()
    df["vintage"] = months
    df["serve_month"] = months
    return df


def train_serve_split(df: pd.DataFrame, train_end: int = 11):
    """Train on vintages <= train_end; serve on later months."""
    train = df[df["vintage"] <= train_end].reset_index(drop=True)
    serve = df[df["vintage"] > train_end].reset_index(drop=True)
    return train, serve
