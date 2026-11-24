"""Population stability + prediction health monitors (Uber Michelangelo-inspired)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two univariate distributions."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 20 or len(actual) < 20:
        return 0.0
    breaks = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(breaks) < 3:
        return 0.0
    e_counts = np.histogram(expected, bins=breaks)[0].astype(float) + 1e-6
    a_counts = np.histogram(actual, bins=breaks)[0].astype(float) + 1e-6
    e = e_counts / e_counts.sum()
    a = a_counts / a_counts.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def psi_table(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], bins: int = 10) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in ref.columns or f not in cur.columns:
            continue
        val = psi(ref[f].to_numpy(), cur[f].to_numpy(), bins=bins)
        rows.append(
            {
                "feature": f,
                "psi": val,
                "status": "FAIL" if val >= 0.25 else ("WARN" if val >= 0.1 else "PASS"),
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def score_psi(ref_scores: np.ndarray, cur_scores: np.ndarray) -> float:
    return psi(ref_scores, cur_scores)


def inject_feature_drift(df: pd.DataFrame, feature: str, scale: float = 1.5) -> pd.DataFrame:
    """Synthetic serve-time drift for monitor testing (documented as injected)."""
    out = df.copy()
    if feature in out.columns:
        out[feature] = out[feature] * scale
    return out


def evaluate_alerts(psi_df: pd.DataFrame, score_psi_val: float, rules: dict) -> pd.DataFrame:
    alerts = []
    feat_warn = float(rules.get("feature_psi_warn", 0.1))
    feat_fail = float(rules.get("feature_psi_fail", 0.25))
    score_fail = float(rules.get("score_psi_fail", 0.2))
    for _, row in psi_df.iterrows():
        if row["psi"] >= feat_fail:
            alerts.append(
                {
                    "severity": "critical",
                    "kind": "feature_psi",
                    "feature": row["feature"],
                    "value": float(row["psi"]),
                    "message": f"Feature PSI FAIL for {row['feature']}",
                }
            )
        elif row["psi"] >= feat_warn:
            alerts.append(
                {
                    "severity": "warn",
                    "kind": "feature_psi",
                    "feature": row["feature"],
                    "value": float(row["psi"]),
                    "message": f"Feature PSI WARN for {row['feature']}",
                }
            )
    if score_psi_val >= score_fail:
        alerts.append(
            {
                "severity": "critical",
                "kind": "score_psi",
                "feature": "calibrated_pd",
                "value": float(score_psi_val),
                "message": "Prediction-score distribution drift FAIL",
            }
        )
    return pd.DataFrame(alerts)
