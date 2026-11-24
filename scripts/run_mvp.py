from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_with_vintages, train_serve_split  # noqa: E402
from model import month_metrics, score, train_pd_model  # noqa: E402
from monitoring import evaluate_alerts, inject_feature_drift, psi_table, score_psi  # noqa: E402


def main():
    out = ROOT / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    rules = yaml.safe_load((ROOT / "monitoring" / "rules.yaml").read_text(encoding="utf-8"))

    df = load_with_vintages(n_months=18)
    train, serve = train_serve_split(df, train_end=11)
    model, feats = train_pd_model(train)

    # Launch reference = first serve month (clean)
    serve_months = sorted(serve["serve_month"].unique())
    ref_month = serve_months[0]
    ref = serve[serve["serve_month"] == ref_month].copy()
    ref_scores = score(model, feats, ref)

    # Prediction log + monthly metrics (outcomes assumed known after seasoning)
    logs = []
    perf_rows = []
    for m in serve_months:
        chunk = serve[serve["serve_month"] == m].copy()
        p = score(model, feats, chunk)
        chunk["predicted_pd"] = p
        chunk["prediction_id"] = [f"{m}_{i}" for i in range(len(chunk))]
        logs.append(chunk[["prediction_id", "serve_month", "predicted_pd", "default"] + feats])
        met = month_metrics(chunk["default"].to_numpy(), p)
        met["serve_month"] = int(m)
        met["score_psi_vs_ref"] = score_psi(ref_scores, p)
        perf_rows.append(met)

    pred_log = pd.concat(logs, ignore_index=True)
    pred_log.to_csv(out / "prediction_log.csv", index=False)
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out / "monthly_performance.csv", index=False)

    # Feature PSI by month vs reference
    psi_frames = []
    for m in serve_months:
        cur = serve[serve["serve_month"] == m]
        tab = psi_table(ref[feats], cur[feats], feats)
        tab["serve_month"] = int(m)
        psi_frames.append(tab)
    psi_by_month = pd.concat(psi_frames, ignore_index=True)
    psi_by_month.to_csv(out / "psi_by_month.csv", index=False)

    # Inject drift into last month and prove monitors fire
    last = serve_months[-1]
    drifted = inject_feature_drift(serve[serve["serve_month"] == last].copy(), "utilization", scale=1.8)
    drifted = inject_feature_drift(drifted, "n_late", scale=1.6)
    drift_scores = score(model, feats, drifted)
    drift_psi = psi_table(ref[feats], drifted[feats], feats)
    drift_score_psi = score_psi(ref_scores, drift_scores)
    alerts = evaluate_alerts(drift_psi, drift_score_psi, rules)
    drift_psi.to_csv(out / "injected_drift_psi.csv", index=False)
    alerts.to_csv(out / "alerts.csv", index=False)

    # Incident postmortem stub metrics
    launch = perf.iloc[0]
    latest = perf.iloc[-1]
    postmortem = {
        "title": "Injected utilization / n_late drift on final serve month",
        "michelangelo_pattern": "Log predictions, join outcomes, monitor feature & prediction distributions (Uber Engineering).",
        "launch_auc": launch.get("auc"),
        "latest_auc": latest.get("auc"),
        "injected_score_psi": float(drift_score_psi),
        "n_alerts": int(len(alerts)),
        "critical_alerts": int((alerts["severity"] == "critical").sum()) if len(alerts) else 0,
        "action": "Page model owner; freeze auto-approvals if score PSI FAIL; retrain or recalibrate.",
    }
    (out / "INCIDENT_POSTMORTEM.md").write_text(
        "# Incident postmortem (simulated)\n\n"
        f"**Trigger:** Synthetic drift on `utilization` and `n_late` in serve month {last}.\n\n"
        f"- Launch AUC: {postmortem['launch_auc']}\n"
        f"- Latest clean-month AUC: {postmortem['latest_auc']}\n"
        f"- Injected score PSI: {postmortem['injected_score_psi']:.3f}\n"
        f"- Alerts fired: {postmortem['n_alerts']} ({postmortem['critical_alerts']} critical)\n\n"
        f"**Action taken (demo policy):** {postmortem['action']}\n\n"
        "This drift was **intentionally injected** to validate monitors — not observed in raw UCI data.\n",
        encoding="utf-8",
    )

    meta = {
        "n_train": int(len(train)),
        "n_serve": int(len(serve)),
        "train_vintages": f"0-{int(train['vintage'].max())}",
        "serve_vintages": f"{int(serve['vintage'].min())}-{int(serve['vintage'].max())}",
        "n_features": len(feats),
        "features": feats,
        "ref_month": int(ref_month),
        "injected_drift_month": int(last),
        "n_alerts_on_injected_drift": int(len(alerts)),
        "note": "Vintages are synthetic on UCI data for offline walk-forward monitoring demo.",
        "sources": [
            "Uber Engineering (2017): Meet Michelangelo",
            "Uber Engineering (2019): Scaling Michelangelo",
            "Two Sigma (2025): Treating Data as Code",
        ],
    }
    (out / "meta.json").write_text(json.dumps({**meta, **postmortem}, indent=2), encoding="utf-8")
    print("P6 monitoring MVP complete ->", out)
    print(pd.Series({k: meta[k] for k in ["n_train", "n_serve", "n_alerts_on_injected_drift"]}))


if __name__ == "__main__":
    main()
