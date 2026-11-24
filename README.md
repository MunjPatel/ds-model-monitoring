# Model Health Monitor (Mini-Michelangelo)

**Role focus:** Data Scientist  
**Decision:** When to **trust, freeze, or retrain** a live PD model.

## Problem

Launch-day AUC is not production health. Uber’s Michelangelo platform logs predictions, joins delayed outcomes, and watches feature/prediction distributions for drift. This project implements that loop around a credit PD model.

## What I built

1. Walk-forward serve simulation (synthetic vintages on UCI credit data)
2. Frozen LightGBM PD model trained only on early vintages
3. Prediction log + monthly AUC/Brier after outcome join
4. Feature PSI heatmap + score PSI vs reference month
5. Alert rules (`monitoring/rules.yaml`) + **injected drift** proving monitors fire
6. Incident postmortem + Streamlit health dashboard

## Sources (verified)

- Hermann & Del Balso (2017). *Meet Michelangelo: Uber’s Machine Learning Platform.* Uber Engineering.
- Uber Engineering (2019). *Scaling Machine Learning at Uber with Michelangelo.*
- Baram, E. (2025). *Treating Data as Code at Two Sigma.*

## Reproduce

```bash
cd portfolio/projects
.venv\Scripts\activate
python 06_ds_model_monitoring/scripts/run_mvp.py
streamlit run 06_ds_model_monitoring/app/dashboard.py
```

## Note on vintages

UCI has no `issue_d`. Vintages are **synthetic** (documented) so the monitoring architecture is demonstrable offline. Swap in Lending Club timestamps when available — the monitor code does not change.



1. Push this repo to GitHub (already done if you cloned from MunjPatel/ds-model-monitoring).
2. Go to https://share.streamlit.io/ → **New app**
3. Select repo `MunjPatel/ds-model-monitoring`, branch `main`
4. Main file path: `app/dashboard.py`
5. Python version 3.11+ recommended → Deploy

The app reads artifacts under `outputs/`. Re-run `scripts/run_mvp.py` locally if you need to refresh them before pushing.
