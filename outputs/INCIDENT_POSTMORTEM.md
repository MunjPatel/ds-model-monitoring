# Incident postmortem (simulated)

**Trigger:** Synthetic drift on `utilization` and `n_late` in serve month 17.

- Launch AUC: 0.7983504764525784
- Latest clean-month AUC: 0.7894809362903671
- Injected score PSI: 0.032
- Alerts fired: 1 (1 critical)

**Action taken (demo policy):** Page model owner; freeze auto-approvals if score PSI FAIL; retrain or recalibrate.

This drift was **intentionally injected** to validate monitors — not observed in raw UCI data.
