# Interview narrative — Model Health Monitor

**30-second version**  
“I treat models as production systems. I trained a PD model on early vintages, logged predictions for later months, joined outcomes, and monitored feature/score PSI. Then I injected drift on purpose to prove the alert rules fire — the Michelangelo pattern Uber describes publicly.”

## Structure

1. Why offline AUC ≠ live health  
2. Uber Michelangelo: prediction log → outcomes → distribution monitors  
3. What I built (walk-forward, PSI, alerts, postmortem)  
4. Limitation: synthetic vintages on UCI  
5. Next: wire to Project 1’s calibrated PD + automatic recalibration trigger  

## Likely follow-ups

| Question | Direction |
|----------|-----------|
| PSI thresholds? | 0.1 warn / 0.25 fail common heuristic; tune to false-alert budget. |
| Labels delayed? | Still monitor score/feature distributions; join outcomes when ready. |
| Retrain always on alert? | No — diagnose pipeline bugs vs true population shift first. |
