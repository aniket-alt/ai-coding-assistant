# Abstract — Energy Demand Forecasting

**How much electricity will this household use in each hour of tomorrow, and how wide should the uncertainty be?**

We study this question on *Individual Household Electric Power Consumption (UCI)*: 2,075,259 rows as downloaded, used under Creative Commons Attribution 4.0 (CC BY 4.0). Following the CRISP-DM process, we compare 6 approaches, 3 of them baselines, on 25,458 evaluation rows and rank them by pinball loss (kw). Evaluation split: walk-forward, 18 refits every 60 days, day-ahead.

The best approach, **LightGBM quantile**, reaches 0.1428 against 0.1750 for the strongest baseline (Four-week hourly average).

4 of 4 leakage assertions passed while training ran, and static analysis of the source reported 0 critical and 0 warning findings.

**Keywords:** probabilistic day-ahead load forecasting, day-ahead quantile forecasting, minimum lag equal to the horizon, walk-forward with paired bootstrap, measured cost of a leaky lag

---

*Generated from this project's artifacts by `scripts/generate_docs.py`. Figures are not typed by hand.*
