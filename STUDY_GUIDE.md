# Study guide

Read project/data.py and manually compute one feature row. Then trace model
selection and residual calibration in project/workflow.py.

Be prepared to explain:

1. The forecast origin, target hour, and information available at the origin.
2. Why casual, registered, and actual target-hour weather would leak information.
3. Why shifting a table row is insufficient if missing hours have not been reindexed.
4. Why shift(1) must precede rolling aggregation.
5. What the minimum-observation requirements do to the usable sample.
6. Why a shuffled split produces an unsuitable future-demand estimate.
7. How the two seasonal naive baselines work without fitted parameters.
8. Why MAE and RMSE can rank models differently.
9. How expanding windows differ from ordinary cross-validation.
10. How the residual quantile rank is computed for nominal 90% coverage.
11. Why nominal and observed interval coverage can differ under drift.
12. Why observed earlier test-hour demand is valid for one-hour rolling forecasts.

For the practical exam, introduce one missing hourly record and inspect lag 24
at the relevant future timestamp. Change a future target count and prove that
an earlier feature vector is unchanged. Then extend the prediction horizon
with a documented origin-aware design; do not merely rename a one-hour result.
