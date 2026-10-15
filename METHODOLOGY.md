# Methodology

The target is the total rental count for hour t. Prediction is made after hour
t minus one has been observed. Calendar values for t are known. Counts and
weather measured at t are unavailable and excluded. This is rolling one-hour
forecasting, not a 24-hour forecast made from a single origin.

The source combines dteday and hr into a datetime index. Duplicate timestamps
are rejected. Missing calendar hours are inserted as missing observations:
they are not relabeled as zero rentals. Counts have exact calendar-hour lags of
1, 24, and 168. Demand rolling windows use shift(1) before aggregation.
The 24-hour mean and standard deviation require 18 observed hours; the
168-hour mean requires 144. The number of observed hours in the 24-hour
window is included as a feature. Required lag and past-weather gaps remove
that forecast example. No target labels are imputed.

Hour, weekday, and month use sine/cosine encodings. A fixed-origin calendar
trend is known for each target timestamp. Only the previous hour's temperature,
humidity, wind speed, and weather category are used. Weather category is used
as its source's numeric severity coding. casual and registered are excluded
because their sum equals the target.

Chronological usable examples allocate 60% to training, 15% to validation,
10% to residual calibration, and 15% to testing. Quick mode uses the final
9,000 calendar hours before feature construction. Full mode uses all data.
No shuffled split or current-hour actual weather is used.

Seasonal naive predictions use lag 24 and lag 168. Ridge regression,
random forests, and histogram gradient boosting use the same features.
Scaling for Ridge is fitted inside its training pipeline. Full mode compares
three Ridge regularization strengths. Selection minimizes validation MAE.
Selected models remain fitted on training only; there is no hidden refit on
the test or calibration period.

Three expanding-window folds inside the training period provide retrospective
diagnostics for the selected model configuration. They do not replace the final
held-out test. Candidate test metrics are reported after selection is frozen.
Predictions are clipped to nonnegative counts for all candidates.

For a nominal 90% residual interval, sort absolute calibration errors and
take rank ceil((n + 1) times 0.9), counting ranks from one. Each interval is the
prediction plus or minus that radius, with a nonnegative lower bound. The code
reports actual test coverage and mean width. Time dependence and distribution
drift mean the exchangeability assumption of ordinary split conformal prediction
is not asserted here; nominal coverage is a target, not a guaranteed result.

During test evaluation, actual earlier test-hour observations are available
for the next one-hour forecast. This matches the stated rolling deployment.
Using these observations would be invalid for a fixed-origin multi-hour claim.
The tests explicitly verify that changing current and future counts or weather
cannot change a feature vector for an earlier forecast.

--history accepts observed raw hourly rows and constructs features for the
hour after the last timestamp. --input accepts already prepared features.
The supplied JSON is a constructed schema example, not a real measurement.
