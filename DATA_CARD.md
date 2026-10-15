# Data card

Dataset: UCI Bike Sharing, hour.csv. It records hourly Capital Bikeshare
rentals in 2011 and 2012, with calendar and weather information.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

Attribution: Fanaee-T, H. (2013). Bike Sharing. UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5W894

UCI lists the dataset under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
The original CSV is downloaded on first use and excluded from Git.
The run records its actual SHA-256, missing-hour count, usable row count, and
the first and last timestamp in each partition.

Target: cnt, total rentals. casual and registered are excluded as target
components. Observed current-hour weather is excluded as unavailable at the
forecast origin. Only past weather and known calendar variables are used.
Missing hourly records remain unknown and are not converted to zero.

data/demo/bike_demo.csv is an original, simulated hourly series for testing
and software demonstration. Its seasonal signal is constructed and its scores
are not measurements on Capital Bikeshare. Original demo fixtures and the
constructed prepared-input JSON use the repository's MIT license.
