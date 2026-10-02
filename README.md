***this is an unsupervised anomaly detection task.

Approach
Load data with pandas.
Outlier removal (Hampel filter) on the training data: for each feature, compute the median and the MAD (median absolute deviation, scaled by 1.6). A value is an outlier if it is farther than 5 scaled MADs from the median. Any training row that is an outlier in at least one feature is dropped, so the model learns what "normal" looks like from cleaner data.
Standardization (z-score) using the mean and standard deviation of the cleaned training set; the same statistics are applied to the test set.
Model: IsolationForest(contamination=0.5, random_state=1) fitted on the standardized training data. Its output (-1 anomaly, 1 normal) is mapped to the required format (1 anomaly, 0 normal).
Prediction on the standardized test set, saved as submission with the column is_anomaly, then exported to submission.csv and zipped into result.zip.
Run it


Put the contest files in a Data/ folder (Data/train.csv, Data/test.csv) and update the paths in the first code cell, then run all cells. The dataset belongs to the contest and is not included in this repository.

Files
notebook.ipynb — full solution
submission.csv — predictions for test.csv
Results

Contest score (F1): 51
