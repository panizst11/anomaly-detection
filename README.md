# 10-Year CHD Prediction with Logistic Regression

Predicting the 10-year risk of coronary heart disease (`TenYearCHD`) from the Framingham longitudinal study data, using logistic regression. The goal of this version is to **detect more true patients**, i.e. reduce Type II errors (false negatives) and improve the CHD-class F1 score.

## Source and Credits
- **Original code:** [Heart Disease Prediction using Logistic Regression, GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/ml-heart-disease-prediction-using-logistic-regression/). This repository contains only my modified version. The original code is not redistributed here; see the link for the baseline.
- **Dataset:** Framingham Heart Study dataset (`framingham.csv`), not included in this repository. `[add your dataset source link]`. Place the file in `data/` before running the notebook.

## Data
3,751 samples after removing missing values (3,179 healthy vs 572 CHD). The classes are heavily imbalanced. Features used: `age`, `Sex_male`, `cigsPerDay`, `totChol`, `sysBP`, `glucose`. The features are the same in the original and modified versions.

## Results

### Original model
Default `LogisticRegression`, scaler fit before the split, default 0.5 threshold. Test set results:

```
              precision    recall  f1-score   support

           0       0.85      0.99      0.92       951
           1       0.61      0.08      0.14       175

    accuracy                           0.85      1126
   macro avg       0.73      0.54      0.53      1126
weighted avg       0.82      0.85      0.80      1126
```

Accuracy looked high (0.85), but only because the model predicted almost everyone as healthy. It found just 8% of the CHD patients (recall = 0.08), so the CHD-class F1 score was only 0.14.

### Changes applied

**1. Preventing data leakage (order of operations)**
The original code standardized the whole dataset *before* splitting, so the mean and standard deviation included test samples. In the modified code the order is corrected:
1. Split into train, validation, and test sets (stratified, `random_state=42`; train 2100 / val 525 / test 1126).
2. Fit `StandardScaler` on the training set only.
3. Apply (`transform`) the fitted scaler to the validation and test sets.

The test set therefore stays unseen until the final evaluation.

**2. Handling class imbalance**
Healthy cases (class 0) far outnumber CHD cases (class 1), so the default model is dominated by the majority class. We used `LogisticRegression(class_weight='balanced')`, which weights classes inversely to their frequency, so missing a CHD patient is penalized more. `stratify=y` keeps the class proportions equal across all splits.

**3. Tuning the decision threshold on the validation set**
Instead of the default 0.5, thresholds from 0.01 to 0.99 were scanned on the **validation set**, and the one that maximizes the CHD-class F1 score was selected. This threshold was then applied unchanged to the test set. Using the validation set (not the test set) for this choice avoids a second form of leakage.

**4. Extra exploratory analysis**
Added a correlation matrix and feature-target correlations. This does not affect the model.

### Modified model
Test set results with the changes above:

```
              precision    recall  f1-score   support

           0       0.91      0.78      0.84       954
           1       0.31      0.55      0.40       172

    accuracy                           0.74      1126
   macro avg       0.61      0.67      0.62      1126
weighted avg       0.82      0.74      0.77      1126
```

The model now detects 55% of the CHD patients instead of 8%, and the CHD-class F1 score improved from **0.14 to 0.40**. Macro F1 also improved, from 0.53 to 0.62.

### Comparison (test set)

| Metric | Class | Original | Modified |
|---|---|---|---|
| Precision | 0 (healthy) | 0.85 | 0.91 |
| Precision | 1 (CHD) | 0.61 | 0.31 |
| Recall | 0 (healthy) | 0.99 | 0.78 |
| Recall | 1 (CHD) | 0.08 | **0.55** |
| F1-score | 0 (healthy) | 0.92 | 0.84 |
| F1-score | 1 (CHD) | 0.14 | **0.40** |
| Accuracy | - | 0.85 | 0.74 |
| Macro F1 | - | 0.53 | 0.62 |

## Type II Error
Here the null hypothesis is "the person does not have CHD".
- **Type II error (false negative):** a person who develops CHD is predicted healthy. The changes above target this error.
- **Type I error (false positive):** a healthy person is flagged as at risk. This error increased.

The Type II error rate equals 1 − recall of the CHD class: about **92% → 45%** (1 − 0.08 and 1 − 0.55). The higher Type I error explains the drop in precision (0.61 → 0.31) and accuracy (0.85 → 0.74).

In disease screening this trade-off is acceptable: a false alarm leads to an extra check-up, while a missed patient may go untreated. Accuracy is misleading on imbalanced data, so recall and F1 are the key metrics here.
