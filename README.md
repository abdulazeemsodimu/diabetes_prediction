# Diabetes Prediction

A machine learning classification project that predicts whether a patient is likely to have diabetes based on demographic and diagnostic measurements.

The project focuses not only on achieving good overall predictive performance, but also on **reducing false negatives**. In a diabetes screening context, failing to identify a patient who may have diabetes can be more costly than incorrectly flagging a non-diabetic patient. Therefore, recall was given particular attention during model evaluation and threshold selection.

---

## Project Overview

This project follows an end-to-end machine learning workflow:

1. Exploratory Data Analysis (EDA)
2. Data cleaning and preprocessing
3. Baseline model comparison
4. Cross-validation
5. Hyperparameter tuning
6. Probability threshold tuning
7. Final model evaluation
8. Error analysis
9. Model serialization for deployment

The project uses several classification algorithms and evaluates them using multiple metrics rather than relying solely on accuracy.

---

## Dataset

The dataset contains medical and demographic measurements used to predict the `Outcome` variable.

### Features

| Feature                    | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `Pregnancies`              | Number of times the patient has been pregnant       |
| `Glucose`                  | Plasma glucose concentration                        |
| `BloodPressure`            | Diastolic blood pressure (mm Hg)                    |
| `SkinThickness`            | Triceps skin fold thickness (mm)                    |
| `Insulin`                  | 2-hour serum insulin (μU/ml)                        |
| `BMI`                      | Body Mass Index                                     |
| `DiabetesPedigreeFunction` | Score representing diabetes history in relatives    |
| `Age`                      | Patient's age in years                              |
| `Outcome`                  | Target variable: `0` = non-diabetic, `1` = diabetic |

---

## Exploratory Data Analysis

The EDA was performed to understand the dataset, identify potential data-quality issues, and determine patterns that could influence modelling decisions.

### Key Findings

* Approximately **65%** of observations belong to the non-diabetic class (`Outcome = 0`), while approximately **35%** belong to the diabetic class (`Outcome = 1`).
* Several features contain zero values that may represent missing or invalid measurements.
* `Insulin` contains a particularly large proportion of zero values (**48.7%**).
* `SkinThickness` also contains a substantial proportion of zero values (**29.6%**).
* Diabetic observations generally have higher glucose values than non-diabetic observations.
* Diabetic observations tend to have higher BMI values, although there is considerable overlap between the two classes.
* `Age` is positively skewed, with considerably more younger observations.
* `Insulin` has substantial dispersion and extreme values and therefore required particular attention during preprocessing.
* Zero insulin values occur in both classes, affecting approximately **51% of diabetic observations and 47% of non-diabetic observations**.

These findings informed the preprocessing strategy used during modelling.

---

## Data Preprocessing

Zero values in the following features were treated as potential missing values:

```text
Glucose
BloodPressure
SkinThickness
Insulin
BMI
```

Rather than removing these observations, the project uses **median imputation**.

### Preprocessing Strategy

For models requiring scaled features:

```text
Potential missing values
        ↓
Median Imputation
        ↓
StandardScaler
        ↓
Model
```

For tree-based models:

```text
Potential missing values
        ↓
Median Imputation
        ↓
Model
```

The preprocessing steps were implemented using `Pipeline` and `ColumnTransformer` so that preprocessing remains part of the model workflow and can be consistently applied during cross-validation and prediction.

---

## Train/Test Split

The dataset was divided into:

* **80% training data**
* **20% test data**

Stratification was used when splitting the data because the target classes are imbalanced.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

The test set was kept separate and used only for final model evaluation.

---

## Baseline Models

Several classification algorithms were evaluated to establish baseline performance:

1. **DummyClassifier**

   * Provides a simple sanity-check baseline.

2. **Logistic Regression**

   * Used as a linear classification baseline.

3. **K-Nearest Neighbors (KNN)**

   * Evaluated as a distance-based classifier.

4. **Support Vector Classifier (SVC)**

   * Used to explore potentially non-linear decision boundaries.

5. **Decision Tree**

   * Provides a tree-based approach to modelling non-linear relationships.

6. **Random Forest**

   * An ensemble tree-based model that can capture more complex relationships while reducing some of the overfitting associated with individual decision trees.

---

## Model Evaluation

Because the classes are imbalanced and the project's primary concern is identifying potential diabetic cases, several evaluation metrics were considered:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Balanced Accuracy
* Average Precision

Five-fold **Stratified Cross-Validation** was used during model comparison.

### Baseline Results

The initial comparison identified the following models as particularly promising:

* Logistic Regression
* SVC
* Random Forest

These models achieved the strongest Average Precision performance among the evaluated approaches.

---

## Hyperparameter Tuning

The three promising models were further optimized using:

* `RandomizedSearchCV`
* `GridSearchCV`

The primary tuning metric was **Average Precision**, reflecting the project's focus on performance when identifying the positive class.

### Improvement After Tuning

| Model               | Before Tuning | After Tuning |
| ------------------- | ------------: | -----------: |
| Logistic Regression |          0.74 |    **0.751** |
| SVC                 |          0.72 |    **0.753** |
| Random Forest       |          0.70 |     **0.74** |

Hyperparameter tuning improved the Average Precision of all three models.

---

## Threshold Tuning

Instead of relying on the default classification threshold of `0.50`, the project also explored different probability thresholds.

The reason is that a classification model does not simply produce "diabetic" or "non-diabetic" internally. It produces a probability, and the threshold determines when that probability becomes a positive prediction.

Since the project prioritizes identifying as many potential diabetic cases as possible, a target recall of **90%** was investigated.

The threshold selection strategy was:

1. Generate out-of-fold predicted probabilities.
2. Test thresholds from `0.01` to `0.99`.
3. Calculate precision, recall, and F1 score for each threshold.
4. Identify thresholds achieving at least the target recall of 90%.
5. Among those thresholds, select the one with the highest precision.

### Selected Threshold

Logistic Regression achieved:

* **Recall:** 0.90
* **Precision:** 0.52
* **Threshold:** 0.36

The threshold was therefore lowered from the default `0.50` to approximately `0.36`.

This increases the model's sensitivity to potential diabetic cases, at the cost of generating more false positives.

---

## Final Model

### Logistic Regression

Logistic Regression was selected as the final model because it provided competitive performance while achieving the project's primary objective of maintaining high recall.

### Final Preprocessing

* Median imputation for zero-coded missing values
* `StandardScaler`

### Final Hyperparameters

```text
C = 0.1
class_weight = balanced
l1_ratio = 1
max_iter = 200
solver = liblinear
```

### Classification Threshold

```text
0.36
```

---

## Final Test Performance

After fitting the final model on the training data and evaluating it on the held-out test set:

| Metric    |    Score |
| --------- | -------: |
| Accuracy  | **0.71** |
| Precision | **0.55** |
| Recall    | **0.89** |
| F1 Score  | **0.68** |
| ROC-AUC   | **0.81** |
| PR-AUC    | **0.66** |

The most important result is the **0.89 recall**, meaning the final model was able to identify a large majority of the positive cases in the held-out test set.

The lower precision reflects the deliberate decision to prioritize recall: accepting more false positives in exchange for reducing false negatives.

---

## Error Analysis

After final evaluation, an error analysis was performed to understand where the model struggled.

Predictions were categorized into:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

The analysis also examined model errors across individual features and feature groups.

### Key Finding

One notable issue was the model's difficulty with observations where:

```text
Insulin = 0
```

This is important because zero insulin values had already been identified during EDA as potential missing measurements.

This suggests that the way zero-coded insulin values are handled may have an influence on model errors and represents an area for future investigation.

---

## Model Artifact

The final trained model and its classification threshold are saved using `joblib`.

The saved artifact contains:

```python
{
    'model': model,
    'threshold': 0.36,
    'best_params': ...
}
```

This allows the preprocessing pipeline, trained model, and selected classification threshold to be loaded together when making predictions.

---

## Project Structure

A typical project structure is:

```text
diabetes-prediction/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   ├── 1.EDA.ipynb
│   └── modelling v2.ipynb
│
├── models/
│   └── best_model_and_threshold.joblib
│
├── src/
│   ├── app.py
│   ├── train.py
│   ├── predict.py
│   └── preprocessing.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

### Machine Learning Techniques

* Exploratory Data Analysis
* Data preprocessing
* Median imputation
* Feature scaling
* Pipelines
* ColumnTransformer
* Stratified train/test splitting
* Stratified K-Fold cross-validation
* Randomized hyperparameter search
* Grid search
* Probability threshold tuning
* Precision-recall analysis
* ROC analysis
* Error analysis

---

## Notebooks

### `1.EDA.ipynb`

Contains the exploratory analysis of the dataset, including:

* Class distribution
* Feature distributions
* Identification of zero-coded missing values
* Relationships between features and the target
* Investigation of insulin and other potentially problematic measurements

### `modelling v2.ipynb`

Contains the complete modelling workflow:

* Train/test split
* Preprocessing pipelines
* Baseline model comparison
* Cross-validation
* Hyperparameter tuning
* Threshold tuning
* Final model selection
* Test-set evaluation
* ROC and Precision-Recall curves
* Confusion matrix
* Error analysis
* Model serialization

---

## Limitations

This project is a machine learning prediction project and **is not intended to provide medical diagnosis or replace professional medical assessment**.

The model's performance is also limited by the available dataset and the quality of the measurements. In particular, the presence of zero-coded values that may represent missing measurements is an important data-quality consideration.

The relatively small dataset also means that model performance may vary when applied to a different population or dataset.

---

## Future Improvements

Potential improvements include:

* Investigating better methods for handling zero-coded missing values.
* Performing more detailed analysis of insulin-related errors.
* Exploring additional models and ensemble methods.
* Testing calibration of predicted probabilities.
* Performing more extensive feature analysis.
* Evaluating the model on an independent external dataset.
* Improving the deployment interface.
* Monitoring model performance after deployment.

---

## Key Takeaway

This project demonstrates that building a useful classification model involves more than selecting the algorithm with the highest accuracy.

The modelling process considered **data quality, class imbalance, multiple evaluation metrics, cross-validation, hyperparameter optimization, decision thresholds, and error analysis**.

The final Logistic Regression model achieved **89% recall on the held-out test set** using a **0.36 probability threshold**, reflecting the project's priority of minimizing false negatives while maintaining reasonable precision.
