# Titanic Survival Prediction (Kaggle)

This project builds a machine learning model to predict whether a passenger survived the Titanic disaster.
The solution follows proper machine learning practices using scikit-learn Pipelines, feature engineering,
and GridSearchCV for model selection.

---

## Problem Statement

Predict passenger survival (`Survived`) based on demographic and travel information.

- Type: Binary Classification
- Target: Survived (0 = No, 1 = Yes)
- Dataset: Kaggle Titanic – Machine Learning from Disaster

---

## Dataset

- train.csv – used for training and validation
- test.csv – used for generating Kaggle submissions

The test dataset does not contain the target variable and is only used for final predictions.

---

## Approach

### 1. Data Preprocessing

All preprocessing is handled using a single pipeline to avoid data leakage.

- Missing numerical values → Median imputation
- Missing categorical values → Most frequent imputation
- Numerical features → StandardScaler
- Categorical features → OneHotEncoder

Implemented using ColumnTransformer.

---

### 2. Feature Engineering

The following features were created to improve model performance:

- FamilySize = SibSp + Parch + 1
- IsAlone (1 if FamilySize == 1 else 0)
- Title extracted from passenger names

Feature engineering is applied consistently to both train and test data through the pipeline.

---

### 3. Model Training

A scikit-learn Pipeline was used:

Preprocessing → Classifier

Logistic Regression was selected due to stable performance and interpretability.

---

### 4. Hyperparameter Tuning

Hyperparameters were optimized using GridSearchCV with 5-fold cross-validation.

Tuned parameters include:

- Regularization strength (C)
- Penalty type (L1, L2)
- Solver (liblinear)

The best estimator from GridSearchCV was used for final predictions.

---

### 5. Kaggle Submission

Predictions were generated on test.csv and saved in Kaggle-compatible format.

- Columns: PassengerId, Survived
- File: submission.csv

This file can be directly uploaded to Kaggle.

---
