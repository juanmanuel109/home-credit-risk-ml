# Sprint Project 02 - Home Credit Default Risk 🏦
> Machine Learning project for credit risk prediction using ensemble models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Code Style](#code-style)
- [Testing](#testing)

## 🎯 Business Problem

This is a **binary classification task** for credit risk assessment:
- **Objective**: Predict whether a loan applicant will be able to repay their debt
- **Target Variable**: 
  - `1` = Client will have payment difficulties (late payment > X days on first Y installments)
  - `0` = Client will repay on time
- **Evaluation Metric**: [Area Under the ROC Curve (AUC-ROC)](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

## 📊 Dataset

The dataset is from the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/competitions/home-credit-default-risk/):
- **Training set**: `application_train_aai.csv` (246,008 samples, 122 features)
- **Test set**: `application_test_aai.csv` (61,503 samples, 121 features)
- **Metadata**: `HomeCredit_columns_description.csv`

**Data is automatically downloaded** when running the notebook (Section 1).

### Key Features:
- Demographic information (age, gender, family status, education)
- Financial data (income, credit amount, annuity, goods price)
- Employment information
- External credit scores
- Document flags
- Credit bureau inquiries

## 📁 Project Structure

```
sprint2/
├── AnyoneAI - Sprint Project 02.ipynb  # Main notebook with all experiments
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── dataset/                            # Data files (auto-generated)
│   ├── application_train_aai.csv
│   ├── application_test_aai.csv
│   └── HomeCredit_columns_description.csv
├── src/                                # Source code modules
│   ├── __init__.py
│   ├── config.py                       # Configuration settings
│   ├── data_utils.py                   # Data loading and splitting
│   └── preprocessing.py                # Feature engineering and preprocessing
└── tests/                              # Unit tests
    ├── __init__.py
    ├── conftest.py
    ├── test_data_utils.py
    └── test_preprocessing.py
```

## 🤖 Models Implemented

### 1. **Logistic Regression** (Baseline)
- Simple linear model
- Validation AUC: **0.6772**
- Good generalization but limited capacity

### 2. **Random Forest** (Default)
- Ensemble of 100 decision trees
- Validation AUC: **0.7052**
- Severe overfitting (Train AUC: 1.0)

### 3. **Random Forest** (Tuned with RandomizedSearchCV)
- Hyperparameter optimization with 5-fold CV
- Best parameters: `max_depth=15`, `min_samples_split=50`, `min_samples_leaf=10`
- Validation AUC: **0.7365** 
- Reduced overfitting (Train AUC: 0.897)

### 4. **LightGBM** (Gradient Boosting)
- Advanced gradient boosting with regularization
- Validation AUC: **0.7548**
- Fast training, good balance between bias and variance

### 5. **Sklearn Pipeline + LightGBM**
- End-to-end pipeline for reproducibility
- Automated preprocessing and model training
- Same performance as standalone LightGBM

### 6. **XGBoost** (Custom with Feature Engineering) 🏆
- 300 boosting rounds with early stopping
- Advanced feature engineering (14 new features)
- Scale positive weight for imbalanced data
- L1/L2 regularization
- **Expected Validation AUC: 0.7548**

## 📈 Results

| Model | Train AUC | Validation AUC | Gap | Status |
|-------|-----------|----------------|-----|--------|
| Logistic Regression | 0.6798 | 0.6772 | 0.0026 | ✅ Underfitting |
| Random Forest (default) | 1.0000 | 0.7052 | 0.2948 | ❌ Overfitting |
| Random Forest (tuned) | 0.8970 | 0.7365 | 0.1605 | ⚠️ Moderate overfitting |
| LightGBM | 0.8058 | 0.7548 | 0.05 | ✅ Best balance |
| XGBoost (custom) | 0.8290 | 0.7547 | 0.07 | 🚀  |

### Feature Engineering Highlights:
- Credit to income ratio
- Annuity to income ratio
- Employment to age ratio
- Income per family member
- External source aggregations (mean, max, min)
- Document submission rate
- Credit bureau inquiry total

## 🛠️ Technologies

- **Python 3.8+**: Main programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: ML models, preprocessing, metrics
- **LightGBM**: Gradient boosting framework
- **XGBoost**: Extreme gradient boosting
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive experimentation

## Installation

A `requirements.txt` file is provided with all the needed Python libraries for running this project. For installing the dependencies just run:

```console
$ pip install -r requirements.txt
```

*Note:* We encourage you to install those inside a virtual environment.

## Code Style

Following a style guide keeps the code's aesthetics clean and improves readability, making contributions and code reviews easier. Automated Python code formatters make sure your codebase stays in a consistent style without any manual work on your end. If adhering to a specific style of coding is important to you, employing an automated to do that job is the obvious thing to do. This avoids bike-shedding on nitpicks during code reviews, saving you an enormous amount of time overall.

We use [Black](https://black.readthedocs.io/) and [isort](https://pycqa.github.io/isort/) for automated code formatting in this project, you can run it with:

```console
$ isort --profile=black . && black --line-length 88 .
```

Wanna read more about Python code style and good practices? Please see:
- [The Hitchhiker’s Guide to Python: Code Style](https://docs.python-guide.org/writing/style/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Tests

We provide unit tests along with the project that you can run and check from your side the code meets the minimum requirements of correctness needed to approve. To run just execute:

```console
$ pytest tests/
```

If you want to learn more about testing Python code, please read:
- [Effective Python Testing With Pytest](https://realpython.com/pytest-python-testing/)
- [The Hitchhiker’s Guide to Python: Testing Your Code](https://docs.python-guide.org/writing/tests/)
