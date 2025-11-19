# Predicting Thirty Day Hospital Readmissions Using Machine Learning

Author: Albert Kabore
Date: November 2025

## Project Overview

This project builds and evaluates a supervised machine learning pipeline to predict thirty day hospital readmissions using a 30,000 row synthetic clinical dataset. The goal is to estimate readmission risk at discharge so that care teams can prioritize proactive outreach, such as early clinic visits, medication reconciliation, and home health referrals.

The work is implemented in Python using pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, matplotlib, and seaborn.

## Data

The main dataset is stored in

- `data/Dataset_Hospital_readmissions_30k.csv`

The original source is the Hospital Readmission Prediction synthetic dataset on Kaggle.

## Methods

The analysis follows a reproducible pipeline:

1. Data loading, validation, and completeness checks
2. Exploratory data analysis (EDA) of numeric and categorical features
3. Advanced feature engineering, including a clinical complexity score, log and square root transforms, and utilization intensity
4. Train test splitting, scaling, and class imbalance handling with SMOTE
5. Training and evaluation of several classifiers
6. Hyperparameter tuning for a Random Forest model
7. Feature importance analysis and interpretation

The main notebook is:

- [`notebooks/Readmission_Project.ipynb`](notebooks/Readmission_Project.ipynb)

## Key Results

- The target class exhibits a 7.17 to 1 imbalance, with a baseline majority class accuracy of 87.75 percent.
- Multiple ensemble models achieved high overall accuracy but low recall for the readmitted class, with ROC AUC values between 0.49 and 0.52 on the held out test set.
- A tuned Random Forest model achieved a cross validated ROC AUC of 0.967 on the SMOTE resampled training data, but only 0.514 on the original test set, indicating overfitting.
- Feature importance analysis showed that length of stay, medication burden, and the engineered clinical complexity score contribute most strongly to predicted risk.

## Report

The full LaTeX report for this project is available on Overleaf:

- [Project report on Overleaf](https://www.overleaf.com/read/vbjqbzwcgjjv#6eb481)

The report includes the abstract, introduction, methods, results, limitations, conclusions, and references.

## Reproducibility

To reproduce the analysis:

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install the dependencies listed in `requirements.txt`.
4. Open `notebooks/Readmission_Project.ipynb` in Jupyter Lab or VS Code.
5. Execute the notebook cells in order.

## References

Key references include

- CMS Hospital Readmissions Reduction Program
- AHRQ resources on hospital readmissions
- Scikit learn, SMOTE, and XGBoost papers

## Decision Framing

### Outcome Variable

- `readmitted_30_days` ∈ {Yes, No}
- Encoded as numeric target: `readmitted` ∈ {1, 0}

### Primary Metric

- **Recall (Sensitivity)** – prioritize not missing high-risk patients.

### Secondary Metrics

- Precision
- F1-score
- Accuracy

### Summary Metrics

- ROC AUC (AUROC)
- Precision–Recall AUC (AUPRC), due to class imbalance

This metric strategy reflects a clinical perspective: it is generally safer to **flag more potential high-risk patients (higher recall)** and then manage resource use using precision and decision thresholds.

---

## Dataset Description

- **File:** `Dataset_Hospital_readmissions_30k.csv`
- **Size:** 30,000 rows × 12 columns
- **Target:** `readmitted_30_days` (Yes/No)
- **Missing values:** 0 in all columns (no imputation required)
- **Duplicates:** 0 duplicate rows detected

### Data Dictionary

| Column                 | Description                                          | Type   | Example  |
|------------------------|------------------------------------------------------|--------|----------|
| `patient_id`           | Unique identifier for each encounter                 | int    | 1        |
| `age`                  | Age in years (adult patients)                        | int    | 74       |
| `gender`               | Gender identity                                      | object | Male     |
| `blood_pressure`       | Blood pressure (systolic/diastolic)                  | object | 130/72   |
| `cholesterol`          | Serum cholesterol (mg/dL)                            | int    | 240      |
| `bmi`                  | Body mass index                                      | float  | 31.5     |
| `diabetes`             | Diabetes diagnosis flag (Yes/No)                     | object | Yes      |
| `hypertension`         | Hypertension diagnosis flag (Yes/No)                 | object | No       |
| `medication_count`     | Number of medications prescribed                     | int    | 5        |
| `length_of_stay`       | Length of index hospital stay (days)                 | int    | 3        |
| `discharge_destination`| Discharge location (Home, Nursing facility, etc.)    | object | Home     |
| `readmitted_30_days`   | Readmission within 30 days (Yes/No)                  | object | No       |

- **Readmission rate:** ~12.25% (approx. 3,675 of 30,000 encounters)
- **Class imbalance:** ~7:1 (No readmission : Yes readmission)

---

## Environment Setup

### 1. Clone the repository

```Powershell
git clone https://github.com/albertokabore/Hospital-Readmission-Prediction
cd Hospital-Readmission-Prediction
```

2. Create and activate virtual environment

```powershell
py -m venv .venv

# Windows
.venv\Scripts\activate

```
3. Install dependencies

#### Install a Package

```
py -m pip install requests
```

```


```powershell
py -m pip install --upgrade pip
py -m pip install jupyterlab pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost pyarrow
```

#### List Installed Packages

```py -m pip list
```

4. Save environment

```Powershell
py -m pip freeze > requirements.txt
```

5. Create or update the .gitignore file in the root project folder and add the following entries:

```Recommended .gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
*.csv
*.log
```

## Tools and Libraries

Python, Jupyter Notebook, VS Code

pandas – data manipulation

numpy – numerical computing

matplotlib, seaborn – visualization and EDA

scikit-learn – modeling, metrics, preprocessing

imbalanced-learn – SMOTE for class imbalance

xgboost – gradient boosting model

Data Cleaning and Curation

The dataset is relatively clean and synthetic, so the cleaning steps focused on validation and light curation rather than heavy correction.

## Main Steps

### Column standardization

Trimmed column names (removed leading/trailing spaces).

Duplicate check

df.duplicated().sum() returned 0.

No rows dropped.

Missing data check

df.isnull().sum() revealed no missing values in any column.

Confirmed: Dataset has no missing values. No imputation required.

Target encoding

Created numeric target for modeling:

df["readmitted"] = df["readmitted_30_days"].map({"Yes": 1, "No": 0})


readmitted becomes the dependent variable for supervised learning.

## Type validation

Numeric features:
age, cholesterol, bmi, medication_count, length_of_stay, readmitted

Categorical features:
gender, blood_pressure, diabetes, hypertension, discharge_destination, readmitted_30_days

The data curation step confirms that the dataset is consistent, complete, and ready for EDA, feature engineering, and modeling without the need for imputation.

## Exploratory Data Analysis (EDA)

### EDA is performed in:
notebooks/Readmission_Project.ipynb

Key tools: histograms, boxplots, correlation heatmaps, categorical count plots.

Numeric Features

Key numeric variables:

age

cholesterol

bmi

medication_count

length_of_stay

readmitted (target)

## Insights:

Age and cholesterol show mild right skew.

BMI clusters around the overweight/obese range in many patients.

Length of stay is typically short (3–8 days), but longer stays are more common in readmitted patients.

Medication count tends to be higher for those later readmitted.

#### Categorical Features

diabetes and hypertension show substantial prevalence, consistent with chronic disease populations.

discharge_destination is dominated by Home, followed by Nursing facility.

gender shows a roughly balanced distribution across the sample.

Target Distribution

Overall readmission rate is low (~12.25%), confirming class imbalance.


#### This motivates:

Use of SMOTE (Synthetic Minority Oversampling Technique) on the training data.

Emphasis on recall and PR AUC in evaluation.

## Feature Engineering


Feature engineering focuses on encoding clinical burden and improving signal for machine learning models.


1. Clinical Complexity Score

A clinical_complexity_score is defined to summarize multiple quantitative indicators of patient burden. It combines:


age


cholesterol


bmi


medication_count


length_of_stay


Encoded comorbidities: diabetes (Yes/No), hypertension (Yes/No)


Core steps (high-level):


Convert comorbidity flags (diabetes, hypertension) to 0/1.


Standardize all numeric burden indicators (z-scores).


Aggregate them (e.g., weighted or equal-weight sum).


Rescale the result to a 0–100 range.


#### Create categorical bands:


Low complexity


Medium complexity


High complexity


Very High complexity


This score is used both:


As a numeric predictor: clinical_complexity_score, and


As a clinical segment variable: complexity_category to compare readmission rates across burden strata.


2. Additional Engineered Features (Conceptual)


#### Parse blood_pressure into:


systolic_bp


diastolic_bp


Encode all categorical variables as numeric (e.g., one-hot encoding for discharge_destination).


Explore interactions (e.g., age × length_of_stay, bmi × diabetes).


These engineered features improve model capacity to capture non-linear and interaction effects important in clinical risk.

## Modeling Pipeline

The modeling pipeline is implemented in the notebook and includes:

Train–Test Split


80% training, 20% test


Stratified by readmitted to preserve class distribution

Scaling


StandardScaler applied to numeric features before modeling.


Class Imbalance Handling


SMOTE applied on the training data to oversample the minority class (readmitted = 1).


The test set remains untouched to represent real-world prevalence.

## Models Evaluated

Decision Tree


Random Forest


Gradient


AdaBoost


XGBoost


Evaluation Metrics


Accuracy


Precision


Recall (Sensitivity)


F1-score


ROC AUC


Precision–Recall AUC


## Hyperparameter Tuning


RandomizedSearchCV on Random Forest with parameters such as:


n_estimators


max_depth


min_samples_split


min_samples_leaf

max_features


The tuned Random Forest is then evaluated and compared to baseline models.


## Key Results (Summary)

Class imbalance was addressed successfully via SMOTE.


Tuned Random Forest and XGBoost perform best on ROC AUC and recall.


Top predictors include:


medication_count


length_of_stay


bmi


(plus contributions from comorbidities and age)


From a clinical standpoint, the model:


Prioritizes recall so that most high-risk patients are identified.


Provides a data-driven method to flag patients needing closer follow-up after discharge.

### Repository Structure

```
Hospital-Readmission-Prediction/
│
├── data/
│   └── Dataset_Hospital_readmissions_30k.csv
│
├── notebooks/
│   └── Readmission_Project.ipynb
│
├── reports/
│   └── Overleaf_Report.pdf   # PDF exported from Overleaf LaTeX report
│
├── requirements.txt
├── .gitignore
└── README.md
```

Author

Albert Kabore, MSN, MA, RN
