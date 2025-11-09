# Predicting 30-Day Hospital Readmissions Using Machine Learning

## Overview

This project predicts **30-day hospital readmissions** using a structured dataset of **30,000 simulated inpatient encounters**. It implements a complete, reproducible data science workflow:

- Data loading and validation
- Exploratory data analysis (EDA)
- Feature engineering (including a clinical complexity score)
- Model training, class imbalance handling, and hyperparameter tuning
- Model evaluation and interpretation

The goal is to help healthcare systems **identify high-risk patients at discharge**, support value-based care, and reduce avoidable readmissions.

- **Clinical aim:** Estimate 30-day readmission risk at discharge to prioritize early follow-up, medication reconciliation, and transitional care.
- **Research question:** Which patient and clinical characteristics best predict 30-day readmission, and how can predictive modeling support safer and more efficient discharge planning?

---

## Repository Links

- **Overleaf report:**
  https://www.overleaf.com/read/vbjqbzwcgjjv#6eb481

- **GitHub repository:**
  https://github.com/albertokabore/Hospital-Readmission-Prediction

- **Jupyter notebook (main analysis):**
  https://github.com/albertokabore/Hospital-Readmission-Prediction/blob/main/notebooks/Readmission_Project.ipynb

- **Dataset (project copy):**
  https://github.com/albertokabore/Hospital-Readmission-Prediction/blob/main/data/Dataset_Hospital_readmissions_30k.csv

- **Original Kaggle source:**
  https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset

---

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

```bash
git clone https://github.com/albertokabore/Hospital-Readmission-Prediction
cd Hospital-Readmission-Prediction

2. Create and activate virtual environment
py -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux (if desired)
# source .venv/bin/activate

3. Install dependencies
py -m pip install --upgrade pip
py -m pip install jupyterlab pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost pyarrow

4. Save environment
py -m pip freeze > requirements.txt

5. Recommended .gitignore
.venv/
__pycache__/
.ipynb_checkpoints/
*.csv
*.log

Tools and Libraries

Python, Jupyter Notebook, VS Code

pandas – data manipulation

numpy – numerical computing

matplotlib, seaborn – visualization and EDA

scikit-learn – modeling, metrics, preprocessing

imbalanced-learn – SMOTE for class imbalance

xgboost – gradient boosting model

Data Cleaning and Curation

The dataset is relatively clean and synthetic, so the cleaning steps focused on validation and light curation rather than heavy correction.

Main Steps

Column standardization

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

Type validation

Numeric features:
age, cholesterol, bmi, medication_count, length_of_stay, readmitted

Categorical features:
gender, blood_pressure, diabetes, hypertension, discharge_destination, readmitted_30_days

The data curation step confirms that the dataset is consistent, complete, and ready for EDA, feature engineering, and modeling without the need for imputation.

Exploratory Data Analysis (EDA)

EDA is performed in:
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

Insights:

Age and cholesterol show mild right skew.

BMI clusters around the overweight/obese range in many patients.

Length of stay is typically short (3–8 days), but longer stays are more common in readmitted patients.

Medication count tends to be higher for those later readmitted.

Categorical Features

diabetes and hypertension show substantial prevalence, consistent with chronic disease populations.

discharge_destination is dominated by Home, followed by Nursing facility.

gender shows a roughly balanced distribution across the sample.

Target Distribution

Overall readmission rate is low (~12.25%), confirming class imbalance.

This motivates:

Use of SMOTE (Synthetic Minority Oversampling Technique) on the training data.

Emphasis on recall and PR AUC in evaluation.

Feature Engineering

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

Create categorical bands:

Low complexity

Medium complexity

High complexity

Very High complexity

This score is used both:

As a numeric predictor: clinical_complexity_score, and

As a clinical segment variable: complexity_category to compare readmission rates across burden strata.

2. Additional Engineered Features (Conceptual)

Parse blood_pressure into:

systolic_bp

diastolic_bp

Encode all categorical variables as numeric (e.g., one-hot encoding for discharge_destination).

Explore interactions (e.g., age × length_of_stay, bmi × diabetes).

These engineered features improve model capacity to capture non-linear and interaction effects important in clinical risk.

Modeling Pipeline

The modeling pipeline is implemented in the notebook and includes:

Train–Test Split

80% training, 20% test

Stratified by readmitted to preserve class distribution

Scaling

StandardScaler applied to numeric features before modeling.

Class Imbalance Handling

SMOTE applied on the training data to oversample the minority class (readmitted = 1).

The test set remains untouched to represent real-world prevalence.

Models Evaluated

Decision Tree

Random Forest

Gradient Boosting

AdaBoost

XGBoost

Evaluation Metrics

Accuracy

Precision

Recall (Sensitivity)

F1-score

ROC AUC

Precision–Recall AUC

Hyperparameter Tuning

RandomizedSearchCV on Random Forest with parameters such as:

n_estimators

max_depth

min_samples_split

min_samples_leaf

max_features

The tuned Random Forest is then evaluated and compared to baseline models.

Key Results (Summary)

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

Repository Structure
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

Author

Albert Kabore, MSN, MA, RN
