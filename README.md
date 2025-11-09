Predicting 30-Day Hospital Readmissions Using Machine Learning
Overview

This project predicts 30-day hospital readmissions using a structured dataset of 30,000 simulated inpatient encounters.
It demonstrates a complete, reproducible data science workflow — from cleaning and EDA to feature engineering and model tuning.

The project supports healthcare systems in identifying high-risk patients at discharge to reduce avoidable readmissions and improve patient outcomes.

Clinical aim: Estimate 30-day readmission risk at discharge to support early intervention, medication reconciliation, and care coordination.

Research question:
Which patient and clinical characteristics best predict 30-day readmission, and how can predictive modeling improve post-discharge care?

Repository Links

📄 Overleaf report: Predicting Readmissions Report

💻 GitHub repo: Hospital-Readmission-Prediction

📊 Jupyter notebook: Readmission_Project.ipynb

🧾 Dataset: Dataset_Hospital_readmissions_30k.csv

📚 Kaggle source: Hospital Readmission Prediction (Synthetic Dataset)

Decision Framing

Primary metric: Recall (Sensitivity) – avoid missing high-risk patients.

Secondary metrics: Precision, F1-score, Accuracy.

Summary metrics: AUROC, AUPRC – handle class imbalance effectively.

This metric strategy ensures that clinically relevant predictions are prioritized over raw accuracy.

Outcome variable:
readmitted_30_days (Yes/No), mapped to numeric target readmitted (1/0).

Dataset Description

File: Dataset_Hospital_readmissions_30k.csv
Size: 30,000 records × 12 features
Target: readmitted_30_days (binary)
Missing values: None
Duplicates: None
Memory usage: ~10.5 MB

Data Dictionary
Column	Description	Type	Example
patient_id	Unique patient encounter ID	int	1
age	Patient age (18–90 years)	int	74
gender	Patient gender	object	Male
blood_pressure	Blood pressure (systolic/diastolic)	object	130/72
cholesterol	Serum cholesterol (mg/dL)	int	240
bmi	Body mass index	float	31.5
diabetes	Diabetes flag	object	Yes
hypertension	Hypertension flag	object	No
medication_count	Medications prescribed	int	5
length_of_stay	Hospital stay duration (days)	int	3
discharge_destination	Discharge destination	object	Home
readmitted_30_days	Readmission within 30 days	object	No

Readmission rate: 12.25%
Imbalance ratio: ~7.2 : 1 (No : Yes)

Environment Setup
# Clone the repository
git clone https://github.com/albertokabore/Hospital-Readmission-Prediction
cd Hospital-Readmission-Prediction

# Create and activate virtual environment
py -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
py -m pip install --upgrade pip
py -m pip install jupyterlab pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost pyarrow

# Save environment
py -m pip freeze > requirements.txt


Recommended .gitignore entries:

.venv/
__pycache__/
.ipynb_checkpoints/
*.csv
*.log

Data Cleaning and Curation

Standardized column names and checked data types.

Removed 0 duplicate rows.

Confirmed no missing values (0%).

Encoded the target variable:

df["readmitted"] = df["readmitted_30_days"].map({"Yes": 1, "No": 0})


Reviewed descriptive statistics for both numeric and categorical data.

✅ Clean, validated dataset ready for EDA and modeling.

Exploratory Data Analysis (EDA)

EDA was conducted in Jupyter using pandas, matplotlib, and seaborn.

Numeric distributions

Age and cholesterol are right-skewed.

Longer hospital stays and higher BMI increase readmission risk.

Categorical distributions

69% of patients discharged home; 27% to nursing facilities.

Gender is roughly balanced across the dataset.

Correlations

Weak but meaningful relationships found between medication_count, length_of_stay, and readmission.

Target distribution

12.25% readmitted vs 87.75% not readmitted — confirming class imbalance.

SMOTE was applied later during training to balance the classes.

Feature Engineering

Clinical Complexity Score
A composite quantitative indicator combining:
age, blood_pressure, cholesterol, bmi, diabetes, hypertension, medication_count, and length_of_stay.

Steps:

Standardize numeric indicators (z-score).

Encode comorbidities (Yes = 1, No = 0).

Normalize scores to 0–100.

Categorize as Low, Medium, High, or Very High complexity.

Other engineered features:

Binary encoding for categorical variables.

Parsing blood_pressure into systolic/diastolic columns.

Added interaction terms for multivariate risk detection.

Modeling Pipeline

Train-test split: 80/20 stratified.

Scaling: StandardScaler applied to numeric features.

Balancing: SMOTE used to handle class imbalance.

Models tested:

Decision Tree

Random Forest

Gradient Boosting

AdaBoost

XGBoost

Evaluation metrics:
Accuracy, Precision, Recall, F1-score, ROC AUC, PR AUC

Tuning:
RandomizedSearchCV optimized Random Forest hyperparameters for better recall and interpretability.

Key Results

Dataset successfully balanced after SMOTE.

Tuned Random Forest achieved the highest recall and robust ROC AUC.

Top predictors: medication_count, length_of_stay, bmi.

Model shows strong potential for clinical decision support in real-world discharge planning.

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
│   └── Overleaf_Report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md

