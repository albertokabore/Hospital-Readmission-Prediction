# 🏥 Predicting Thirty-Day Hospital Readmissions Using Machine Learning

## 📘 Project Overview
This project develops a supervised machine-learning model to predict whether a patient will be readmitted within **30 days** of discharge.
Using a synthetic dataset of 30 000 inpatient encounters, the goal is to identify high-risk patients early and support **value-based care** through proactive outreach, medication reconciliation, and improved discharge planning.

**Objectives**
- Reduce hospital readmissions through data-driven prediction
- Build a transparent, reproducible data-science workflow
- Improve patient outcomes and reduce preventable costs

**Clinical Motivation**
Hospital readmissions are a national quality-of-care concern.
The Centers for Medicare & Medicaid Services (CMS) established the **Hospital Readmissions Reduction Program (HRRP)** to encourage hospitals to improve post-discharge care coordination and reduce avoidable readmissions.

---

##  Research Question
> *Which patients are most likely to be readmitted within thirty days of discharge?*

**Dependent variable:**
`readmitted_30_days` → Encoded as binary (`1 = Yes`, `0 = No`)

**Independent variables:**
Demographics, clinical burden, medication count, and discharge destination.

---

##  Dataset Description
- **File:** `data/Dataset_Hospital_readmissions_30k.csv`
- **Rows:** 30 000
- **Columns:** 12
- **Source:** [Kaggle – Hospital Readmission Prediction (Synthetic Dataset)](https://www.kaggle.com/datasets/siddharth0935/hospital-readmission-predictionsynthetic-dataset)

### Data Dictionary

| Column | Description | Type |
|--------|--------------|------|
| `patient_id` | Unique encounter ID | Integer |
| `age` | Patient age (18–90) | Integer |
| `gender` | Male, Female, or Other | Categorical |
| `blood_pressure` | Systolic/Diastolic (e.g., 130/72) | String |
| `cholesterol` | Total cholesterol (mg/dL) | Integer |
| `bmi` | Body-mass index | Float |
| `diabetes` | Yes/No | Binary |
| `hypertension` | Yes/No | Binary |
| `medication_count` | Number of active medications | Integer |
| `length_of_stay` | Hospital stay length (days) | Integer |
| `discharge_destination` | Home, Nursing Facility, etc. | Categorical |
| `readmitted_30_days` | 30-day readmission (Yes/No) | Binary |

**No missing or duplicate values** were detected.

---

## ⚙️ Environment Setup

### 1. Create and Activate Virtual Environment
```bash
# Create environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate
