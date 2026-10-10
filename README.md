# JobFair AI: Bias Auditing, Explainability & Mitigation for Hiring & Credit Models

> **IBM SkillsBuild for Academia Internship Project** — AI & Machine Learning Track

JobFair AI is an interactive workbench designed to audit, explain, and mitigate demographic bias in machine learning models used for hiring shortlisting and credit scoring. 

The goal of this project is to make fairness testing practical and easy to run locally, allowing developers and risk managers to evaluate models against standard regulatory guidelines like the **EEOC 80% (4/5ths) rule** and inspect feature-level decisions using **SHAP values**.

---

## 🛠️ Key Features

- **Data Explorer:** Load benchmark datasets (UCI Adult Income, German Credit) or upload custom CSV files to inspect group representation and base-rate disparities.
- **Model Training Pipeline:** Train classification models (**Logistic Regression**, **Random Forest**, **XGBoost**) with deterministic random seeds and baseline quality warnings.
- **Bias Audit Engine:** Evaluate group fairness metrics including **Disparate Impact Ratio**, **Demographic Parity Difference**, **Equal Opportunity Difference (TPR parity)**, **Average Odds Difference**, and **Theil Index** (individual fairness).
- **Bias Mitigation:** Apply debiasing techniques such as **AIF360 Reweighing** (pre-processing) and **Fairlearn Threshold Optimizer** (post-processing) and inspect the trade-off between model accuracy and fairness.
- **Explainability (SHAP):** Global feature importance charts, individual candidate recourse waterfall plots, and demographic SHAP impact comparisons.
- **PDF Report Export:** Automatically compile audit results, fairness tables, and compliance checklists into a downloadable PDF report.

---

## 📁 Repository Structure

```text
jobfair-ai/
├── app/
│   ├── Home.py                  # Main Streamlit dashboard
│   ├── components/
│   │   ├── sidebar.py           # Shared sidebar & navigation workflow
│   │   └── charts.py            # Plotly visualisations
│   └── pages/
│       ├── 1_Data.py            # Data Explorer page
│       ├── 2_Model.py           # Model Training page
│       ├── 3_Audit.py           # Bias Audit page
│       ├── 4_Mitigation.py      # Bias Mitigation page
│       ├── 5_Explainability.py  # SHAP Explainability page
│       ├── 6_Report.py          # Executive Report PDF page
│       └── 7_Ethics.py          # Limitations & Ethics page
├── core/
│   ├── data_loader.py           # Ingestion, cleaning, & AIF360 dataset wrappers
│   ├── models.py                # ML model training, evaluation, & serialisation
│   ├── bias_metrics.py          # Group & individual fairness auditing engine
│   ├── mitigation.py            # Reweighing & Threshold Optimizer algorithms
│   ├── explainability.py        # SHAP calculation & disparity breakdown
│   ├── report_gen.py            # ReportLab PDF report compiler
│   └── db.py                    # SQLite ORM persistence for audit history
├── tests/
│   ├── test_data_loader.py      # Dataset loader unit tests
│   ├── test_models.py           # Model pipeline unit tests
│   ├── test_bias_metrics.py     # Bias metrics unit tests
│   ├── test_mitigation.py       # Debiasing algorithm tests
│   ├── test_explainability.py   # SHAP explainability unit tests
│   ├── test_report.py           # PDF report generator unit tests
│   ├── test_db.py               # SQLite CRUD unit tests
│   └── test_integration.py      # Full end-to-end integration test
├── data/                        # Local dataset caches
├── models/                      # Saved trained models (.joblib)
├── requirements.txt             # Dependency specifications
├── pyproject.toml               # Project config
└── README.md                    # Project documentation
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 recommended)

### 2. Virtual Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/jobfair-ai.git
cd jobfair-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Web Application

To start the interactive Streamlit dashboard:

```bash
streamlit run app/Home.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 🧪 Running Unit & Integration Tests

All 35 unit and end-to-end integration tests can be executed via `pytest`:

```bash
pytest tests/ -v
```

---

## 🏛️ Guidelines & Frameworks Evaluated

- **EEOC Title VII (USA):** Tests whether the unprivileged selection rate is at least 80% (4/5ths rule) of the privileged selection rate.
- **EU AI Act Guidelines:** Audit trail documentation and risk management for automated high-risk decision systems.
- **NIST AI RMF 1.0:** Empirical measurement of demographic disparity and management of accuracy vs. fairness trade-offs.

---

## 📄 License & Disclaimer

Academic internship prototype developed for the **IBM SkillsBuild for Academia Internship Program**. This tool is meant to assist human decision-makers and risk managers, not to replace human oversight.
