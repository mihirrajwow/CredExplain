# CredExplain 🏦
### Explainable Credit Decisioning System

An end-to-end credit risk scoring system that goes beyond a black-box score —
it tells loan officers *why* an applicant was declined and *what* they can do to get approved.

---

## 🎯 The Problem

Traditional credit models output a score with no explanation.
- Regulators increasingly require explainability (RBI guidelines on AI in lending)
- Loan officers cannot justify decisions to applicants
- Applicants have no actionable path to improve their creditworthiness

## ✅ What This System Does

| Feature | Description |
|---|---|
| **Risk Score** | XGBoost model, AUC-ROC 0.754, trained on 307k applicants |
| **Credit Score** | 300–900 scale (familiar to lenders and borrowers) |
| **SHAP Explanations** | Per-applicant waterfall charts showing feature contributions |
| **Reason Codes** | Plain English explanations of every decision |
| **Counterfactuals** | Actionable suggestions — *"reduce loan amount to get approved"* |
| **Fakability Flags** | Alerts loan officers to easily-manipulated signals |
| **3-Tier Access** | Loan Officer / Applicant / Admin role system |
| **REST API** | FastAPI endpoint deployable on Render |

---

## 🏗️ Architecture

```
Data Inputs          ML Model Layer        Explainability Layer    Output
─────────────        ──────────────        ────────────────────    ──────
UPI patterns    →    Feature              SHAP analysis      →    Loan Officer
Bank cash flow  →    Engineering     →    Reason codes       →    Dashboard
Bill regularity →    XGBoost model        Counterfactuals    →    REST API
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.754 |
| Training samples | 246,008 |
| Validation samples | 61,503 |
| Features used | 26 |
| Class imbalance handling | scale_pos_weight = 11.39 |

---

## 🖥️ Dashboard — 3 Role System

| Role | Access Code | What They See |
|---|---|---|
| Loan Officer | `officer123` | Risk score + SHAP waterfall + fakability flags |
| Applicant | `applicant123` | Credit score + plain English reasons + improvement tips |
| Admin | `admin123` | Everything + access control toggle + audit log |

---

## 🚀 Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/CredExplain.git
cd CredExplain
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add data files** *(not included — download from Kaggle)*
- [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
- Place `application_train.csv` in `data/`
- Run notebooks 02 through 11 in order to generate model files

**4. Launch dashboard**
```bash
streamlit run app.py
```

**5. Launch API**
```bash
uvicorn api:app --reload
# Swagger UI: http://127.0.0.1:8000/docs
```

---

## 📁 Project Structure

```
CredExplain/
├── app.py                    # Streamlit dashboard (3-tier)
├── api.py                    # FastAPI scoring endpoint
├── requirements.txt
├── notebooks/
│   ├── 02_pandas_basics.ipynb
│   ├── 03_explore_homecredit.ipynb
│   ├── 05_clean_data.ipynb
│   ├── 07_feature_engineering.ipynb
│   ├── 08_train_model.ipynb
│   ├── 10_shap_explainability.ipynb
│   └── 11_reason_codes.ipynb
├── data/                     # CSVs and model files (not tracked)
└── assets/                   # SHAP visualisations
```

---

## 🔌 API Usage

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "EXT_SOURCE_2": 0.55,
    "EXT_SOURCE_3": 0.60,
    "ANNUITY_TO_INCOME": 0.18,
    "CREDIT_TO_GOODS": 1.1,
    ...
  }'
```

**Response:**
```json
{
  "risk_score": 0.2249,
  "credit_score": 765,
  "decision": "APPROVE",
  "risk_factors": ["Loan amount is high"],
  "protective_factors": ["External credit score 3 is strong"],
  "counterfactuals": [],
  "shap_top_features": [...]
}
```

---

## 🗺️ Production Roadmap

- [ ] Integrate with India's AA (Account Aggregator) framework via Setu/Onemoney
- [ ] Replace proxy dataset with real UPI + bank statement features
- [ ] Add PostgreSQL audit log persistence
- [ ] Add applicant-level authentication (replace demo password system)
- [ ] RBI explainability compliance report generation

---

## 🛠️ Built With

Python · XGBoost · SHAP · Streamlit · FastAPI · Pandas · Scikit-learn

---

*Built as a demonstration of explainable AI in credit decisioning.
Trained on Home Credit Default Risk dataset as a proxy for AA-framework data.*