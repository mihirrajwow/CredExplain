from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle
import shap
from typing import Optional

# ── Load model ────────────────────────────────────────────
with open("data/credit_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("data/feature_list.pkl", "rb") as f:
    features = pickle.load(f)

explainer = shap.TreeExplainer(model)

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="CredExplain API",
    description="Explainable credit scoring API — returns risk score, credit score, reason codes and counterfactuals.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Reason map ────────────────────────────────────────────
REASON_MAP = {
    "EXT_SOURCE_1":            ("External credit score 1 is low",         "External credit score 1 is strong"),
    "EXT_SOURCE_2":            ("External credit score 2 is low",         "External credit score 2 is strong"),
    "EXT_SOURCE_3":            ("External credit score 3 is low",         "External credit score 3 is strong"),
    "CREDIT_TO_GOODS":         ("Loan amount exceeds goods price",         "Loan amount matched to goods price"),
    "ANNUITY_TO_INCOME":       ("Monthly repayment burden is heavy",       "Monthly repayment is manageable"),
    "EMPLOYMENT_RATIO":        ("Limited employment history",              "Strong employment history"),
    "DAYS_EMPLOYED":           ("Short current job tenure",                "Long stable job tenure"),
    "DAYS_BIRTH":              ("Age is a risk factor for this profile",   "Age is a positive factor"),
    "AMT_CREDIT":              ("Loan amount is high",                     "Loan amount is acceptable"),
    "AMT_INCOME_TOTAL":        ("Income is low relative to obligations",   "Income is strong relative to obligations"),
    "INCOME_PER_PERSON":       ("High household financial pressure",       "Healthy income per family member"),
    "DOCUMENT_COUNT":          ("Few documents provided",                  "Good documentation provided"),
    "EXT_SOURCE_1_MISSING":    ("No external credit score 1 available",    "External credit score 1 is available"),
    "NAME_EDUCATION_TYPE_ENC": ("Education level is a risk factor",        "Education level is a positive factor"),
    "CREDIT_TO_INCOME":        ("Total loan is large relative to income",  "Total loan is reasonable vs income"),
}

CF_MAP = {
    "ANNUITY_TO_INCOME":  lambda v: f"Repayment burden is {v:.1%} of income. Requesting a smaller loan or longer tenure would reduce this.",
    "CREDIT_TO_GOODS":    lambda v: f"Loan is {v:.2f}x the goods price. Aligning loan closer to goods value would reduce risk.",
    "CREDIT_TO_INCOME":   lambda v: f"Loan is {v:.1f}x annual income. A smaller loan would improve this ratio.",
    "DAYS_EMPLOYED":      lambda v: f"Current job tenure is {abs(v)/365:.1f} years. Longer tenure would strengthen this profile.",
    "DOCUMENT_COUNT":     lambda v: f"Only {int(v)} documents provided. Additional documents would help.",
    "INCOME_PER_PERSON":  lambda v: f"Income per family member is {v:,.0f}/month. Additional income sources would help.",
}
ACTIONABLE = list(CF_MAP.keys())

# ── Request schema ────────────────────────────────────────
class ApplicantFeatures(BaseModel):
    # Engineered features
    ANNUITY_TO_INCOME:          float = Field(..., example=0.18)
    CREDIT_TO_INCOME:           float = Field(..., example=3.5)
    CREDIT_TO_GOODS:            float = Field(..., example=1.1)
    EMPLOYMENT_RATIO:           float = Field(..., example=0.4)
    INCOME_PER_PERSON:          float = Field(..., example=85000)
    CHILDREN_RATIO:             float = Field(..., example=0.1)
    # External scores
    EXT_SOURCE_1:               float = Field(..., example=0.5)
    EXT_SOURCE_1_MISSING:       int   = Field(..., example=0)
    EXT_SOURCE_2:               float = Field(..., example=0.55)
    EXT_SOURCE_3:               float = Field(..., example=0.6)
    # Core financials
    AMT_INCOME_TOTAL:           float = Field(..., example=150000)
    AMT_CREDIT:                 float = Field(..., example=500000)
    AMT_ANNUITY:                float = Field(..., example=27000)
    DAYS_BIRTH:                 int   = Field(..., example=-12000)
    DAYS_EMPLOYED:              float = Field(..., example=-1500)
    DAYS_EMPLOYED_MISSING:      int   = Field(..., example=0)
    CNT_CHILDREN:               int   = Field(..., example=0)
    CNT_FAM_MEMBERS:            float = Field(..., example=2.0)
    # Encoded categoricals
    NAME_CONTRACT_TYPE_ENC:     int   = Field(..., example=0)
    CODE_GENDER_ENC:            int   = Field(..., example=1)
    NAME_INCOME_TYPE_ENC:       int   = Field(..., example=0)
    NAME_EDUCATION_TYPE_ENC:    int   = Field(..., example=1)
    NAME_FAMILY_STATUS_ENC:     int   = Field(..., example=0)
    OCCUPATION_TYPE_ENC:        int   = Field(..., example=2)
    # Aggregated flags
    DOCUMENT_COUNT:             int   = Field(..., example=3)
    CONTACT_REACHABILITY:       int   = Field(..., example=3)

# ── Response schema ───────────────────────────────────────
class ScoreResponse(BaseModel):
    applicant_id:        Optional[str]
    risk_score:          float
    credit_score:        int
    decision:            str
    risk_factors:        list[str]
    protective_factors:  list[str]
    counterfactuals:     list[str]
    shap_top_features:   list[dict]

# ── Health check ──────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "CredExplain API v1.0",
        "endpoints": ["/score", "/docs"]
    }

# ── Main scoring endpoint ─────────────────────────────────
@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score_applicant(data: ApplicantFeatures, applicant_id: Optional[str] = None):
    try:
        # Build feature vector in correct order
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])[features]

        # Risk score + credit score
        risk_score   = float(model.predict_proba(input_df)[:, 1][0])
        credit_score = int(round(900 - (risk_score * 600)))
        decision     = "DECLINE" if risk_score > 0.5 else "APPROVE"

        # SHAP values
        shap_vals = explainer.shap_values(input_df)[0]
        shap_df = pd.DataFrame({
            "feature":       features,
            "shap_value":    shap_vals,
            "feature_value": input_df.values[0]
        }).sort_values("shap_value", key=abs, ascending=False)

        # Reason codes
        risk_factors, protective_factors = [], []
        for _, row in shap_df.head(12).iterrows():
            feat = row["feature"]
            if feat not in REASON_MAP:
                continue
            pos, neg = REASON_MAP[feat]
            if row["shap_value"] > 0:
                risk_factors.append(pos)
            else:
                protective_factors.append(neg)

        # Counterfactuals
        counterfactuals = []
        if decision == "DECLINE":
            actionable_rows = shap_df[
                (shap_df["feature"].isin(ACTIONABLE)) &
                (shap_df["shap_value"] > 0)
            ].head(2)
            for _, row in actionable_rows.iterrows():
                feat = row["feature"]
                if feat in CF_MAP:
                    counterfactuals.append(CF_MAP[feat](row["feature_value"]))
            if not counterfactuals:
                counterfactuals.append(
                    "Primary risk is from bureau scores. "
                    "Improving repayment history over 6-12 months is the strongest path to approval."
                )

        # Top SHAP features for transparency
        shap_top = shap_df.head(5)[["feature", "shap_value", "feature_value"]].to_dict("records")

        return ScoreResponse(
            applicant_id       = applicant_id,
            risk_score         = round(risk_score, 4),
            credit_score       = credit_score,
            decision           = decision,
            risk_factors       = risk_factors[:3],
            protective_factors = protective_factors[:3],
            counterfactuals    = counterfactuals,
            shap_top_features  = [
                {
                    "feature":       r["feature"],
                    "shap_value":    round(float(r["shap_value"]), 4),
                    "feature_value": round(float(r["feature_value"]), 4)
                }
                for r in shap_top
            ]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))