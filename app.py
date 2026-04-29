import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
import os
import requests
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CredExplain",
    page_icon="🏦",
    layout="wide"
)

# ── Download model files from Hugging Face ────────────────────────────────────
HF_BASE = "https://huggingface.co/mihirrajwow/credexplain-models/resolve/main"

def download_file(url, dest):
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = requests.get(url)
        with open(dest, "wb") as f:
            f.write(r.content)

@st.cache_resource
def load_model():
    download_file(f"{HF_BASE}/credit_model.json",  "data/credit_model.json")
    download_file(f"{HF_BASE}/feature_list.txt",   "data/feature_list.txt")
    
    model = xgb.XGBClassifier()
    model.load_model("data/credit_model.json")
    
    with open("data/feature_list.txt", "r") as f:
        features = [line.strip() for line in f.readlines()]
    
    return model, features

@st.cache_data
def load_data():
    download_file(f"{HF_BASE}/app_sample.csv", "data/app_sample.csv")
    return pd.read_csv("data/app_sample.csv")

model, features = load_model()
df = load_data()
X = df[features]
explainer = shap.TreeExplainer(model)

# ── Role passwords ────────────────────────────────────────────────────────────
ROLES = {
    "officer123":   "Loan Officer",
    "applicant123": "Applicant",
    "admin123":     "Admin",
}

# ── Fakability registry ───────────────────────────────────────────────────────
FAKABILITY = {
    "EXT_SOURCE_1":         ("🟢 Hard",   "Bureau score — historically verified, cannot be faked"),
    "EXT_SOURCE_2":         ("🟢 Hard",   "Bureau score — historically verified, cannot be faked"),
    "EXT_SOURCE_3":         ("🟢 Hard",   "Bureau score — historically verified, cannot be faked"),
    "DAYS_BIRTH":           ("🟢 Hard",   "Age from government ID — immutable"),
    "CREDIT_TO_GOODS":      ("🔴 Easy",   "Applicant controls loan amount requested — verify goods invoice independently"),
    "ANNUITY_TO_INCOME":    ("🟡 Medium", "Derived from income + loan amount — cross-check with payslips"),
    "DAYS_EMPLOYED":        ("🟡 Medium", "Employer tenure — verify via HR letter or EPF records"),
    "AMT_INCOME_TOTAL":     ("🟡 Medium", "Self-reported income — verify via ITR or Form 16"),
    "EMPLOYMENT_RATIO":     ("🟡 Medium", "Derived from employment + age — check source documents"),
    "DOCUMENT_COUNT":       ("🔴 Easy",   "Applicant submits documents — quantity does not guarantee quality"),
    "INCOME_PER_PERSON":    ("🟡 Medium", "Family size is self-reported — field verification recommended"),
    "CNT_CHILDREN":         ("🔴 Easy",   "Self-reported — easy to misrepresent"),
    "CONTACT_REACHABILITY": ("🔴 Easy",   "Phone numbers are easy to fabricate — do a live call check"),
    "NAME_EDUCATION_TYPE_ENC": ("🟡 Medium", "Verify via certificate — not independently checked by bureau"),
}

# ── Reason map ────────────────────────────────────────────────────────────────
REASON_MAP = {
    "EXT_SOURCE_1":            ("External credit score 1 is low",              "External credit score 1 is strong"),
    "EXT_SOURCE_2":            ("External credit score 2 is low",              "External credit score 2 is strong"),
    "EXT_SOURCE_3":            ("External credit score 3 is low",              "External credit score 3 is strong"),
    "CREDIT_TO_GOODS":         ("Loan amount exceeds goods price",             "Loan amount matched to goods price"),
    "ANNUITY_TO_INCOME":       ("Monthly repayment burden is heavy",           "Monthly repayment is manageable"),
    "EMPLOYMENT_RATIO":        ("Limited employment history",                  "Strong employment history"),
    "DAYS_EMPLOYED":           ("Short current job tenure",                    "Long stable job tenure"),
    "DAYS_BIRTH":              ("Age is a risk factor for this profile",       "Age is a positive factor"),
    "AMT_CREDIT":              ("Loan amount is high",                         "Loan amount is acceptable"),
    "AMT_INCOME_TOTAL":        ("Income is low relative to obligations",       "Income is strong relative to obligations"),
    "INCOME_PER_PERSON":       ("High household financial pressure",           "Healthy income per family member"),
    "DOCUMENT_COUNT":          ("Few documents provided",                      "Good documentation provided"),
    "EXT_SOURCE_1_MISSING":    ("No external credit score 1 available",        "External credit score 1 is available"),
    "NAME_EDUCATION_TYPE_ENC": ("Education level is a risk factor",            "Education level is a positive factor"),
    "CODE_GENDER_ENC":         ("Gender profile is a risk factor",             "Gender profile is a positive factor"),
    "NAME_FAMILY_STATUS_ENC":  ("Family status is a risk factor",              "Family status is a positive factor"),
    "CREDIT_TO_INCOME":        ("Total loan is large relative to income",      "Total loan is reasonable vs income"),
}

# ── Counterfactual map ────────────────────────────────────────────────────────
CF_MAP = {
    "ANNUITY_TO_INCOME":  lambda v: f"Repayment burden is {v:.1%} of income. Requesting a smaller loan or longer tenure would reduce this.",
    "CREDIT_TO_GOODS":    lambda v: f"Loan is {v:.2f}x the goods price. Aligning loan closer to goods value would reduce risk.",
    "CREDIT_TO_INCOME":   lambda v: f"Loan is {v:.1f}x annual income. A smaller loan would improve this ratio.",
    "DAYS_EMPLOYED":      lambda v: f"Current job tenure is {abs(v)/365:.1f} years. Longer tenure would strengthen this profile.",
    "DOCUMENT_COUNT":     lambda v: f"Only {int(v)} documents provided. Additional income and identity documents would help.",
    "INCOME_PER_PERSON":  lambda v: f"Income per family member is ₹{v:,.0f}/month. Demonstrating additional income sources would help.",
}
ACTIONABLE = list(CF_MAP.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────
def to_credit_score(risk):
    return int(round(900 - (risk * 600)))

def get_shap(idx):
    applicant = X.loc[[idx]]
    shap_vals = explainer.shap_values(applicant)[0]
    return pd.DataFrame({
        "feature":       features,
        "shap_value":    shap_vals,
        "feature_value": applicant.values[0]
    }).sort_values("shap_value", key=abs, ascending=False)

# ── Session state ─────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = None
if "applicant_access" not in st.session_state:
    st.session_state.applicant_access = False
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.role is None:
    st.title("🏦 CredExplain")
    st.subheader("Explainable Credit Decisioning System")
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("### Sign In")
        pwd = st.text_input("Access code", type="password")
        if st.button("Enter", use_container_width=True):
            if pwd in ROLES:
                st.session_state.role = ROLES[pwd]
                st.rerun()
            else:
                st.error("Invalid access code")
        st.caption("officer123 · applicant123 · admin123")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
role = st.session_state.role

with st.sidebar:
    st.markdown(f"### 🏦 CredExplain")
    st.caption(f"Signed in as: **{role}**")
    st.divider()

    if role in ["Loan Officer", "Admin"]:
        mode = st.radio(
            "Select applicant",
            ["Random High Risk", "Random Low Risk",
             "Borderline Case", "Enter ID manually"]
        )
        if mode == "Random High Risk":
            idx = df[df["TARGET"] == 1].sample(
                1, random_state=np.random.randint(999)).index[0]
        elif mode == "Random Low Risk":
            idx = df[df["TARGET"] == 0].sample(
                1, random_state=np.random.randint(999)).index[0]
        elif mode == "Borderline Case":
            scores = model.predict_proba(X)[:, 1]
            mask = (scores > 0.48) & (scores < 0.58)
            idx = df[mask].sample(
                1, random_state=np.random.randint(999)).index[0]
        else:
            idx = st.number_input(
                "Applicant ID",
                min_value=int(df.index.min()),
                max_value=int(df.index.max()),
                value=int(df.index[0])
            )
        st.session_state.audit_log.append(
            {"role": role, "applicant_id": int(idx), "action": "Viewed report"}
        )

    elif role == "Applicant":
        if st.session_state.applicant_access:
            idx = df[df["TARGET"] == 1].sample(1, random_state=77).index[0]
        else:
            st.warning("Access not yet enabled by your loan officer.")
            if st.button("Sign out"):
                st.session_state.role = None
                st.rerun()
            st.stop()

    if st.button("Sign out", use_container_width=True):
        st.session_state.role = None
        st.rerun()

# ── Compute scores ────────────────────────────────────────────────────────────
applicant    = X.loc[[idx]]
risk_score   = model.predict_proba(applicant)[:, 1][0]
credit_score = to_credit_score(risk_score)
decision     = "DECLINE" if risk_score > 0.5 else "APPROVE"
actual       = df.loc[idx, "TARGET"]
shap_df      = get_shap(idx)

risk_factors, protective_factors = [], []
for _, row in shap_df.head(12).iterrows():
    feat = row["feature"]
    if feat not in REASON_MAP:
        continue
    pos, neg = REASON_MAP[feat]
    if row["shap_value"] > 0:
        risk_factors.append((feat, pos, row["shap_value"], row["feature_value"]))
    else:
        protective_factors.append((feat, neg, row["shap_value"], row["feature_value"]))

# ══════════════════════════════════════════════════════════════════════════════
# LOAN OFFICER VIEW
# ══════════════════════════════════════════════════════════════════════════════
if role == "Loan Officer":
    st.title("Credit Decision Report")
    st.caption(f"Applicant ID: {idx}  |  Actual outcome: {'Defaulted' if actual==1 else 'Repaid'}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score",    f"{risk_score:.3f}", help="0 = safe · 1 = high risk")
    c2.metric("Credit Score",  credit_score,         help="300–900 scale")
    c3.metric("Decision",      decision)
    c4.metric("Actual Outcome","Defaulted" if actual==1 else "Repaid")

    if decision == "DECLINE":
        st.error(f"⛔ DECLINED — Risk score {risk_score:.3f} exceeds threshold 0.50")
    else:
        st.success(f"✅ APPROVED — Risk score {risk_score:.3f} is below threshold 0.50")

    st.divider()
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Decision Reasons")
        st.markdown("**Risk Factors**")
        for feat, reason, sv, fv in risk_factors[:3]:
            fake_level, fake_note = FAKABILITY.get(feat, ("⚪ Unknown", ""))
            st.markdown(
                f"- ⚠️ {reason}  \n"
                f"  <span style='font-size:12px;color:gray;'>"
                f"Fakability: {fake_level} — {fake_note}</span>",
                unsafe_allow_html=True
            )
        st.markdown("**Protective Factors**")
        for feat, reason, sv, fv in protective_factors[:3]:
            st.markdown(f"- ✅ {reason}")

        if decision == "DECLINE":
            st.divider()
            st.subheader("💡 What Could Change This Decision")
            actionable_rows = shap_df[
                (shap_df["feature"].isin(ACTIONABLE)) &
                (shap_df["shap_value"] > 0)
            ].head(2)
            shown = 0
            for _, row in actionable_rows.iterrows():
                feat = row["feature"]
                if feat in CF_MAP:
                    st.info(f"**#{shown+1}** — {CF_MAP[feat](row['feature_value'])}")
                    shown += 1
            if shown == 0:
                st.info("Primary risk is from bureau scores. Improved repayment history over 6–12 months is the strongest path to approval.")

    with right:
        st.subheader("SHAP Waterfall")
        explanation = explainer(applicant)
        fig, ax = plt.subplots(figsize=(7, 5))
        shap.plots.waterfall(explanation[0], show=False, max_display=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.caption("CredExplain v1.0 — For authorised loan officer use only")

# ══════════════════════════════════════════════════════════════════════════════
# APPLICANT VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif role == "Applicant":
    st.title("Your Credit Report")
    st.caption("This report has been shared with you by your loan officer.")
    st.divider()

    score_color = (
        "🟢" if credit_score >= 750 else
        "🟡" if credit_score >= 600 else "🔴"
    )
    c1, c2 = st.columns(2)
    c1.metric("Your Credit Score", f"{score_color} {credit_score}",
              help="300 = high risk · 900 = excellent")
    c2.metric("Decision", decision)

    if decision == "DECLINE":
        st.error("⛔ Your loan application has been declined at this time.")
    else:
        st.success("✅ Your loan application has been approved.")

    st.divider()
    st.subheader("What's Affecting Your Score")
    st.markdown("**Factors working against you:**")
    for feat, reason, sv, fv in risk_factors[:3]:
        st.markdown(f"- ⚠️ {reason}")

    st.markdown("**Factors working in your favour:**")
    for feat, reason, sv, fv in protective_factors[:3]:
        st.markdown(f"- ✅ {reason}")

    if decision == "DECLINE":
        st.divider()
        st.subheader("💡 How You Could Improve Your Score")
        actionable_rows = shap_df[
            (shap_df["feature"].isin(ACTIONABLE)) &
            (shap_df["shap_value"] > 0)
        ].head(2)
        shown = 0
        for _, row in actionable_rows.iterrows():
            feat = row["feature"]
            if feat in CF_MAP:
                st.info(f"**Suggestion {shown+1}:** {CF_MAP[feat](row['feature_value'])}")
                shown += 1
        if shown == 0:
            st.info("Your main risk factor is your credit history. Making consistent payments over the next 6–12 months is the most effective way to improve your score.")

    st.divider()
    st.caption("CredExplain v1.0 — Report shared by your lending institution")

# ══════════════════════════════════════════════════════════════════════════════
# ADMIN VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif role == "Admin":
    st.title("Admin Panel")
    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📊 Decision Report", "🔐 Access Control", "📋 Audit Log"
    ])

    with tab1:
        st.subheader(f"Applicant {idx} — Full Report")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk Score",   f"{risk_score:.3f}")
        c2.metric("Credit Score", credit_score)
        c3.metric("Decision",     decision)
        c4.metric("Actual",       "Defaulted" if actual==1 else "Repaid")

        if decision == "DECLINE":
            st.error(f"⛔ DECLINED — Score {risk_score:.3f}")
        else:
            st.success(f"✅ APPROVED — Score {risk_score:.3f}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Risk Factors + Fakability**")
            for feat, reason, sv, fv in risk_factors[:4]:
                fake_level, fake_note = FAKABILITY.get(feat, ("⚪ Unknown", ""))
                st.markdown(
                    f"- ⚠️ {reason}  \n"
                    f"  <span style='font-size:12px;color:gray;'>"
                    f"{fake_level} — {fake_note}</span>",
                    unsafe_allow_html=True
                )
            st.markdown("**Protective Factors**")
            for feat, reason, sv, fv in protective_factors[:3]:
                st.markdown(f"- ✅ {reason}")

        with right:
            explanation = explainer(applicant)
            fig, ax = plt.subplots(figsize=(7, 5))
            shap.plots.waterfall(explanation[0], show=False, max_display=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab2:
        st.subheader("Applicant Access Control")
        current = st.session_state.applicant_access
        status_label = "🟢 Enabled" if current else "🔴 Disabled"
        st.markdown(f"**Current applicant access:** {status_label}")
        st.caption("When enabled, applicants can sign in with code `applicant123` to view their score and improvement suggestions.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Enable Applicant Access",
                         disabled=current, use_container_width=True):
                st.session_state.applicant_access = True
                st.session_state.audit_log.append({
                    "role": "Admin", "applicant_id": "—",
                    "action": "Enabled applicant access"
                })
                st.rerun()
        with col2:
            if st.button("⛔ Disable Applicant Access",
                         disabled=not current, use_container_width=True):
                st.session_state.applicant_access = False
                st.session_state.audit_log.append({
                    "role": "Admin", "applicant_id": "—",
                    "action": "Disabled applicant access"
                })
                st.rerun()

    with tab3:
        st.subheader("Audit Log")
        if st.session_state.audit_log:
            st.dataframe(pd.DataFrame(st.session_state.audit_log),
                         use_container_width=True)
        else:
            st.info("No activity logged yet in this session.")

    st.divider()
    st.caption("CredExplain v1.0 — Admin access only")