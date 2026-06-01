import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.predict import predict_loan

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(83, 74, 183, 0.3) 0%, transparent 70%);
        pointer-events: none;
    }

    .main-header h1 {
        font-family: 'DM Serif Display', serif;
        color: #ffffff;
        font-size: 2.2rem;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(255,255,255,0.65);
        margin: 0;
        font-size: 1rem;
        font-weight: 300;
    }

    .model-badge {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        color: rgba(255,255,255,0.85);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 4px 3px;
    }

    .section-card {
        background: #ffffff;
        border: 0.5px solid #e8e8e8;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .section-title::after {
        content: '';
        flex: 1;
        height: 1px;
        background: #f0f0f0;
    }

    .result-approved {
        background: linear-gradient(135deg, #0f6e56 0%, #1D9E75 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }

    .result-rejected {
        background: linear-gradient(135deg, #993C1D 0%, #D85A30 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }

    .result-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        margin: 0.5rem 0;
    }

    .result-confidence {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }

    .metric-chip {
        background: #f6f6f9;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }

    .metric-chip .val {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
    }

    .metric-chip .lbl {
        font-size: 0.72rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 2px;
    }

    .risk-bar-wrap {
        margin: 1rem 0;
    }

    .risk-bar-bg {
        height: 10px;
        background: #f0f0f0;
        border-radius: 999px;
        overflow: hidden;
    }

    .risk-bar-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 0.4s ease;
    }

    .stButton > button {
        background: linear-gradient(135deg, #534AB7 0%, #7F77DD 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        width: 100% !important;
        cursor: pointer !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.88 !important;
    }

    .tip-box {
        background: #eeedfe;
        border-left: 3px solid #534AB7;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.88rem;
        color: #3C3489;
        margin-top: 0.8rem;
    }

    .history-item {
        background: #fafafa;
        border: 0.5px solid #ececec;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.88rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSlider"] label {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: #555 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Loan Approval Predictor</h1>
    <p>Stacking Classifier — AI-powered credit risk assessment</p>
    <div style="margin-top:1rem;">
        <span class="model-badge">Random Forest</span>
        <span class="model-badge">Logistic Regression</span>
        <span class="model-badge">XGBoost</span>
        <span class="model-badge">Meta-Learner: LR</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: History ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Prediction History")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-8:])):
            icon = "✅" if h["result"] == "Approved" else "❌"
            st.markdown(f"""
            <div class="history-item">
                <span>{icon} <b>₹{h['loan']:,}</b> — {h['area']}</span>
                <span style="color:#888;font-size:0.8rem;">{h['conf']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption(
        "This tool uses an ensemble stacking classifier trained on historical "
        "loan application data. Results are indicative only."
    )

# ── Main form ────────────────────────────────────────────────────────────────
col_form, col_result = st.columns([3, 2], gap="large")

with col_form:
    # Personal Details
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        married = st.selectbox("Marital Status", ["Yes", "No"], format_func=lambda x: "Married" if x == "Yes" else "Single")
    with c3:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

    c4, c5 = st.columns(2)
    with c4:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with c5:
        self_employed = st.selectbox("Employment Type", ["No", "Yes"],
                                     format_func=lambda x: "Salaried" if x == "No" else "Self-Employed")

    st.markdown('</div>', unsafe_allow_html=True)

    # Financial Details
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💰 Financial Details</div>', unsafe_allow_html=True)

    c6, c7 = st.columns(2)
    with c6:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, step=1000, value=50000)
    with c7:
        coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, step=1000, value=0)

    total_income = applicant_income + coapplicant_income
    st.markdown(f"""
    <div style="font-size:0.83rem; color:#534AB7; background:#eeedfe; border-radius:8px;
                padding:0.5rem 0.8rem; margin-bottom:0.8rem;">
        💡 Combined monthly income: <b>₹{total_income:,}</b>
    </div>
    """, unsafe_allow_html=True)

    c8, c9 = st.columns(2)
    with c8:
        loan_amount = st.number_input("Loan Amount (₹ 000s)", min_value=0, step=10, value=150)
    with c9:
        loan_term = st.selectbox("Loan Term (months)",
                                 [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                                 index=9)

    # EMI calculator
    if loan_amount > 0 and loan_term > 0:
        monthly_rate = 0.085 / 12   # assumed 8.5% p.a.
        n = loan_term
        emi = (loan_amount * 1000 * monthly_rate * (1 + monthly_rate) ** n) / ((1 + monthly_rate) ** n - 1)
        emi_ratio = (emi / total_income * 100) if total_income > 0 else 0
        emi_color = "#0f6e56" if emi_ratio <= 40 else "#993C1D"
        st.markdown(f"""
        <div style="font-size:0.83rem; color:{emi_color}; background:#f6f6f9; border-radius:8px;
                    padding:0.5rem 0.8rem; margin-bottom:0.8rem;">
            📊 Estimated EMI: <b>₹{emi:,.0f}/mo</b> &nbsp;|&nbsp; EMI-to-income ratio: <b>{emi_ratio:.1f}%</b>
            {"&nbsp;✅ Healthy" if emi_ratio <= 40 else "&nbsp;⚠️ High — may reduce chances"}
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Property & Credit
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🏠 Property & Credit</div>', unsafe_allow_html=True)

    c10, c11 = st.columns(2)
    with c10:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    with c11:
        credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                      format_func=lambda x: "✅ Good (1.0)" if x == 1.0 else "❌ Poor (0.0)")

    if credit_history == 0.0:
        st.markdown("""
        <div class="tip-box">
            ⚠️ Poor credit history significantly reduces approval chances.
            Consider improving your CIBIL score before applying.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Additional Features ────────────────────────────────────────────────
    with st.expander("⚙️ Additional Details (optional)"):
        c12, c13 = st.columns(2)
        with c12:
            existing_loans = st.number_input("Existing Active Loans", min_value=0, max_value=10, value=0)
        with c13:
            purpose = st.selectbox("Loan Purpose",
                                   ["Home Purchase", "Home Renovation", "Business", "Education", "Medical", "Other"])

        c14, c15 = st.columns(2)
        with c14:
            cibil_score = st.slider("CIBIL Score", 300, 900, 700, step=10)
        with c15:
            collateral = st.selectbox("Collateral Available", ["Yes", "No"])

        employment_years = st.slider("Years in Current Employment", 0, 40, 3)

    # ── Submit ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Loan Status")

# ── Result Panel ─────────────────────────────────────────────────────────────
with col_result:
    if predict_clicked:
        features = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area,
        }

        with st.spinner("Analysing application…"):
            prediction, probability = predict_loan(features)

        approved = prediction == 1
        conf = probability if approved else (1 - probability)

        # Save to history
        st.session_state.last_result = {
            "prediction": prediction,
            "probability": probability,
            "approved": approved,
            "conf": conf,
            "loan": loan_amount * 1000,
            "area": property_area,
        }
        st.session_state.history.append({
            "result": "Approved" if approved else "Rejected",
            "conf": conf,
            "loan": loan_amount * 1000,
            "area": property_area,
        })

    if st.session_state.last_result:
        r = st.session_state.last_result
        approved = r["approved"]
        conf = r["conf"]
        probability = r["probability"]

        # Main verdict
        css_cls = "result-approved" if approved else "result-rejected"
        verdict_icon = "✅" if approved else "❌"
        verdict_text = "Loan Approved" if approved else "Loan Rejected"
        st.markdown(f"""
        <div class="{css_cls}">
            <div style="font-size:2.5rem;">{verdict_icon}</div>
            <div class="result-title">{verdict_text}</div>
            <div class="result-confidence">Confidence: {conf:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Metric chips
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-chip">
                <div class="val">{probability:.0%}</div>
                <div class="lbl">Approval Prob.</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            emi_display = "N/A"
            if loan_amount > 0 and loan_term > 0:
                mr = 0.085 / 12
                n = loan_term
                emi_v = (loan_amount * 1000 * mr * (1 + mr) ** n) / ((1 + mr) ** n - 1)
                emi_display = f"₹{emi_v:,.0f}"
            st.markdown(f"""
            <div class="metric-chip">
                <div class="val">{emi_display}</div>
                <div class="lbl">Est. EMI/mo</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            risk_label = "Low" if probability >= 0.7 else ("Medium" if probability >= 0.4 else "High")
            risk_color = "#1D9E75" if probability >= 0.7 else ("#EF9F27" if probability >= 0.4 else "#D85A30")
            st.markdown(f"""
            <div class="metric-chip">
                <div class="val" style="color:{risk_color};">{risk_label}</div>
                <div class="lbl">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)

        # Approval probability gauge
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(probability * 100, 1),
            number={"suffix": "%", "font": {"size": 26}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
                "bar": {"color": "#534AB7"},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#FAECE7"},
                    {"range": [40, 70], "color": "#FAEEDA"},
                    {"range": [70, 100], "color": "#E1F5EE"},
                ],
                "threshold": {
                    "line": {"color": "#534AB7", "width": 3},
                    "thickness": 0.75,
                    "value": probability * 100,
                },
            },
            title={"text": "Approval Probability", "font": {"size": 14, "color": "#888"}},
        ))
        fig.update_layout(
            height=220, margin=dict(t=30, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "DM Sans"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature influence radar
        st.markdown("#### 📊 Key Factors")
        categories = ["Income", "Credit\nHistory", "Loan\nAmount", "Employment", "Property\nArea"]

        income_score = min(applicant_income / 100000, 1.0)
        credit_score = credit_history
        loan_score = max(0, 1 - (loan_amount / 500))
        emp_score = min(employment_years / 10, 1.0) if 'employment_years' in dir() else 0.5
        area_score = {"Urban": 0.8, "Semiurban": 0.9, "Rural": 0.6}.get(property_area, 0.7)

        values = [income_score, credit_score, loan_score, emp_score, area_score]
        values_pct = [round(v * 100) for v in values]

        fig2 = go.Figure(go.Scatterpolar(
            r=values_pct,
            theta=categories,
            fill='toself',
            fillcolor='rgba(83, 74, 183, 0.15)',
            line=dict(color='#534AB7', width=2),
            marker=dict(size=6, color='#534AB7'),
        ))
        fig2.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
                angularaxis=dict(tickfont=dict(size=11)),
            ),
            showlegend=False,
            height=260,
            margin=dict(t=20, b=20, l=40, r=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "DM Sans"},
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Tips
        st.markdown("#### 💡 Insights")
        tips = []
        if credit_history == 0.0:
            tips.append("🔴 Improving your credit score is the single biggest lever.")
        if applicant_income < 30000:
            tips.append("🟡 Higher income significantly boosts approval odds.")
        if loan_amount > 300:
            tips.append("🟡 Consider a lower loan amount to improve your ratio.")
        if not tips:
            tips.append("🟢 Your profile looks strong — maintain your credit health.")

        for tip in tips:
            st.markdown(f"<div class='tip-box'>{tip}</div>", unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center; padding:4rem 1rem; color:#aaa;">
            <div style="font-size:3rem; margin-bottom:1rem;">🏦</div>
            <div style="font-size:1rem; font-weight:500; color:#666;">Fill in the form and click<br><b>Predict Loan Status</b></div>
            <div style="font-size:0.82rem; margin-top:0.8rem;">Results appear here</div>
        </div>
        """, unsafe_allow_html=True)

# ── Historical Trend (if enough history) ─────────────────────────────────────
if len(st.session_state.history) >= 3:
    st.markdown("---")
    st.markdown("### 📈 Session History")
    df = pd.DataFrame(st.session_state.history)
    df["index"] = range(1, len(df) + 1)
    df["approved"] = (df["result"] == "Approved").astype(int)

    fig3 = px.line(df, x="index", y="conf", color="result",
                   color_discrete_map={"Approved": "#1D9E75", "Rejected": "#D85A30"},
                   markers=True, labels={"index": "Application #", "conf": "Confidence", "result": "Decision"})
    fig3.update_layout(
        height=200, margin=dict(t=10, b=30, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font={"family": "DM Sans"},
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(tickformat=".0%", showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig3, use_container_width=True)