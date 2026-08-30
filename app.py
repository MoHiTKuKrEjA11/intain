import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Loan Performance Intelligence Engine", layout="wide")

st.title("Loan Performance Intelligence Engine")
st.caption("Intain Campus FinTech Challenge 2026 — AI Track — Live Prototype Demo")

# ============================================================
# CONFIG
# ============================================================
categorical_cols = ['credit_score_band', 'ltv_band', 'dti_band', 'state',
                     'loan_purpose', 'occupancy_type', 'property_type',
                     'servicer_name', 'current_status', 'source_system',
                     'document_status']
feature_cols = [
    'loan_age_months', 'remaining_term_months', 'original_balance',
    'current_balance', 'interest_rate', 'days_past_due',
    'modification_flag', 'prepayment_flag'
] + [c + '_enc' for c in categorical_cols]
target_col = 'next_12m_default_flag'
anomaly_features = ['original_balance', 'current_balance', 'interest_rate',
                     'days_past_due', 'loan_age_months']


@st.cache_resource
def train_model():
    """Train once, cache across the whole app session so it's fast."""
    df = pd.read_csv('data/sim_train.csv')
    df['current_balance'] = df['current_balance'].abs()
    df['interest_rate'] = df['interest_rate'].fillna(df['interest_rate'].median())

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    df['reporting_month'] = pd.to_datetime(df['reporting_month'])
    cutoff_date = df['reporting_month'].quantile(0.8)
    train_df = df[df['reporting_month'] <= cutoff_date]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    model.fit(train_df[feature_cols], train_df[target_col])

    iso = IsolationForest(contamination=0.03, random_state=42)
    iso.fit(df[anomaly_features])

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return model, encoders, iso, importances, df['interest_rate'].median()


def safe_encode(series, encoder):
    known_classes = set(encoder.classes_)
    return series.astype(str).apply(
        lambda x: encoder.transform([x])[0] if x in known_classes else -1
    )


model, encoders, iso, importances, median_rate = train_model()

# ============================================================
# SIDEBAR: data input
# ============================================================
st.sidebar.header("1. Load Data")
data_source = st.sidebar.radio("Choose data source:", ["Use sample test data", "Upload your own CSV"])

if data_source == "Upload your own CSV":
    uploaded = st.sidebar.file_uploader("Upload a loan CSV (same columns as the training data)", type="csv")
    if uploaded is None:
        st.info("Upload a CSV in the sidebar, or switch to 'Use sample test data' to see the demo.")
        st.stop()
    test_df = pd.read_csv(uploaded)
else:
    test_df = pd.read_csv('data/sim_test.csv')
    st.sidebar.success(f"Loaded {len(test_df)} sample records")

# ============================================================
# SCENARIO CONTROLS
# ============================================================
st.sidebar.header("2. Scenario Simulation")
rate_shift = st.sidebar.slider("Interest rate shift (percentage points)", -3.0, 3.0, 0.0, 0.1)
dpd_multiplier = st.sidebar.slider("Days-past-due multiplier (stress factor)", 0.5, 3.0, 1.0, 0.1)

# ============================================================
# RUN PREDICTIONS
# ============================================================
scored_df = test_df.copy()
scored_df['current_balance'] = scored_df['current_balance'].abs()
scored_df['interest_rate'] = scored_df['interest_rate'].fillna(median_rate)

# Apply scenario adjustments
scored_df['interest_rate'] = scored_df['interest_rate'] + rate_shift
scored_df['days_past_due'] = scored_df['days_past_due'] * dpd_multiplier

for col in categorical_cols:
    scored_df[col + '_enc'] = safe_encode(scored_df[col], encoders[col])

scored_df['predicted_default_prob'] = model.predict_proba(scored_df[feature_cols])[:, 1]
scored_df['anomaly_flag'] = iso.predict(scored_df[anomaly_features])
scored_df['anomaly_flag'] = scored_df['anomaly_flag'].map({1: 0, -1: 1})
scored_df['action'] = np.where(scored_df['predicted_default_prob'] > 0.5, 'Review', 'Monitor')
scored_df['confidence'] = np.where(scored_df['predicted_default_prob'] > 0.7, 'High',
                            np.where(scored_df['predicted_default_prob'] > 0.3, 'Medium', 'Low'))

# ============================================================
# MAIN DASHBOARD
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Loans Scored", len(scored_df))
col2.metric("Avg. Predicted Default Risk", f"{scored_df['predicted_default_prob'].mean()*100:.1f}%")
col3.metric("Flagged for Review", int((scored_df['action'] == 'Review').sum()))
col4.metric("Anomalies Detected", int(scored_df['anomaly_flag'].sum()))

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Explainability", "Anomalies", "AI Reviewer Note"])

with tab1:
    st.subheader("Loan-Level Predictions")
    display_cols = ['loan_id', 'predicted_default_prob', 'action', 'confidence', 'anomaly_flag']
    st.dataframe(
        scored_df[display_cols].sort_values('predicted_default_prob', ascending=False),
        use_container_width=True
    )
    st.download_button(
        "Download predictions as CSV",
        scored_df[display_cols].to_csv(index=False),
        file_name="submission.csv"
    )

with tab2:
    st.subheader("What Drives These Predictions (Model Explainability)")
    st.bar_chart(importances.head(10))
    st.caption("Top 10 features by importance in the trained Random Forest model.")

with tab3:
    st.subheader("Anomalous Records")
    anomalies = scored_df[scored_df['anomaly_flag'] == 1]
    st.write(f"{len(anomalies)} anomalous records found out of {len(scored_df)} ({len(anomalies)/len(scored_df)*100:.1f}%)")
    st.dataframe(
        anomalies[['loan_id', 'current_balance', 'interest_rate', 'days_past_due', 'predicted_default_prob']],
        use_container_width=True
    )

with tab4:
    st.subheader("LLM-Generated Reviewer Note")
    st.caption("Select a loan to generate a plain-English reviewer explanation using Google Gemini (free tier). The LLM only explains the model's output — it never makes the prediction itself.")

    selected_loan = st.selectbox("Choose a loan to review:", scored_df.sort_values('predicted_default_prob', ascending=False)['loan_id'].head(20))
    api_key_input = st.text_input("Gemini API key (get one free at aistudio.google.com):", type="password")

    if st.button("Generate Reviewer Note"):
        if not api_key_input:
            st.error("Please enter a Gemini API key above.")
        else:
            row = scored_df[scored_df['loan_id'] == selected_loan].iloc[0]
            top_driver = importances.index[0]
            prompt = (
                f"You are a loan reviewer assistant. A model flagged loan {row['loan_id']} "
                f"with a predicted default probability of {row['predicted_default_prob']:.2f} "
                f"(confidence: {row['confidence']}). "
                f"The top model driver was '{top_driver}'. "
                f"Anomaly flag: {'yes' if row['anomaly_flag']==1 else 'no'}. "
                f"Write a 2-sentence plain-English note for a human reviewer explaining "
                f"why this loan needs attention, based only on the numbers given. "
                f"Do not invent facts not provided."
            )
            try:
                from google import genai
                client = genai.Client(api_key=api_key_input)
                response = client.models.generate_content(model='gemini-3.6-flash', contents=prompt)
                st.success(response.text)
            except Exception as e:
                st.error(f"Error calling Gemini API: {e}")

st.divider()
st.caption("Loan Performance Intelligence Engine — Prototype for Intain Campus FinTech Challenge 2026, AI Track")