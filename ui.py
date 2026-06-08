import streamlit as st
import joblib
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Retention Engine",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Telecom Customer Churn Predictor")
st.markdown("Enter customer details below to calculate real-time churn liability scores.")

# 2. Load the trained pipeline artifact
@st.cache_resource
def load_pipeline():
    return joblib.load("models/churn_pipeline.joblib")

try:
    pipeline = load_pipeline()
except Exception:
    st.error("⚠️ Model artifact not found! Please run 'python main.py' first to train the model.")
    st.stop()

# 3. Build Form Layout for User Inputs
with st.form("churn_form"):
    st.subheader("Customer Demographics & Profile")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (Years)", 18, 100, 35)
        tenure = st.slider("Tenure (Months with company)", 1, 72, 12)
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=250.0, value=75.0, step=0.5)
        paperless = st.selectbox("Paperless Billing Active?", ["Yes", "No"])
        
    st.markdown("---")
    st.subheader("Account & Billing Specifications")
    
    col3, col4 = st.columns(2)
    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    with col4:
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        
    submit = st.form_submit_button("Run Risk Assessment")

# 4. Trigger Model Inference Logic
if submit:
    # Structure inputs into a matching DataFrame format
    input_data = pd.DataFrame([{
        "Age": age,
        "Tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "Contract": contract,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless
    }])
    
    # Calculate inference scores
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]
    
    # Display results cleanly
    st.markdown("---")
    st.subheader("Model Assessment Results")
    
    if prediction == 1:
        st.error(f"🚨 **High Churn Risk Detected!**")
        st.metric(label="Calculated Risk Probability", value=f"{probability:.2%}")
        st.warning("Recommendation: Proactively reach out with custom loyalty promotions.")
    else:
        st.success(f"✅ **Low Risk Customer**")
        st.metric(label="Calculated Risk Probability", value=f"{probability:.2%}")
        st.info("Recommendation: Maintain standard service workflows.")