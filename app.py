import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF
import base64
import joblib
import os
import gdown

# --------------------------------
# ✅ Download & Load model and features from Google Drive
# --------------------------------
@st.cache_resource
def load_model_and_features():
    model_file = "credit_model.pkl"
    features_file = "feature_names.pkl"

    model_file_id = "1Uh_kgkaIVeRlZsvp6oHMW2_cKZZlosbu"
    features_file_id = "12XJQ4vl1zf68Ou2EQg5far77M5NAzkx1"

    if not os.path.exists(model_file):
        gdown.download(f"https://drive.google.com/uc?id={model_file_id}", model_file, quiet=False)

    if not os.path.exists(features_file):
        gdown.download(f"https://drive.google.com/uc?id={features_file_id}", features_file, quiet=False)

    model = joblib.load(model_file)
    feature_names = joblib.load(features_file)
    return model, feature_names

model, feature_names = load_model_and_features()

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Predict", "🧠 Model Explanation", "📘 Disclaimer"])

# --------------------------------
# 🏠 Home Page
# --------------------------------
if page == "🏠 Home":
    st.title("💳 Credit Default Prediction App")
    st.markdown("""
    Welcome to the Credit Default Risk Assessment Tool.  
    This app estimates the probability that a borrower may default on credit.

    ### 🔍 What’s included?
    - A fairness-aware machine learning model (Random Forest)
    - Trained on 10,000 real + synthetic borrower profiles
    - Excludes sensitive variables like gender and marital status

    ⚠️ Use responsibly. This is not a substitute for expert financial advice.
    """)

# --------------------------------
# 🔮 Prediction Page
# --------------------------------
elif page == "🔮 Predict":
    st.title("🔮 Predict Credit Default Risk")

    with st.form("prediction_form"):
        st.markdown("### 📥 Enter Borrower Information")

        INCOME = st.number_input("INCOME", min_value=0)
        SAVINGS = st.number_input("SAVINGS", min_value=0)
        DEBT = st.number_input("DEBT", min_value=0)

        education = st.selectbox("Education Level", [
            "No formal education", "Primary", "Secondary", "High School", "Diploma",
            "Bachelor's", "Master's", "PhD"
        ])
        occupation = st.selectbox("Occupation", [
            "Unemployed", "Student", "Agriculture", "Manual labor", "Sales", "Clerical",
            "Skilled Trade", "Health Care", "Education", "Engineering/Tech", "Managerial",
            "Professional Services", "Self-employed", "Retired", "Other"
        ])
        relationship = st.selectbox("Household Role", [
            "Single", "Married", "Divorced", "Widowed", "Supporting dependents", "Living with family"
        ])

        threshold = st.slider("Set Risk Threshold", 0.0, 1.0, 0.4, 0.01)
        submitted = st.form_submit_button("🔮 Predict Default Risk")

    if submitted:
        if INCOME == 0 and SAVINGS == 0 and DEBT == 0:
            st.error("❌ Please enter valid borrower financial details before predicting.")
        else:
            if INCOME == 0 or SAVINGS == 0 or DEBT == 0:
                st.warning("⚠️ Some inputs are set to 0. This may lead to unrealistic predictions.")

            # Feature Engineering
            R_DEBT_INCOME = DEBT / INCOME if INCOME > 0 else 0
            R_DEBT_SAVINGS = DEBT / SAVINGS if SAVINGS > 0 else 0
            CAT_DEBT = 1 if DEBT > 0 else 0
            CAT_SAVINGS_ACCOUNT = 1 if SAVINGS > 0 else 0

            input_dict = {
                'INCOME': INCOME,
                'SAVINGS': SAVINGS,
                'DEBT': DEBT,
                'R_DEBT_INCOME': R_DEBT_INCOME,
                'R_DEBT_SAVINGS': R_DEBT_SAVINGS,
                'CAT_DEBT': CAT_DEBT,
                'CAT_SAVINGS_ACCOUNT': CAT_SAVINGS_ACCOUNT,
                f'education_{education}': 1,
                f'occupation_{occupation}': 1,
                f'relationship_{relationship}': 1
            }

            # Fill missing columns with 0
            full_input = pd.DataFrame([{col: input_dict.get(col, 0) for col in feature_names}])
            prob = model.predict_proba(full_input)[0][1]
            pred = 1 if prob >= threshold else 0

            # Risk Band Classification
            risk_band = "Low Risk" if prob < 0.3 else "Medium Risk" if prob < 0.6 else "High Risk"

            st.subheader("📊 Prediction Result")
            st.write(f"**Predicted Probability:** {prob:.2%}")
            st.write(f"**Classification:** {'⚠️ High Risk' if pred else '✅ Low Risk'}")
            st.write(f"**Risk Level:** `{risk_band}`")

            if pred == 1:
                st.error("⚠️ High Risk: This borrower is likely to default.")
            else:
                st.success("✅ Low Risk: This borrower is unlikely to default.")

            # ----------------------------------
            # 📄 PDF Report
            # ----------------------------------
            def build_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Credit Default Risk Report", ln=True, align='C')
                pdf.ln(5)
                pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                pdf.ln(5)

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Borrower Information:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, f"""
INCOME: {INCOME}
SAVINGS: {SAVINGS}
DEBT: {DEBT}
Debt-to-Income Ratio: {R_DEBT_INCOME:.2f}
Debt-to-Savings Ratio: {R_DEBT_SAVINGS:.2f}
Education: {education}
Occupation: {occupation}
Relationship: {relationship}
""")

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Prediction Summary:", ln=True)
                pdf.set_font("Arial", size=12)
                pdf.multi_cell(0, 10, f"""
Predicted Risk Probability: {prob:.2%}
Threshold Used: {threshold}
Final Classification: {'High Risk' if pred else 'Low Risk'}
Risk Band: {risk_band}
""")

                pdf.set_y(-30)
                pdf.set_font("Arial", size=8)
                pdf.multi_cell(0, 5, "Disclaimer: This prediction is based on a statistical model and does not constitute financial advice.")
                return pdf

            pdf = build_pdf()
            pdf.output("borrower_report.pdf")

            with open("borrower_report.pdf", "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="borrower_report.pdf">📄 Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)

# --------------------------------
# 🧠 Model Explanation Page
# --------------------------------
elif page == "🧠 Model Explanation":
    st.title("🧠 How the Model Works")
    st.markdown("""
This app uses a **Random Forest Classifier**, trained on 10,000 borrower records (both real and synthetic).

### 💡 Features Used:
- Income, Savings, Debt
- Debt-to-Income Ratio
- Debt-to-Savings Ratio
- Whether the borrower has any debt or savings
- Education, Occupation, and Household Role (encoded)

### ✅ Fairness by Design:
We **excluded** sensitive variables like:
- Gender
- Marital Status

### 🔍 Evaluation Metrics:
- **Accuracy** – general performance
- **F1-score** – balance of precision and recall
- **AUC-ROC** – distinguishing defaulters from non-defaulters

This helps the model stay reliable across new data while minimizing discrimination risk.
""")

# --------------------------------
# 📘 Disclaimer Page
# --------------------------------
elif page == "📘 Disclaimer":
    st.title("📘 Disclaimer & Contact")
    st.markdown("""
This tool is built for **academic and demonstration purposes only**.

### ⚠️ Disclaimer:
- The model is **probabilistic**, not deterministic.
- Do **not** use this tool as the sole basis for credit decisions.

### 📫 Contact:
For questions or feedback, reach out to: `regina.gathimba@strathmore.edu`
""")
