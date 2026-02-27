import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from datetime import datetime
import pathlib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: black;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .prediction-success {
        background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    .prediction-warning {
        background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-header">
    <h1>📊 Customer Churn Prediction System</h1>
    <p>Advanced ML-powered analytics to predict customer behavior and reduce churn</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    base_path = pathlib.Path(__file__).parent  # path to Churn_Prediction/
    try:
        model = joblib.load(base_path / "model.pkl")
        scaler = joblib.load(base_path / "scaler.pkl")
        feature_names = joblib.load(base_path / "feature_names.pkl")
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"⚠️ Model file not found: {e}")
        return None, None, None


model, scaler, feature_names = load_models()

if model is None:
    st.stop()


with st.sidebar:
    st.markdown("### 📝 Customer Information")
    st.markdown("Fill in the customer details below:")
    

    st.markdown("#### 👤 Personal Details")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    dependents = st.selectbox("Has Dependents?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    

    st.markdown("#### 💳 Account Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12, help="Number of months the customer has been with the company")
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0, step=50.0)
    paperless_billing = st.selectbox("Paperless Billing?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    

    st.markdown("#### 📋 Contract & Payment")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"
    ])
    

    st.markdown("#### 🌐 Services")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    col1, col2 = st.columns(2)
    with col1:
        techsupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        onlinebackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        dev_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        streamingtv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    
    with col2:
        streamingmovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        onlinesecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        multiplelines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    predict_button = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)


if predict_button:
    
    data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'PaperlessBilling': paperless_billing,
        'MultipleLines_No phone service': 1 if multiplelines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiplelines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
        'InternetService_No': 1 if internet == "No" else 0,
        'OnlineSecurity_No internet service': 1 if onlinesecurity == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if onlinesecurity == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if onlinebackup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if onlinebackup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if dev_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if dev_protection == "Yes" else 0,
        'TechSupport_No internet service': 1 if techsupport == "No internet service" else 0,
        'TechSupport_Yes': 1 if techsupport == "Yes" else 0,
        'StreamingTV_No internet service': 1 if streamingtv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streamingtv == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if streamingmovies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streamingmovies == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0,
    }
    
    input_df = pd.DataFrame([data])
    

    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    

    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    

    input_df = input_df[feature_names]
    

    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    

    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.markdown("### 📊 Prediction Results")
        
      
        if proba < 0.3:
            risk_level = "Low"
            risk_color = "#28a745"
            risk_icon = "🟢"
        elif proba < 0.7:
            risk_level = "Medium"
            risk_color = "#ffc107"
            risk_icon = "🟡"
        else:
            risk_level = "High"
            risk_color = "#dc3545"
            risk_icon = "🔴"
        
       
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-warning">
                {risk_icon} HIGH CHURN RISK DETECTED!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-success">
                ✅ LOW CHURN RISK - Customer likely to stay
            </div>
            """, unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Churn Probability", f"{proba:.1%}", delta=f"{proba-0.5:.1%}")
        with col_b:
            st.metric("Risk Level", risk_level, delta=None)
        with col_c:
            st.metric("Confidence", f"{max(proba, 1-proba):.1%}", delta=None)
    
    with col2:

        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = proba * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk %"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': "#d4edda"},
                    {'range': [30, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#f8d7da"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col3:
 
        st.markdown("### 👤 Customer Summary")
        st.markdown(f"""
        - **Gender:** {gender}
        - **Senior Citizen:** {'Yes' if senior else 'No'}
        - **Tenure:** {tenure} months
        - **Monthly Charges:** ${monthly_charges:.2f}
        - **Total Charges:** ${total_charges:.2f}
        - **Contract:** {contract}
        - **Internet Service:** {internet}
        """)
    

    st.markdown("### 🔍 Feature Importance Analysis")
    
    try:
        explainer = shap.Explainer(model, input_df)
        shap_values = explainer(input_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Waterfall Plot")
            fig_waterfall, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig_waterfall)
        
        with col2:
            st.markdown("#### Feature Contributions")
            

            feature_contributions = pd.DataFrame({
                'Feature': feature_names,
                'Contribution': shap_values.values[0]
            }).sort_values('Contribution', key=abs, ascending=False)
            

            top_features = feature_contributions.head(10)
            
            fig_bar = px.bar(
                top_features,
                x='Contribution',
                y='Feature',
                orientation='h',
                color='Contribution',
                color_continuous_scale='RdYlBu_r',
                title="Top 10 Feature Contributions"
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {str(e)}")
    

    st.markdown("### 💡 Recommendations")
    
    if prediction == 1:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 1rem; border-radius: 5px; border-left: 4px solid #ffc107;color:black;">
        <h4>🚨 Retention Strategy Required</h4>
        <ul>
            <li>Immediate customer outreach recommended</li>
            <li>Consider offering contract extension incentives</li>
            <li>Review service satisfaction and address pain points</li>
            <li>Evaluate competitive pricing options</li>
            <li>Enhance customer support engagement</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #d4edda; padding: 1rem; border-radius: 5px; border-left: 4px solid #28a745; color:green;">
        <h4>✅ Customer Retention Strategies</h4>
        <ul>
            <li>Continue providing excellent service</li>
            <li>Consider upselling additional services</li>
            <li>Maintain regular satisfaction surveys</li>
            <li>Reward loyalty with special offers</li>
            <li>Monitor for any changes in usage patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

else:

    st.markdown("### 🎯 Welcome to Customer Churn Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🔮 Predictive Analytics</h4>
            <p>Advanced machine learning models to predict customer churn probability with high accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>📊 Feature Analysis</h4>
            <p>SHAP-powered explanations showing which factors contribute most to churn risk.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>💡 Actionable Insights</h4>
            <p>Data-driven recommendations to improve customer retention and reduce churn.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🚀 How to Use")
    st.markdown("""
    1. **Fill in customer details** in the sidebar
    2. **Click 'Predict Churn Risk'** to analyze
    3. **Review the results** and feature importance
    4. **Implement recommendations** to improve retention
    """)
    

    st.markdown("### 📈 Key Insights")
    

    sample_data = pd.DataFrame({
        'Risk Level': ['Low', 'Medium', 'High'],
        'Percentage': [60, 25, 15],
        'Color': ['#28a745', '#ffc107', '#dc3545']
    })
    
    fig_pie = px.pie(sample_data, values='Percentage', names='Risk Level', 
                     title="Typical Customer Risk Distribution",
                     color_discrete_sequence=['#28a745', '#ffc107', '#dc3545'])
    st.plotly_chart(fig_pie, use_container_width=True)


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
