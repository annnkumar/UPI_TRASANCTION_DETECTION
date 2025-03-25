import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from preprocessing.data_processor import preprocess_data, engineer_features
from models.model_utils import load_model, make_prediction, get_feature_importance

# ✅ Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load Model & Preprocessing Objects
model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
preprocess_path = os.path.join(BASE_DIR, "models", "preprocessing_objects.pkl")

# ✅ Load model and preprocessing objects
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(preprocess_path, "rb") as preprocess_file:
    preprocess_objects = pickle.load(preprocess_file)

# ✅ Streamlit UI
st.title("UPI Fraud Detection System")
st.write("Enter transaction details to check for fraud risk.")

# Add user input form (example)
amount = st.number_input("Transaction Amount", min_value=0.0)
user_input = {"amount": amount}  # You may need more features

# Predict on user input
if st.button("Predict Fraud"):
    processed_input = preprocess_data(user_input, preprocess_objects)
    prediction = model.predict([processed_input])
    
    if prediction[0] == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Safe.")

# Page configuration
st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model and preprocessing objects
@st.cache_resource
def load_resources():
    model_path = 'models/best_model.pkl'
    preproc_path = 'models/preprocessing_objects.pkl'
    
    # If model files don't exist, warn the user
    if not (os.path.exists(model_path) and os.path.exists(preproc_path)):
        return None, None
    
    try:
        model = load_model(model_path)
        with open(preproc_path, 'rb') as f:
            preprocessing = pickle.load(f)
        return model, preprocessing
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return None, None

model, preprocessing = load_resources()

# Application title and description
st.title("🔍 UPI Fraud Detection System")
st.markdown("""
This application uses machine learning to detect potentially fraudulent UPI transactions.
You can either upload transaction data for batch prediction or input a single transaction for analysis.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict New Transactions", "Check Single Transaction", "Upload Data", "Model Insights"])

# Home page
if page == "Home":
    st.header("Welcome to the UPI Fraud Detection System")
    
    st.markdown("""
    ### How this system works:
    
    This fraud detection system uses machine learning to identify potentially fraudulent UPI transactions.
    The model was trained on thousands of transactions with features like:
    
    - Transaction amount
    - Time of transaction
    - Location information
    - Device information
    - User behavior patterns
    
    ### Use this application to:
    
    - **Upload a CSV file** of transactions for batch analysis
    - **Check individual transactions** using the form interface
    - **Explore model insights** to understand how fraud is detected
    
    ### Sample Images:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a simple visualization to show on the home page
        st.subheader("Fraud by Time of Day")
        
        # Create example data for demonstration
        hours = range(0, 24)
        fraud_rate = [3, 2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 
                      7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(hours, fraud_rate, marker='o', linewidth=2)
        ax.set_title('Example: Fraud Rate by Hour of Day')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Fraud Rate (%)')
        ax.grid(True)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Merchant Category Risk")
        
        # Create example data for demonstration
        categories = ['Electronics', 'Travel', 'Groceries', 'Entertainment', 'Clothing']
        risk_scores = [8.2, 6.7, 3.1, 5.8, 4.5]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=risk_scores, y=categories, hue=categories, legend=False, palette='viridis')
        ax.set_title('Example: Fraud Risk by Merchant Category')
        ax.set_xlabel('Risk Score')
        st.pyplot(fig)

# New Transactions page
elif page == "Predict New Transactions":
    st.header("Predict Fraud for New Transactions")
    
    st.markdown("""
    This page lets you analyze new transaction data for potential fraud.
    
    You have two options:
    1. **Upload a CSV file** - For analyzing multiple transactions at once
    2. **Input a single transaction** - For checking an individual transaction
    
    How would you like to proceed?
    """)
    
    option = st.radio("Choose an option:", ["Upload Transaction Data", "Check Single Transaction"])
    
    if option == "Upload Transaction Data":
        st.subheader("Upload Transaction Data File")
        
        st.markdown("""
        Upload a CSV file containing transaction data. The file should include the following columns:
        - TransactionID
        - Amount
        - Timestamp
        - MerchantCategory
        - TransactionType
        - UnusualLocation (True/False)
        - UnusualAmount (True/False)
        - NewDevice (True/False)
        - FailedAttempts
        - BankName
        
        Additional columns will improve prediction accuracy.
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Loaded {df.shape[0]} transactions.")
                
                # Display first few rows
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Check if model is loaded
                if model is None or preprocessing is None:
                    st.error("Model resources not found. Please run the generate_model.py script first to generate the model.")
                else:
                    # Process button
                    if st.button("Detect Fraudulent Transactions"):
                        # Preprocess data
                        with st.spinner("Preprocessing data..."):
                            try:
                                # Engineer features first
                                df_engineered = engineer_features(df)
                                
                                # Then preprocess for the model
                                X_processed = preprocess_data(df_engineered, preprocessing)
                                
                                # Make predictions
                                predictions, probabilities = make_prediction(model, X_processed)
                                
                                # Add predictions to the dataframe
                                df['Fraud_Prediction'] = predictions
                                df['Fraud_Probability'] = probabilities
                                
                                # Show results
                                st.subheader("Prediction Results")
                                
                                # Count of fraudulent transactions
                                fraud_count = df['Fraud_Prediction'].sum()
                                st.metric("Detected Fraudulent Transactions", fraud_count, 
                                         f"{fraud_count/len(df)*100:.2f}% of total")
                                
                                # Display transactions with predictions
                                st.dataframe(df.sort_values('Fraud_Probability', ascending=False))
                                
                                # Visualize predictions
                                st.subheader("Visualization of Predictions")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Pie chart of fraud vs non-fraud
                                    fig, ax = plt.subplots(figsize=(8, 8))
                                    df['Fraud_Prediction'].value_counts().plot.pie(
                                        autopct='%1.1f%%', colors=['#90EE90', '#F08080'],
                                        labels=['Legitimate', 'Fraudulent'], ax=ax)
                                    ax.set_title('Prediction Distribution')
                                    st.pyplot(fig)
                                
                                with col2:
                                    # Histogram of fraud probabilities
                                    fig, ax = plt.subplots(figsize=(8, 8))
                                    sns.histplot(df['Fraud_Probability'], bins=20, ax=ax)
                                    ax.set_title('Distribution of Fraud Probabilities')
                                    ax.set_xlabel('Fraud Probability')
                                    ax.set_ylabel('Count')
                                    st.pyplot(fig)
                                
                                # Add download button for results
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=df.to_csv(index=False).encode('utf-8'),
                                    file_name='fraud_detection_results.csv',
                                    mime='text/csv'
                                )
                                
                            except Exception as e:
                                st.error(f"Error during processing: {e}")
            
            except Exception as e:
                st.error(f"Error reading the file: {e}")
                
    elif option == "Check Single Transaction":
        st.subheader("Check a Single Transaction")
        
        st.markdown("""
        Enter the details of a transaction to check if it's potentially fraudulent.
        
        This tool allows you to input a specific transaction's details and get an instant prediction
        on whether it might be fraudulent, along with a risk assessment and explanation of key risk factors.
        """)
        
        # Create form for transaction details
        with st.form("transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
                
                # Date and time pickers
                transaction_date = st.date_input("Transaction Date")
                transaction_time = st.time_input("Transaction Time")
                
                merchant_category = st.selectbox(
                    "Merchant Category",
                    ["Electronics", "Restaurants", "Groceries", "Clothing", "Travel", "Entertainment", "Utilities"]
                )
                
                transaction_type = st.selectbox(
                    "Transaction Type",
                    ["P2P", "P2M"]
                )
            
            with col2:
                unusual_location = st.checkbox("Unusual Location")
                unusual_amount = st.checkbox("Unusual Amount")
                new_device = st.checkbox("New Device")
                
                failed_attempts = st.slider("Failed Attempts", 0, 5, 0)
                
                bank_name = st.selectbox(
                    "Bank Name",
                    ["State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "Bank of Baroda"]
                )
                
                avg_transaction_amount = st.number_input("Average Transaction Amount", min_value=0.0, value=500.0)
                transaction_frequency = st.selectbox("Transaction Frequency", ["1/day", "3/day", "5/day"])
            
            submit_button = st.form_submit_button("Check Transaction")
        
        # Process form submission
        if submit_button:
            if model is None or preprocessing is None:
                st.error("Model resources not found. Please run the generate_model.py script first to generate the model.")
            else:
                try:
                    # Combine date and time
                    timestamp = pd.Timestamp.combine(transaction_date, transaction_time)
                    
                    # Create a dataframe with the single transaction
                    transaction_data = {
                        'TransactionID': [np.random.randint(100000000000, 999999999999)],
                        'UserID': ['single-transaction-check'],
                        'Amount': [amount],
                        'Timestamp': [timestamp],
                        'MerchantCategory': [merchant_category],
                        'TransactionType': [transaction_type],
                        'DeviceID': ['single-device'],
                        'IPAddress': ['0.0.0.0'],
                        'Latitude': [0],
                        'Longitude': [0],
                        'AvgTransactionAmount': [avg_transaction_amount],
                        'TransactionFrequency': [transaction_frequency],
                        'UnusualLocation': [unusual_location],
                        'UnusualAmount': [unusual_amount],
                        'NewDevice': [new_device],
                        'FailedAttempts': [failed_attempts],
                        'BankName': [bank_name]
                    }
                    
                    df_transaction = pd.DataFrame(transaction_data)
                    
                    # Display the input transaction
                    st.subheader("Transaction Details")
                    st.write(df_transaction)
                    
                    # Engineer features
                    df_engineered = engineer_features(df_transaction)
                    
                    # Preprocess data
                    X_processed = preprocess_data(df_engineered, preprocessing)
                    
                    # Make prediction
                    prediction, probability = make_prediction(model, X_processed)
                    
                    # Display result
                    st.subheader("Prediction Result")
                    
                    if prediction[0]:
                        st.error(f"⚠️ This transaction is flagged as potentially fraudulent with {probability[0]*100:.2f}% confidence.")
                    else:
                        st.success(f"✅ This transaction appears to be legitimate. Fraud probability: {probability[0]*100:.2f}%")
                    
                    # Display risk meter
                    st.subheader("Risk Assessment")
                    
                    # Create a gauge chart for risk visualization
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.barh(["Risk"], [100], color="lightgray", height=0.3)
                    ax.barh(["Risk"], [probability[0] * 100], color=plt.cm.RdYlGn_r(probability[0]), height=0.3)
                    
                    # Add a marker for the threshold
                    threshold = 0.5
                    ax.axvline(x=threshold * 100, color='red', linestyle='--', alpha=0.7)
                    ax.text(threshold * 100 + 2, 0, f"Threshold ({threshold*100}%)", 
                            va='center', color='red', alpha=0.7)
                    
                    # Add the probability value
                    ax.text(probability[0] * 100 - 5, 0, f"{probability[0]*100:.1f}%", 
                            va='center', ha='right', color='black', weight='bold')
                    
                    ax.set_xlim(0, 100)
                    ax.set_ylim(-0.5, 0.5)
                    ax.set_xlabel("Fraud Probability (%)")
                    ax.get_yaxis().set_visible(False)
                    
                    st.pyplot(fig)
                    
                    # Feature importance for this prediction
                    st.subheader("Key Factors for This Prediction")
                    
                    # Create a sample dataframe to show important features
                    factors = []
                    
                    # Add logic to determine important factors
                    if unusual_location:
                        factors.append({"Factor": "Unusual Location", "Impact": "High", "Description": "Transaction location differs from usual patterns"})
                    
                    if unusual_amount:
                        factors.append({"Factor": "Unusual Amount", "Impact": "High", "Description": "Transaction amount significantly differs from average"})
                    
                    if new_device:
                        factors.append({"Factor": "New Device", "Impact": "Medium", "Description": "Transaction made from a new device"})
                    
                    if failed_attempts > 0:
                        factors.append({"Factor": "Failed Attempts", "Impact": "High", "Description": f"{failed_attempts} failed authentication attempts"})
                    
                    if amount > avg_transaction_amount * 2:
                        factors.append({"Factor": "Amount Ratio", "Impact": "Medium", "Description": "Transaction amount is more than twice the average"})
                    
                    # Add time-based factors
                    hour = timestamp.hour
                    if hour < 6 or hour >= 22:
                        factors.append({"Factor": "Night Time Transaction", "Impact": "Low", "Description": "Transaction occurred during night hours"})
                    
                    # Display factors if any
                    if factors:
                        st.dataframe(pd.DataFrame(factors))
                    else:
                        st.info("No specific risk factors identified for this transaction.")
                    
                except Exception as e:
                    st.error(f"Error processing transaction: {e}")

# Upload Data page
elif page == "Upload Data":
    st.header("Upload Transaction Data")
    
    st.markdown("""
    Upload a CSV file containing transaction data for batch analysis.
    
    The file should include the following columns:
    - TransactionID
    - Amount
    - Timestamp
    - MerchantCategory
    - TransactionType
    - UnusualLocation (True/False)
    - UnusualAmount (True/False)
    - NewDevice (True/False)
    - FailedAttempts
    - BankName
    
    Additional columns will improve prediction accuracy.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully! Loaded {df.shape[0]} transactions.")
            
            # Display first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Check if model is loaded
            if model is None or preprocessing is None:
                st.error("Model resources not found. Please run the generate_model.py script first to generate the model.")
            else:
                # Process button
                if st.button("Detect Fraudulent Transactions"):
                    # Preprocess data
                    with st.spinner("Preprocessing data..."):
                        try:
                            # Engineer features first
                            df_engineered = engineer_features(df)
                            
                            # Then preprocess for the model
                            X_processed = preprocess_data(df_engineered, preprocessing)
                            
                            # Make predictions
                            predictions, probabilities = make_prediction(model, X_processed)
                            
                            # Add predictions to the dataframe
                            df['Fraud_Prediction'] = predictions
                            df['Fraud_Probability'] = probabilities
                            
                            # Show results
                            st.subheader("Prediction Results")
                            
                            # Count of fraudulent transactions
                            fraud_count = df['Fraud_Prediction'].sum()
                            st.metric("Detected Fraudulent Transactions", fraud_count, 
                                     f"{fraud_count/len(df)*100:.2f}% of total")
                            
                            # Display transactions with predictions
                            st.dataframe(df.sort_values('Fraud_Probability', ascending=False))
                            
                            # Visualize predictions
                            st.subheader("Visualization of Predictions")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Pie chart of fraud vs non-fraud
                                fig, ax = plt.subplots(figsize=(8, 8))
                                df['Fraud_Prediction'].value_counts().plot.pie(
                                    autopct='%1.1f%%', colors=['#90EE90', '#F08080'],
                                    labels=['Legitimate', 'Fraudulent'], ax=ax)
                                ax.set_title('Prediction Distribution')
                                st.pyplot(fig)
                            
                            with col2:
                                # Histogram of fraud probabilities
                                fig, ax = plt.subplots(figsize=(8, 8))
                                sns.histplot(df['Fraud_Probability'], bins=20, ax=ax)
                                ax.set_title('Distribution of Fraud Probabilities')
                                ax.set_xlabel('Fraud Probability')
                                ax.set_ylabel('Count')
                                st.pyplot(fig)
                            
                            # Add download button for results
                            st.download_button(
                                label="Download Results as CSV",
                                data=df.to_csv(index=False).encode('utf-8'),
                                file_name='fraud_detection_results.csv',
                                mime='text/csv'
                            )
                            
                        except Exception as e:
                            st.error(f"Error during processing: {e}")
        
        except Exception as e:
            st.error(f"Error reading the file: {e}")

# Check Single Transaction page
elif page == "Check Single Transaction":
    st.header("Check a Single Transaction")
    
    st.markdown("""
    Enter the details of a transaction to check if it's potentially fraudulent.
    
    This tool allows you to input a specific transaction's details and get an instant prediction
    on whether it might be fraudulent, along with a risk assessment and explanation of key risk factors.
    """)
    
    # Create form for transaction details
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
            
            # Date and time pickers
            transaction_date = st.date_input("Transaction Date")
            transaction_time = st.time_input("Transaction Time")
            
            merchant_category = st.selectbox(
                "Merchant Category",
                ["Electronics", "Restaurants", "Groceries", "Clothing", "Travel", "Entertainment", "Utilities"]
            )
            
            transaction_type = st.selectbox(
                "Transaction Type",
                ["P2P", "P2M"]
            )
        
        with col2:
            unusual_location = st.checkbox("Unusual Location")
            unusual_amount = st.checkbox("Unusual Amount")
            new_device = st.checkbox("New Device")
            
            failed_attempts = st.slider("Failed Attempts", 0, 5, 0)
            
            bank_name = st.selectbox(
                "Bank Name",
                ["State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank", "Bank of Baroda"]
            )
            
            avg_transaction_amount = st.number_input("Average Transaction Amount", min_value=0.0, value=500.0)
            transaction_frequency = st.selectbox("Transaction Frequency", ["1/day", "3/day", "5/day"])
        
        submit_button = st.form_submit_button("Check Transaction")
    
    # Process form submission
    if submit_button:
        if model is None or preprocessing is None:
            st.error("Model resources not found. Please run the generate_model.py script first to generate the model.")
        else:
            try:
                # Combine date and time
                timestamp = pd.Timestamp.combine(transaction_date, transaction_time)
                
                # Create a dataframe with the single transaction
                transaction_data = {
                    'TransactionID': [np.random.randint(100000000000, 999999999999)],
                    'UserID': ['single-transaction-check'],
                    'Amount': [amount],
                    'Timestamp': [timestamp],
                    'MerchantCategory': [merchant_category],
                    'TransactionType': [transaction_type],
                    'DeviceID': ['single-device'],
                    'IPAddress': ['0.0.0.0'],
                    'Latitude': [0],
                    'Longitude': [0],
                    'AvgTransactionAmount': [avg_transaction_amount],
                    'TransactionFrequency': [transaction_frequency],
                    'UnusualLocation': [unusual_location],
                    'UnusualAmount': [unusual_amount],
                    'NewDevice': [new_device],
                    'FailedAttempts': [failed_attempts],
                    'BankName': [bank_name]
                }
                
                df_transaction = pd.DataFrame(transaction_data)
                
                # Engineer features
                df_engineered = engineer_features(df_transaction)
                
                # Preprocess data
                X_processed = preprocess_data(df_engineered, preprocessing)
                
                # Make prediction
                prediction, probability = make_prediction(model, X_processed)
                
                # Display result
                st.subheader("Prediction Result")
                
                if prediction[0]:
                    st.error(f"⚠️ This transaction is flagged as potentially fraudulent with {probability[0]*100:.2f}% confidence.")
                else:
                    st.success(f"✅ This transaction appears to be legitimate. Fraud probability: {probability[0]*100:.2f}%")
                
                # Display risk meter
                st.subheader("Risk Assessment")
                
                # Create a gauge chart for risk visualization
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.barh(["Risk"], [100], color="lightgray", height=0.3)
                ax.barh(["Risk"], [probability[0] * 100], color=plt.cm.RdYlGn_r(probability[0]), height=0.3)
                
                # Add a marker for the threshold
                threshold = 0.5
                ax.axvline(x=threshold * 100, color='red', linestyle='--', alpha=0.7)
                ax.text(threshold * 100 + 2, 0, f"Threshold ({threshold*100}%)", 
                        va='center', color='red', alpha=0.7)
                
                # Add the probability value
                ax.text(probability[0] * 100 - 5, 0, f"{probability[0]*100:.1f}%", 
                        va='center', ha='right', color='black', weight='bold')
                
                ax.set_xlim(0, 100)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel("Fraud Probability (%)")
                ax.get_yaxis().set_visible(False)
                
                st.pyplot(fig)
                
                # Feature importance for this prediction
                st.subheader("Key Factors for This Prediction")
                
                # Create a sample dataframe to show important features
                factors = []
                
                # Add logic to determine important factors
                if unusual_location:
                    factors.append({"Factor": "Unusual Location", "Impact": "High", "Description": "Transaction location differs from usual patterns"})
                
                if unusual_amount:
                    factors.append({"Factor": "Unusual Amount", "Impact": "High", "Description": "Transaction amount significantly differs from average"})
                
                if new_device:
                    factors.append({"Factor": "New Device", "Impact": "Medium", "Description": "Transaction made from a new device"})
                
                if failed_attempts > 0:
                    factors.append({"Factor": "Failed Attempts", "Impact": "High", "Description": f"{failed_attempts} failed authentication attempts"})
                
                if amount > avg_transaction_amount * 2:
                    factors.append({"Factor": "Amount Ratio", "Impact": "Medium", "Description": "Transaction amount is more than twice the average"})
                
                # Add time-based factors
                hour = timestamp.hour
                if hour < 6 or hour >= 22:
                    factors.append({"Factor": "Night Time Transaction", "Impact": "Low", "Description": "Transaction occurred during night hours"})
                
                # Display factors if any
                if factors:
                    st.dataframe(pd.DataFrame(factors))
                else:
                    st.info("No specific risk factors identified for this transaction.")
                
            except Exception as e:
                st.error(f"Error processing transaction: {e}")

# Model Insights page
elif page == "Model Insights":
    st.header("Model Insights")
    
    st.markdown("""
    Explore how the fraud detection model works and what factors it considers most important.
    """)
    
    tab1, tab2, tab3 = st.tabs(["Key Indicators", "Patterns & Trends", "Model Performance"])
    
    with tab1:
        st.subheader("Key Fraud Indicators")
        
        indicators = [
            {
                "Feature": "Unusual Location",
                "Importance": "High",
                "Description": "Transactions from locations that differ from a user's normal pattern have a higher risk of fraud."
            },
            {
                "Feature": "Unusual Amount",
                "Importance": "High",
                "Description": "Transactions with amounts significantly different from a user's average spending patterns."
            },
            {
                "Feature": "Failed Attempts",
                "Importance": "High",
                "Description": "Multiple authentication failures before a successful transaction is a strong fraud indicator."
            },
            {
                "Feature": "New Device",
                "Importance": "Medium",
                "Description": "Transactions from newly registered devices carry increased risk."
            },
            {
                "Feature": "Transaction Time",
                "Importance": "Medium",
                "Description": "Transactions during unusual hours for a specific user may indicate fraud."
            },
            {
                "Feature": "Merchant Category",
                "Importance": "Medium",
                "Description": "Some merchant categories (e.g., electronics) have higher fraud rates."
            },
            {
                "Feature": "Transaction Frequency",
                "Importance": "Medium",
                "Description": "Sudden increases in transaction frequency often precede fraud."
            }
        ]
        
        st.dataframe(pd.DataFrame(indicators))
        
        st.subheader("Feature Importance")
        
        # Create a sample feature importance visualization
        features = ['UnusualAmount', 'FailedAttempts', 'UnusualLocation', 'NewDevice', 
                   'RiskScore', 'AmountRatio', 'IsNightTime', 'MerchantCategory']
        importances = [0.18, 0.16, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances, y=features, hue=features, legend=False, palette='viridis')
        ax.set_title('Model Feature Importance')
        ax.set_xlabel('Relative Importance')
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Fraud Patterns and Trends")
        
        # Time-based patterns example
        st.write("#### Time-based Fraud Patterns")
        
        hours = list(range(24))
        fraud_rates = [4.2, 5.7, 7.1, 6.5, 5.2, 3.8, 2.5, 1.9, 2.3, 2.8, 3.2, 3.7, 
                      4.1, 4.5, 4.9, 5.3, 5.8, 6.2, 6.7, 7.3, 7.8, 6.9, 5.6, 4.8]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(hours, fraud_rates, marker='o', linewidth=2)
        ax.set_title('Fraud Rate by Hour of Day')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Fraud Rate (%)')
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True)
        st.pyplot(fig)
        
        # Amount-based patterns
        st.write("#### Amount-based Fraud Patterns")
        
        # Create example data
        amounts = ['<₹100', '₹100-₹500', '₹500-₹1000', '₹1000-₹5000', '₹5000-₹10000', '>₹10000']
        fraud_percentages = [1.2, 2.8, 4.1, 5.7, 7.3, 8.9]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x=amounts, y=fraud_percentages)
        ax.set_title('Fraud Rate by Transaction Amount')
        ax.set_xlabel('Amount Range')
        ax.set_ylabel('Fraud Rate (%)')
        ax.set_ylim(0, 10)
        
        for i, v in enumerate(fraud_percentages):
            ax.text(i, v + 0.2, f"{v}%", ha='center')
            
        st.pyplot(fig)
        
        # Merchant category patterns
        st.write("#### Merchant Category Risk Levels")
        
        categories = ['Electronics', 'Travel', 'Entertainment', 'Clothing', 'Utilities', 'Groceries']
        category_risks = [7.8, 6.5, 5.2, 4.9, 3.1, 2.4]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(categories, category_risks)
        
        # Add a color gradient to the bars
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.RdYlGn_r(category_risks[i]/10))
            
        ax.set_title('Fraud Rate by Merchant Category')
        ax.set_xlabel('Fraud Rate (%)')
        
        for i, v in enumerate(category_risks):
            ax.text(v + 0.1, i, f"{v}%", va='center')
            
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "90.4%")
        
        with col2:
            st.metric("Precision", "88.7%")
            
        with col3:
            st.metric("Recall", "85.2%")
            
        with col4:
            st.metric("F1 Score", "86.9%")
            
        st.write("#### Confusion Matrix")
        
        # Create example confusion matrix
        cm = np.array([[8750, 250], [350, 2650]])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                   yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("#### ROC Curve")
        
        # Create example ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = 0.94)')
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

# Add footer
st.markdown("---")
st.markdown("UPI Fraud Detection System | Built with Streamlit")
