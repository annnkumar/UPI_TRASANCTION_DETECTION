import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from preprocessing.data_processor import engineer_features

print("Loading dataset...")
# Load the dataset
df = pd.read_csv('attached_assets/Upi_fraud_dataset-checkpoint.csv')

print(f"Dataset loaded with shape: {df.shape}")

# Basic preprocessing
print("Preprocessing data...")
# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Extract TransactionFrequency numeric value
df['TransactionFrequencyValue'] = df['TransactionFrequency'].str.split('/').str[0].astype(int)

# Convert boolean columns to integers
bool_cols = ['UnusualLocation', 'UnusualAmount', 'NewDevice', 'FraudFlag']
for col in bool_cols:
    df[col] = df[col].astype(int)

# Apply feature engineering
print("Applying feature engineering...")
df_engineered = engineer_features(df)

# Identify features and target
exclude_cols = ['TransactionID', 'UserID', 'DeviceID', 'Timestamp', 'IPAddress', 'PhoneNumber', 'TransactionFrequency']
target_col = 'FraudFlag'

# Get features
features = [col for col in df_engineered.columns if col not in exclude_cols and col != target_col]
print(f"Using {len(features)} features for model training")

# Separate features and target
X = df_engineered[features]
y = df_engineered[target_col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify categorical and numerical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
boolean_cols = ['UnusualLocation', 'UnusualAmount', 'NewDevice', 'IsWeekend', 'IsNightTime', 'HighRiskIP']

# Remove boolean columns from numerical_cols
numerical_cols = [col for col in numerical_cols if col not in boolean_cols]

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")
print(f"Boolean columns: {len(boolean_cols)}")

# Preprocess the data
print("Preprocessing features...")
# 1. One-hot encoding for categorical features
if categorical_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train[categorical_cols])
    # Transform training data
    cat_features_train = ohe.transform(X_train[categorical_cols])
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    # Transform test data
    cat_features_test = ohe.transform(X_test[categorical_cols])
else:
    cat_features_train = np.empty((X_train.shape[0], 0))
    cat_features_test = np.empty((X_test.shape[0], 0))
    cat_feature_names = []

# 2. Scaling numerical features
if numerical_cols:
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])
    # Transform training data
    num_features_train = scaler.transform(X_train[numerical_cols])
    # Transform test data
    num_features_test = scaler.transform(X_test[numerical_cols])
else:
    num_features_train = np.empty((X_train.shape[0], 0))
    num_features_test = np.empty((X_test.shape[0], 0))

# 3. Extract boolean features
if boolean_cols:
    bool_features_train = X_train[boolean_cols].values
    bool_features_test = X_test[boolean_cols].values
else:
    bool_features_train = np.empty((X_train.shape[0], 0))
    bool_features_test = np.empty((X_test.shape[0], 0))

# 4. Combine all features
X_train_processed = np.hstack((num_features_train, cat_features_train, bool_features_train))
X_test_processed = np.hstack((num_features_test, cat_features_test, bool_features_test))

# Create feature names for the processed data
processed_feature_names = numerical_cols + list(cat_feature_names) + boolean_cols

print(f"Processed training data shape: {X_train_processed.shape}")
print(f"Processed test data shape: {X_test_processed.shape}")

# Save preprocessing objects for later use in the app
preprocessing_objects = {
    'ohe': ohe,
    'scaler': scaler,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'boolean_cols': boolean_cols,
    'final_features': processed_feature_names
}

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save preprocessing objects
with open('models/preprocessing_objects.pkl', 'wb') as f:
    pickle.dump(preprocessing_objects, f)

print("Preprocessing objects saved to models/preprocessing_objects.pkl")

# Train model (Random Forest as a good default)
print("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train_processed, y_train)

# Evaluate on test set
y_pred = model.predict(X_test_processed)
accuracy = (y_pred == y_test).mean()
print(f"Test accuracy: {accuracy:.4f}")

# Save the model
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved to models/best_model.pkl")
print("Model generation complete! You can now run the Streamlit app.")