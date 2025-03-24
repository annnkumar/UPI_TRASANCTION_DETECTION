import pandas as pd
import numpy as np

def preprocess_data(df, preprocessing_objects):
    """
    Preprocess transaction data for model prediction.
    
    Args:
        df (DataFrame): Input transaction data
        preprocessing_objects (dict): Dictionary containing preprocessing objects
        
    Returns:
        DataFrame: Preprocessed features ready for model prediction
    """
    # Extract preprocessing objects
    ohe = preprocessing_objects['ohe']
    scaler = preprocessing_objects['scaler']
    categorical_cols = preprocessing_objects['categorical_cols']
    numerical_cols = preprocessing_objects['numerical_cols']
    boolean_cols = preprocessing_objects['boolean_cols']
    final_features = preprocessing_objects['final_features']
    
    # Ensure all required columns exist
    for col in categorical_cols + numerical_cols + boolean_cols:
        if col not in df.columns:
            if col in numerical_cols:
                df[col] = 0  # Default for numerical columns
            elif col in boolean_cols:
                df[col] = False  # Default for boolean columns
            else:
                df[col] = "Unknown"  # Default for categorical columns
    
    # Process categorical features
    cat_features = ohe.transform(df[categorical_cols])
    cat_feature_names = ohe.get_feature_names_out(categorical_cols)
    cat_features_df = pd.DataFrame(cat_features, columns=cat_feature_names, index=df.index)
    
    # Process numerical features
    numerical_cols_present = [col for col in numerical_cols if col in df.columns]
    if numerical_cols_present:
        num_features = scaler.transform(df[numerical_cols_present])
        num_features_df = pd.DataFrame(num_features, columns=numerical_cols_present, index=df.index)
    else:
        num_features_df = pd.DataFrame(index=df.index)
    
    # Process boolean features
    boolean_cols_present = [col for col in boolean_cols if col in df.columns]
    if boolean_cols_present:
        boolean_features_df = df[boolean_cols_present].astype(int)
    else:
        boolean_features_df = pd.DataFrame(index=df.index)
    
    # Combine all features
    processed_df = pd.concat([num_features_df, cat_features_df, boolean_features_df], axis=1)
    
    # Ensure all final features are present
    for feature in final_features:
        if feature not in processed_df.columns:
            processed_df[feature] = 0  # Default value for missing features
    
    # Select only the features used by the model
    processed_df = processed_df[final_features]
    
    return processed_df

def engineer_features(df):
    """
    Engineer additional features for model prediction.
    
    Args:
        df (DataFrame): Input transaction data
        
    Returns:
        DataFrame: DataFrame with additional engineered features
    """
    # Make a copy to avoid modifying the original dataframe
    df_engineered = df.copy()
    
    # Convert Timestamp to datetime if it's not already
    if 'Timestamp' in df_engineered.columns and not pd.api.types.is_datetime64_any_dtype(df_engineered['Timestamp']):
        df_engineered['Timestamp'] = pd.to_datetime(df_engineered['Timestamp'])
    
    # Extract time components if Timestamp is available
    if 'Timestamp' in df_engineered.columns:
        df_engineered['Day'] = df_engineered['Timestamp'].dt.day
        df_engineered['Month'] = df_engineered['Timestamp'].dt.month
        df_engineered['Year'] = df_engineered['Timestamp'].dt.year
        df_engineered['Hour'] = df_engineered['Timestamp'].dt.hour
        df_engineered['DayOfWeek'] = df_engineered['Timestamp'].dt.dayofweek
        
        # Create weekend flag
        df_engineered['IsWeekend'] = df_engineered['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Create night time flag (10 PM - 6 AM)
        df_engineered['IsNightTime'] = df_engineered['Hour'].apply(lambda x: 1 if (x >= 22 or x < 6) else 0)
    
    # Extract TransactionFrequency numeric value if available
    if 'TransactionFrequency' in df_engineered.columns:
        # Check if it's in the expected format (e.g., '5/day')
        if df_engineered['TransactionFrequency'].dtype == 'object' and '/' in str(df_engineered['TransactionFrequency'].iloc[0]):
            df_engineered['TransactionFrequencyValue'] = df_engineered['TransactionFrequency'].str.split('/').str[0].astype(int)
        else:
            # Default to 1 if format is unexpected
            df_engineered['TransactionFrequencyValue'] = 1
    
    # Create amount-based features if relevant columns exist
    if 'Amount' in df_engineered.columns and 'AvgTransactionAmount' in df_engineered.columns:
        df_engineered['AmountRatio'] = df_engineered['Amount'] / df_engineered['AvgTransactionAmount'].replace(0, 1)
        df_engineered['AmountDiff'] = df_engineered['Amount'] - df_engineered['AvgTransactionAmount']
    
    # Create risk score feature if relevant columns exist
    risk_factors = []
    
    if 'UnusualLocation' in df_engineered.columns:
        risk_factors.append(df_engineered['UnusualLocation'].astype(int))
    
    if 'UnusualAmount' in df_engineered.columns:
        risk_factors.append(df_engineered['UnusualAmount'].astype(int))
    
    if 'NewDevice' in df_engineered.columns:
        risk_factors.append(df_engineered['NewDevice'].astype(int))
    
    if 'FailedAttempts' in df_engineered.columns:
        risk_factors.append(df_engineered['FailedAttempts'])
    
    if 'IsNightTime' in df_engineered.columns:
        risk_factors.append(df_engineered['IsNightTime'])
    
    if risk_factors:
        df_engineered['RiskScore'] = sum(risk_factors)
    
    # Process IP Address if available
    if 'IPAddress' in df_engineered.columns:
        try:
            # Extract first octet of IP
            df_engineered['IP_FirstOctet'] = df_engineered['IPAddress'].astype(str).str.split('.').str[0].astype(int)
            
            # Create High-Risk IP Flag (simplified approach)
            high_risk_ranges = [(0, 10), (172, 172), (192, 192), (198, 198)]
            df_engineered['HighRiskIP'] = df_engineered['IP_FirstOctet'].apply(
                lambda x: 1 if any(lower <= x <= upper for lower, upper in high_risk_ranges) else 0
            )
        except:
            # If IP address format is unexpected
            df_engineered['IP_FirstOctet'] = 0
            df_engineered['HighRiskIP'] = 0
    
    # Convert boolean columns to integers
    bool_cols = ['UnusualLocation', 'UnusualAmount', 'NewDevice']
    for col in bool_cols:
        if col in df_engineered.columns:
            if df_engineered[col].dtype == bool:
                df_engineered[col] = df_engineered[col].astype(int)
            elif df_engineered[col].dtype == object:
                # Convert string 'True'/'False' to integer if needed
                df_engineered[col] = df_engineered[col].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
    
    return df_engineered