import pickle
import numpy as np

def load_model(model_path):
    """
    Load the trained model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: The loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def make_prediction(model, X):
    """
    Make predictions using the loaded model.
    
    Args:
        model: The trained model
        X: Preprocessed features
        
    Returns:
        tuple: (predictions, probabilities)
    """
    # Make binary predictions
    predictions = model.predict(X)
    
    # Get prediction probabilities if the model supports it
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)[:, 1]  # Get probabilities for the positive class
    else:
        # If model doesn't support predict_proba (e.g., SVM without probability=True)
        probabilities = np.zeros(len(predictions))
        
    return predictions, probabilities

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the model if available.
    
    Args:
        model: The trained model
        feature_names (list): List of feature names
        
    Returns:
        dict: Dictionary mapping feature names to importance scores
    """
    # Check if model has feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        return importance_dict
    
    # Check if model has coef_ attribute (linear models)
    elif hasattr(model, 'coef_'):
        # For binary classification, coef_ is 1D array
        if len(model.coef_.shape) == 1:
            importance_dict = dict(zip(feature_names, np.abs(model.coef_)))
        # For multiclass, coef_ is 2D array
        else:
            # Use average of absolute coefficients across classes
            importance_dict = dict(zip(feature_names, np.mean(np.abs(model.coef_), axis=0)))
        return importance_dict
    
    # For other models that don't expose feature importance
    else:
        return None
