import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from main import LassoCD
import os

def train_lasso_models():
    # Load and preprocess data
    data = pd.read_csv('Agrofood_co2_emission.csv')
    
    # Handle numeric columns and fill NaN values
    numeric_cols = data.select_dtypes(include='number').columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    
    # Handle Area feature with LabelEncoder
    area_encoder = LabelEncoder()
    data.Area = area_encoder.fit_transform(data.Area)
    
    # Save the encoder for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(area_encoder, 'models/area_encoder.joblib')
    
    # Extract X and y
    X = data.drop("total_emission", axis=1)
    y = data["total_emission"]
    
    lower_bound = y.quantile(0.05)
    upper_bound = y.quantile(0.95)
    mask = (y >= lower_bound) & (y <= upper_bound)
    y = y[mask]
    X = X[mask]
        


    
    # Train models for different alpha values
    alpha_values = np.arange(0, 40.5, 0.5)
    all_loss_histories = {}
    
    for alpha in alpha_values:
        print(f"Training Lasso model with alpha = {alpha:.1f}")
        
        # Initialize and train model
        model = LassoCD(alpha=alpha, max_iter=10000)
        model.fit(X, y)
        
        # Save model
        model_path = f'models/cd_lasso_at_{alpha:.1f}.joblib'
        joblib.dump(model, model_path)
        
        # Store loss history
        all_loss_histories[alpha] = model.loss_history
    
    # Save all loss histories
    joblib.dump(all_loss_histories, 'models/lasso_loss_histories.joblib')
    
    print("Training completed. Models and loss histories saved.")

if __name__ == "__main__":
    train_lasso_models() 