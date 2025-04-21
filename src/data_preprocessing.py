import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import json
import os

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.01):
    """
    Generate synthetic credit card transaction data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate normal transactions
    n_normal = int(n_samples * (1 - fraud_ratio))
    normal_amounts = np.random.exponential(scale=100, size=n_normal)
    normal_times = np.random.uniform(0, 172800, size=n_normal)  # 48 hours in seconds
    normal_v1 = np.random.normal(0, 1, size=n_normal)
    normal_v2 = np.random.normal(0, 1, size=n_normal)
    
    # Generate fraudulent transactions
    n_fraud = int(n_samples * fraud_ratio)
    fraud_amounts = np.random.exponential(scale=500, size=n_fraud)  # Higher average amount
    fraud_times = np.random.uniform(0, 172800, size=n_fraud)
    fraud_v1 = np.random.normal(-2, 1, size=n_fraud)  # Different distribution
    fraud_v2 = np.random.normal(2, 1, size=n_fraud)   # Different distribution
    
    # Combine data
    data = {
        'amount': np.concatenate([normal_amounts, fraud_amounts]),
        'time': np.concatenate([normal_times, fraud_times]),
        'v1': np.concatenate([normal_v1, fraud_v1]),
        'v2': np.concatenate([normal_v2, fraud_v2]),
        'is_fraud': np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    }
    
    return pd.DataFrame(data)

def load_data(file_path):
    """
    Load the credit card fraud dataset
    """
    if not os.path.exists(file_path):
        print("Generating synthetic data...")
        df = generate_synthetic_data()
        df.to_csv(file_path, index=False)
        print(f"Synthetic data saved to {file_path}")
    else:
        df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocess the data for model training
    """
    # Split features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Save processed data and scaler
    np.save('data/X_train.npy', X_train_resampled)
    np.save('data/X_test.npy', X_test_scaled)
    np.save('data/y_train.npy', y_train_resampled)
    np.save('data/y_test.npy', y_test)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(X.columns.tolist(), f)
    
    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load or generate data
    df = load_data('data/synthetic_fraud_dataset.csv')
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

if __name__ == "__main__":
    main() 