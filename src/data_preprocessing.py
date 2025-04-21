import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

def generate_synthetic_data(n_samples=10000, fraud_ratio=0.01):
    """
    Generate synthetic credit card transaction data
    """
    np.random.seed(42)
    
    # Generate normal transactions
    n_normal = int(n_samples * (1 - fraud_ratio))
    n_fraud = n_samples - n_normal
    
    # Generate features for normal transactions
    normal_amount = np.random.normal(100, 50, n_normal)
    normal_time = np.random.normal(0, 1, n_normal)
    normal_v1 = np.random.normal(0, 1, n_normal)
    normal_v2 = np.random.normal(0, 1, n_normal)
    
    # Generate features for fraud transactions
    fraud_amount = np.random.normal(500, 200, n_fraud)  # Higher amounts
    fraud_time = np.random.normal(2, 1, n_fraud)  # Different time distribution
    fraud_v1 = np.random.normal(2, 1, n_fraud)  # Different feature distribution
    fraud_v2 = np.random.normal(-2, 1, n_fraud)  # Different feature distribution
    
    # Combine normal and fraud transactions
    amount = np.concatenate([normal_amount, fraud_amount])
    time = np.concatenate([normal_time, fraud_time])
    v1 = np.concatenate([normal_v1, fraud_v1])
    v2 = np.concatenate([normal_v2, fraud_v2])
    is_fraud = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
    
    # Create DataFrame
    df = pd.DataFrame({
        'amount': amount,
        'time': time,
        'v1': v1,
        'v2': v2,
        'is_fraud': is_fraud
    })
    
    return df

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
    Preprocess the data:
    1. Handle missing values
    2. Scale numerical features
    3. Split into features and target
    """
    # Drop any rows with missing values
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for later use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def handle_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE
    """
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load or generate data
    df = load_data('data/synthetic_fraud_dataset.csv')
    
    # Preprocess data
    X, y = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_imbalance(X_train, y_train)
    
    # Save processed data
    np.save('data/X_train.npy', X_train_resampled)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train_resampled)
    np.save('data/y_test.npy', y_test)
    
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train_resampled.shape}")
    print(f"Testing set shape: {X_test.shape}")

if __name__ == "__main__":
    main() 