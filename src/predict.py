import numpy as np
import joblib
import json
from datetime import datetime


class FraudDetector:
    def __init__(self):
        # Load models
        self.rf_model = joblib.load('models/random_forest_model.joblib')
        self.xgb_model = joblib.load('models/xgboost_model.joblib')
        self.scaler = joblib.load('models/scaler.joblib')
        
        # Load feature names
        with open('models/feature_names.json', 'r') as f:
            self.feature_names = json.load(f)
    
    def preprocess_transaction(self, transaction_data):
        """
        Preprocess a single transaction
        """
        # Create feature array
        features = np.array([
            transaction_data['amount'],
            transaction_data['time'],
            transaction_data['v1'],
            transaction_data['v2']
        ]).reshape(1, -1)
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        return scaled_features
    
    def predict(self, transaction_data):
        """
        Make prediction using ensemble of models
        """
        # Preprocess transaction
        X = self.preprocess_transaction(transaction_data)
        
        # Get predictions from both models
        rf_pred_proba = self.rf_model.predict_proba(X)[0, 1]
        xgb_pred_proba = self.xgb_model.predict_proba(X)[0, 1]
        
        # Ensemble prediction (average of probabilities)
        fraud_probability = (rf_pred_proba + xgb_pred_proba) / 2
        is_fraud = fraud_probability > 0.5
        
        return {
            'is_fraud': bool(is_fraud),
            'fraud_probability': float(fraud_probability),
            'model_details': {
                'random_forest_probability': float(rf_pred_proba),
                'xgboost_probability': float(xgb_pred_proba)
            },
            'timestamp': datetime.now().isoformat()
        }


def main():
    # Example usage
    detector = FraudDetector()
    
    # Test with example transactions
    normal_transaction = {
        'amount': 100,
        'time': 0.5,
        'v1': 0.1,
        'v2': 0.2
    }
    
    suspicious_transaction = {
        'amount': 500,
        'time': 2.0,
        'v1': 2.0,
        'v2': -2.0
    }
    
    print("\nNormal Transaction Analysis:")
    print(json.dumps(
        detector.predict(normal_transaction),
        indent=2
    ))
    
    print("\nSuspicious Transaction Analysis:")
    print(json.dumps(
        detector.predict(suspicious_transaction),
        indent=2
    ))


if __name__ == "__main__":
    main() 