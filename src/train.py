import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def load_processed_data():
    """
    Load the processed training and testing data
    """
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train multiple models and return their instances
    """
    # Initialize models
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=1
    )
    
    # Train models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    return {
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }


def evaluate_models(models, X_test, y_test):
    """
    Evaluate models and return their performance metrics
    """
    results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        results[name] = metrics
        
        # Save model
        os.makedirs('models', exist_ok=True)
        model_path = f'models/{name.lower().replace(" ", "_")}_model.joblib'
        joblib.dump(model, model_path)
    
    return results


def plot_results(results, X_test, y_test):
    """
    Plot evaluation metrics and confusion matrices
    """
    # Create figure for ROC curves
    plt.figure(figsize=(10, 6))
    for name, metrics in results.items():
        model_path = f'models/{name.lower().replace(" ", "_")}_model.joblib'
        model = joblib.load(model_path)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(
            recall,
            precision,
            label=f'{name} (AP={metrics["average_precision"]:.2f})'
        )
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('models/precision_recall_curve.png')
    plt.close()
    
    # Plot confusion matrices
    for name, metrics in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        cm_path = f'models/confusion_matrix_{name.lower().replace(" ", "_")}.png'
        plt.savefig(cm_path)
        plt.close()


def main():
    # Load processed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_test, y_test)
    
    # Plot results
    plot_results(results, X_test, y_test)
    
    # Print results
    for name, metrics in results.items():
        print(f"\n{name} Model Results:")
        print("Classification Report:")
        print(metrics['classification_report'])
        print(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")


if __name__ == "__main__":
    main() 