import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import joblib
import json


def load_models():
    """
    Load trained models
    """
    rf_model = joblib.load('models/random_forest_model.joblib')
    xgb_model = joblib.load('models/xgboost_model.joblib')
    return rf_model, xgb_model


def load_data():
    """
    Load test data
    """
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    return X_test, y_test


def analyze_feature_importance(models, feature_names):
    """
    Analyze and visualize feature importance
    """
    importance_data = {}
    
    # Random Forest feature importance
    rf_importance = models[0].feature_importances_
    importance_data['Random Forest'] = dict(zip(feature_names, rf_importance))
    
    # XGBoost feature importance
    xgb_importance = models[1].feature_importances_
    importance_data['XGBoost'] = dict(zip(feature_names, xgb_importance))
    
    # Save feature importance data
    with open('models/feature_importance.json', 'w') as f:
        json.dump(importance_data, f, indent=4)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    
    # Random Forest importance plot
    plt.subplot(1, 2, 1)
    sns.barplot(x=list(importance_data['Random Forest'].values()),
                y=list(importance_data['Random Forest'].keys()))
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    
    # XGBoost importance plot
    plt.subplot(1, 2, 2)
    sns.barplot(x=list(importance_data['XGBoost'].values()),
                y=list(importance_data['XGBoost'].keys()))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()


def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for both models
    """
    plt.figure(figsize=(10, 6))
    
    for name, model in zip(['Random Forest', 'XGBoost'], models):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.savefig('models/roc_curves.png')
    plt.close()


def main():
    print("Loading models and data...")
    models = load_models()
    X_test, y_test = load_data()
    
    print("\nAnalyzing feature importance...")
    with open('models/feature_names.json', 'r') as f:
        feature_names = json.load(f)
    analyze_feature_importance(models, feature_names)
    
    print("\nPlotting ROC curves...")
    plot_roc_curves(models, X_test, y_test)
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main() 