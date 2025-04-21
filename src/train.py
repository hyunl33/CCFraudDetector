import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix
import seaborn as sns

def load_data():
    """
    Load preprocessed data
    """
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train):
    """
    Train Random Forest model
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    """
    Train XGBoost model
    """
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=1
    )
    xgb.fit(X_train, y_train)
    return xgb

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print(f"\n{model_name} Model Results:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    
    # Plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig('models/precision_recall_curve.png')
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'models/confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return {
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }

def main():
    # Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Train Random Forest
    print("\nTraining Random Forest model...")
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, 'models/random_forest_model.joblib')
    
    # Train XGBoost
    print("\nTraining XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train)
    joblib.dump(xgb_model, 'models/xgboost_model.joblib')
    
    # Evaluate models
    print("\nEvaluating models...")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    print("\nTraining and evaluation completed successfully!")

if __name__ == "__main__":
    main() 