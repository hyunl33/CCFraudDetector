import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance


def load_data():
    """
    Load the processed data and feature names
    """
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Load original data to get feature names
    df = pd.read_csv('data/synthetic_fraud_dataset.csv')
    feature_names = df.drop('is_fraud', axis=1).columns.tolist()
    
    return X_test, y_test, feature_names


def analyze_random_forest(X_test, y_test, feature_names):
    """
    Analyze Random Forest model's feature importance
    """
    model = joblib.load('models/random_forest_model.joblib')
    
    # Get feature importance from model
    importance = model.feature_importances_
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    
    return {
        'model_importance': importance,
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std
    }


def analyze_xgboost(X_test, y_test, feature_names):
    """
    Analyze XGBoost model's feature importance
    """
    model = joblib.load('models/xgboost_model.joblib')
    
    # Get feature importance from model
    importance = model.feature_importances_
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42
    )
    
    return {
        'model_importance': importance,
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std
    }


def plot_feature_importance(importance_dict, feature_names, model_name):
    """
    Plot feature importance comparison
    """
    plt.figure(figsize=(12, 6))
    
    # Model's built-in importance
    plt.subplot(1, 2, 1)
    importance = pd.Series(
        importance_dict['model_importance'],
        index=feature_names
    ).sort_values(ascending=True)
    
    plt.barh(range(len(importance)), importance)
    plt.yticks(range(len(importance)), importance.index)
    plt.xlabel('Importance')
    plt.title(f'{model_name} Built-in Feature Importance')
    
    # Permutation importance
    plt.subplot(1, 2, 2)
    perm_importance = pd.Series(
        importance_dict['perm_importance_mean'],
        index=feature_names
    ).sort_values(ascending=True)
    
    plt.barh(
        range(len(perm_importance)),
        perm_importance,
        xerr=importance_dict['perm_importance_std']
    )
    plt.yticks(range(len(perm_importance)), perm_importance.index)
    plt.xlabel('Permutation Importance')
    plt.title(f'{model_name} Permutation Feature Importance')
    
    plt.tight_layout()
    plt.savefig(f'models/{model_name.lower().replace(" ", "_")}_importance.png')
    plt.close()


def analyze_feature_interactions(X_test, feature_names):
    """
    Analyze and plot feature interactions
    """
    # Convert to DataFrame for easier analysis
    X_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Calculate correlation matrix
    corr_matrix = X_df.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f'
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('models/feature_correlations.png')
    plt.close()
    
    return corr_matrix


def main():
    # Load data
    X_test, y_test, feature_names = load_data()
    
    # Analyze Random Forest
    rf_importance = analyze_random_forest(X_test, y_test, feature_names)
    plot_feature_importance(rf_importance, feature_names, 'Random Forest')
    
    # Analyze XGBoost
    xgb_importance = analyze_xgboost(X_test, y_test, feature_names)
    plot_feature_importance(xgb_importance, feature_names, 'XGBoost')
    
    # Analyze feature interactions
    corr_matrix = analyze_feature_interactions(X_test, feature_names)
    
    # Print summary
    print("\nFeature Importance Analysis Summary:")
    print("\nRandom Forest Top Features:")
    rf_top = pd.Series(
        rf_importance['model_importance'],
        index=feature_names
    ).sort_values(ascending=False)
    print(rf_top)
    
    print("\nXGBoost Top Features:")
    xgb_top = pd.Series(
        xgb_importance['model_importance'],
        index=feature_names
    ).sort_values(ascending=False)
    print(xgb_top)
    
    print("\nFeature Correlation Summary:")
    print(corr_matrix)


if __name__ == "__main__":
    main() 