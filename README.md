# Credit Card Fraud Detection System

A real-time credit card fraud detection system using machine learning models.

## Overview

This project implements a machine learning-based system for detecting fraudulent credit card transactions in real-time. The system uses two powerful models (Random Forest and XGBoost) to achieve high accuracy in fraud detection.

## Features

- Real-time transaction analysis
- Web-based user interface
- API endpoints for single and batch predictions
- Model performance visualization
- Feature importance analysis

## Model Performance

### Random Forest Model
- Accuracy: 1.00
- ROC AUC: 0.9996
- Average Precision: 0.9715
- Fraud Detection Precision: 0.76
- Fraud Detection Recall: 0.95

### XGBoost Model
- Accuracy: 1.00
- ROC AUC: 0.9999
- Average Precision: 0.9951
- Fraud Detection Precision: 0.77
- Fraud Detection Recall: 1.00

## Feature Importance

Top features for fraud detection:
1. Transaction Amount (53.49% - Random Forest, 79.45% - XGBoost)
2. Time (19.28% - Random Forest, 9.69% - XGBoost)
3. V1 (19.68% - Random Forest, 4.26% - XGBoost)
4. V2 (7.55% - Random Forest, 6.61% - XGBoost)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CCFraudDetector.git
cd CCFraudDetector
```

2. Create and activate virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python src/data_preprocessing.py
```

2. Model Training:
```bash
python src/train.py
```

3. Start Web Application:
```bash
python src/app.py
```

4. Access the web interface at `http://localhost:8000`

## API Endpoints

### Single Transaction Prediction
```bash
POST /predict
{
    "amount": 100.0,
    "time": 1000,
    "v1": -1.0,
    "v2": 0.5
}
```

### Batch Prediction
```bash
POST /batch_predict
[
    {
        "amount": 100.0,
        "time": 1000,
        "v1": -1.0,
        "v2": 0.5
    },
    {
        "amount": 200.0,
        "time": 2000,
        "v1": -2.0,
        "v2": 1.0
    }
]
```

## Project Structure

```
CCFraudDetector/
├── data/                  # Data files
├── models/               # Trained models and visualizations
├── src/                  # Source code
│   ├── app.py           # Flask web application
│   ├── data_preprocessing.py
│   ├── model_analysis.py
│   ├── predict.py
│   ├── train.py
│   └── templates/       # HTML templates
├── requirements.txt     # Dependencies
└── README.md           # Project documentation
```

## Technologies Used

- Python 3.11
- scikit-learn 1.3.0
- XGBoost
- Flask 3.1.0
- NumPy 1.24.3
- Pandas
- Joblib 1.3.1

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 