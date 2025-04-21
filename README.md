# Credit Card Fraud Detection System

A machine learning-based system for detecting fraudulent credit card transactions using Random Forest and XGBoost models.

## Features

- Synthetic data generation for credit card transactions
- Data preprocessing with SMOTE for handling class imbalance
- Ensemble of Random Forest and XGBoost models
- Real-time fraud detection API
- Web interface for transaction analysis
- Model performance visualization and analysis

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CCFraudDetector.git
cd CCFraudDetector
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the data preprocessing and model training:
```bash
python src/data_preprocessing.py
python src/train.py
```

5. Start the web application:
```bash
python src/app.py
```

The application will be available at `http://localhost:8000`

## Project Structure

```
CCFraudDetector/
├── data/                  # Data files
├── models/                # Trained models and visualizations
├── src/                   # Source code
│   ├── app.py            # Flask web application
│   ├── predict.py        # Prediction module
│   ├── train.py          # Model training
│   ├── analyze.py        # Model analysis
│   └── data_preprocessing.py  # Data preprocessing
├── templates/            # HTML templates
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## API Endpoints

- `POST /predict`: Analyze a single transaction
- `POST /batch_predict`: Analyze multiple transactions

## Example Usage

```python
import requests

# Single transaction
transaction = {
    "amount": 100.0,
    "time": 0.5,
    "v1": 0.1,
    "v2": 0.2
}

response = requests.post("http://localhost:8000/predict", json=transaction)
print(response.json())
```

## License

MIT License 