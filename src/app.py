from flask import Flask, request, jsonify, render_template
from predict import FraudDetector
import json

app = Flask(__name__)
detector = FraudDetector()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get transaction data from request
        transaction_data = request.get_json()
        
        # Validate input
        required_fields = ['amount', 'time', 'v1', 'v2']
        for field in required_fields:
            if field not in transaction_data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Make prediction
        result = detector.predict(transaction_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Get multiple transactions
        transactions = request.get_json()
        
        if not isinstance(transactions, list):
            return jsonify({
                'error': 'Input must be a list of transactions'
            }), 400
        
        # Process each transaction
        results = []
        for transaction in transactions:
            result = detector.predict(transaction)
            results.append(result)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 