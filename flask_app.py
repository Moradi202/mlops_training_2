from flask import Flask, jsonify, request
from pickle import load
from logzero import logger, logfile
from datetime import datetime

# Set up logging
logfile('app.log', maxBytes=1e6, backupCount=3, disableStderrLogger=True)

# Load model (consider loading inside the route for better error handling)
try:
    model = load(open('model.pkl', 'rb'))
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

app = Flask(__name__)
logger.info('Flask app started')

@app.route('/')
def health_check():
    return 'OK'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        if not input_data:
            return jsonify({'message': 'No input data provided'}), 400
        
        # Prepare input for model
        value_list = [list(input_data.values())]
        prediction = model.predict(value_list)

        # Interpret prediction
        if prediction[0] == 0:
            prediction_result = 'Not likely to purchase'
        elif prediction[0] == 1:
            prediction_result = 'Likely to purchase'

        # Log the prediction with timestamp
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"{current_datetime} - Prediction: {prediction_result}")
        
        return jsonify({'prediction': prediction_result}), 200

    except Exception as e:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.error(f"{current_datetime} - Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
