from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the logistic regression model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json(force=True)
    features = np.array(input_data['features']).reshape(1, -1)
    prediction = model.predict(features)
    probability = model.predict_proba(features)[:, 1]

    return jsonify({
        'prediction': int(prediction[0]),
        'probability': float(probability[0])
    })

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8888, debug=True)
