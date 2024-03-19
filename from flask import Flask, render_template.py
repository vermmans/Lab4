from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('fish_weight_prediction_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    length1 = float(request.form['length1'])
    length2 = float(request.form['length2'])
    length3 = float(request.form['length3'])
    height = float(request.form['height'])
    width = float(request.form['width'])

    # Make prediction using the loaded model
    prediction = model.predict([[length1, length2, length3, height, width]])

    # Return the prediction as JSON
    return jsonify({'weight': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
