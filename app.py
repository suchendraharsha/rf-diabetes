from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from preprocess_input import preprocess_input
from preprocessing import preprocess_data # Import preprocessing.py
import os

app = Flask(__name__)

# Load and preprocess the data, train the model, and save.
try:
    diabet = pd.read_csv('diabetic_data.csv') # Load the data.
    X_rf, y_rf = preprocess_data(diabet.copy()) # Preprocess the data.
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_rf_scaled = scaler.fit_transform(X_train_rf)
    X_test_rf_scaled = scaler.transform(X_test_rf)

    RF = RandomForestClassifier()
    RF.fit(X_train_rf_scaled, y_train_rf)

    joblib.dump(scaler, 'models/your_scaler.pkl') # Save the scaler.
    joblib.dump(RF, 'models/your_model.pkl') # Save the model.
    print("Model and scaler saved successfully.")

except Exception as e:
    print(f"Error during model training/saving: {e}")

# Load the model and scaler for predictions.
LR = joblib.load('models/your_model.pkl')
SC = joblib.load('models/your_scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        processed_data = preprocess_input(data, SC)
        prediction = LR.predict(processed_data)[0]
        return jsonify({'prediction': str(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))