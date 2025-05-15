# Importing essential libraries
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Create a list with all features
        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        # Standardize the data
        # Note: In a production environment, we should use the same scaler that was used during training
        # Since we don't have access to that scaler, we'll use a more appropriate approach for standardizing a single data point
        
        # Load the dataset to compute mean and std for standardization
        heart_df = pd.read_csv("heart_cleveland_upload.csv")
        # Rename 'condition' to 'target' to match training data
        heart_df = heart_df.rename(columns={'condition':'target'})
        # Get feature data
        X = heart_df.drop(columns='target')
        
        # Create a new scaler and fit it on the training data
        scaler = StandardScaler()
        scaler.fit(X)
        
        # Transform the input data using the fitted scaler
        data_scaled = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(data_scaled)
        
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
