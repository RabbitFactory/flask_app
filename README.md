# Heart Disease Prediction Web App

## Overview

A modern web application that uses a Random Forest Machine Learning algorithm to predict a person's risk of heart disease. By entering health information such as age, gender, blood pressure, and cholesterol levels, users can get an instant prediction about their heart health risk. The application is built with Flask and features a clean, user-friendly interface.

![Heart Disease Predictor](https://github.com/asthasharma98/Heart-Disease-Prediction-Deployment/blob/master/Readme_resources/heart_disease.gif)

## Features

- **Instant Predictions**: Get immediate heart disease risk assessment
- **User-friendly Interface**: Clean, modern design with intuitive form inputs
- **High Accuracy**: Model achieves 77.33% accuracy in predicting heart disease risk
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

2. **Create a virtual environment (recommended)**

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Running the Application

1. **Start the Flask server**

```bash
python app.py
```

2. **Access the application**

Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

3. **Using the application**

- Fill in all the required health information
- Click the "Predict" button
- View your heart disease risk prediction result

## Project Structure

- `app.py`: Main Flask application file
- `prediction.py`: Contains the machine learning model training code
- `heart-disease-prediction-knn-model.pkl`: Pre-trained Random Forest model
- `templates/`: HTML templates for the web interface
- `static/`: CSS stylesheets and static assets

## Model Information

The application uses a Random Forest Classifier trained on the Cleveland Heart Disease dataset with the following features:

| Feature | Description |
|---------|-------------|
| Age | Age of the patient |
| Sex | Gender (1 = male, 0 = female) |
| Chest Pain Type | Type of chest pain experienced |
| Resting Blood Pressure | Blood pressure (mm Hg) at rest |
| Serum Cholesterol | Cholesterol level (mg/dl) |
| Fasting Blood Sugar | Blood sugar > 120 mg/dl (1 = true, 0 = false) |
| Resting ECG Results | Electrocardiogram results at rest |
| Max Heart Rate | Maximum heart rate achieved |
| Exercise-induced Angina | Angina caused by exercise (1 = yes, 0 = no) |
| ST Depression | ST depression induced by exercise relative to rest |
| Slope | Slope of the peak exercise ST segment |
| Number of Major Vessels | Number of major vessels colored by fluoroscopy |
| Thalassemia | Blood disorder (0 = normal, 1 = fixed defect, 2 = reversible defect) |

## Technologies Used

- **Python**: Core programming language
- **Flask**: Web framework
- **Scikit-learn**: Machine learning library
- **Pandas/NumPy**: Data processing
- **HTML/CSS**: Frontend interface

## Future Enhancements

- Improve model performance with advanced algorithms
- Add data visualization of risk factors
- Implement user accounts for tracking health metrics over time
- Provide personalized health recommendations based on prediction results

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Heart Disease Dataset Information](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
