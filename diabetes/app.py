import numpy as np
import pandas as pd
from flask import Flask, render_template, request

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('model.csv')  # Update the file path
features = data.drop(columns='Outcome', axis=1)
target = data['Outcome']

# Scale the features
scaler = StandardScaler()
scaler.fit(features)
std_data = scaler.transform(features)

# Create an SVM model with linear kernel
model = SVC(kernel='linear')
model.fit(std_data, target)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        input_values = [float(request.form['pregnancies']),
                        float(request.form['glucose']),
                        float(request.form['blood_pressure']),
                        float(request.form['skin_thickness']),
                        float(request.form['insulin']),
                        float(request.form['bmi']),
                        float(request.form['diabetes_pedigree']),
                        float(request.form['age'])]

        x_input = scaler.transform([input_values])
        input_result = model.predict(x_input)

        if input_result == 0:
            prediction = "The person does not have Diabetes"
        else:
            prediction = "The person has Diabetes"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    
    app.run(debug=True)

