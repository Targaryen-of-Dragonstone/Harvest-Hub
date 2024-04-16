from flask import Flask, render_template, request, jsonify
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('/Users/codesundar/Downloads/Crop_recommendation (1).csv')

# Select features and target
features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']

# Create an instance of the RFClassifier and train the model
RF = RandomForestClassifier(n_estimators=20, random_state=0)
 

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, target, test_size=0.2, random_state=2)
RF.fit(Xtrain,Ytrain)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    input_data = [float(data['N']), float(data['P']), float(data['K']),
                  float(data['temperature']), float(data['humidity']),
                  float(data['ph']), float(data['rainfall'])]
    prediction = RF.predict([input_data])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
