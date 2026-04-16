from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load files
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
car_names = pickle.load(open("car_names.pkl", "rb"))
accuracy = pickle.load(open("accuracy.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html',
                           car_names=car_names,
                           accuracy=round(accuracy*100, 2))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        car_name = request.form['Car_Name']
        car_encoded = le.transform([car_name])[0]

        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Fuel_Type = int(request.form['Fuel_Type'])
        Seller_Type = int(request.form['Seller_Type'])
        Transmission = int(request.form['Transmission'])
        Owner = int(request.form['Owner'])

        features = np.array([[car_encoded, Year, Present_Price,
                              Kms_Driven, Fuel_Type,
                              Seller_Type, Transmission, Owner]])

        prediction = model.predict(features)[0]
        prediction = round(prediction, 2)

        return render_template('index.html',
                               prediction_text=f"Estimated Price: {prediction} Lakhs",
                               car_names=car_names,
                               accuracy=round(accuracy*100, 2))

    except Exception as e:
        return render_template('index.html',
                               prediction_text="⚠️ Error in input!",
                               car_names=car_names,
                               accuracy=round(accuracy*100, 2))

if __name__ == "__main__":
    app.run(debug=True)