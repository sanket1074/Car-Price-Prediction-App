import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("car data.csv")

# Store car names
car_names = sorted(data['Car_Name'].unique())

# Encoding categorical data
data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol':0, 'Diesel':1, 'CNG':2})
data['Seller_Type'] = data['Seller_Type'].map({'Dealer':0, 'Individual':1})
data['Transmission'] = data['Transmission'].map({'Manual':0, 'Automatic':1})

# Encode Car_Name
le = LabelEncoder()
data['Car_Name'] = le.fit_transform(data['Car_Name'])

# Features & target
X = data.drop(['Selling_Price'], axis=1)
y = data['Selling_Price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))
pickle.dump(car_names, open("car_names.pkl", "wb"))
pickle.dump(accuracy, open("accuracy.pkl", "wb"))

print("✅ Model trained successfully")
print("Accuracy:", accuracy)