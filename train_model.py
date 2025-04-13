# train_model.py

import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data/insurance.csv")

# Encode categorical features
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

data['sex'] = le_sex.fit_transform(data['sex'])
data['smoker'] = le_smoker.fit_transform(data['smoker'])
data['region'] = le_region.fit_transform(data['region'])

# Features and target
X = data.drop('charges', axis=1)
y = data['charges']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump({
    "model": model,
    "le_sex": le_sex,
    "le_smoker": le_smoker,
    "le_region": le_region
}, "insurance_model.pkl")

print("✅ Model trained and saved to insurance_model.pkl")
