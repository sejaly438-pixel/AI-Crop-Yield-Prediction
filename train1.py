import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("crop_data1.csv")

# Separate features and target
X = df.drop("Yield", axis=1)
y = df["Yield"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("Model Accuracy:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "crop_model.pkl")

print("Model saved successfully!")