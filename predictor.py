import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Example training data (replace with your full dataset)
X_train = np.array([[140, 26, 85, 60, 45, 35]])
y_train = np.array([3.7])

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model to disk
joblib.dump(model, "crop_model.pkl")
print("Model saved as crop_model.pkl")