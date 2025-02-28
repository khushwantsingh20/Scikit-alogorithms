import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("disease_dataset_500.csv")

# Separate features and target variable
X = data.iloc[:, :-1]  # All columns except the last (Symptoms)
y = data.iloc[:, -1]   # Last column (Disease)

# Convert categorical labels to numbers (Label Encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)  # Convert disease names to numbers

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Predict on test data
y_pred = gb_model.predict(X_test)

# Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Disease Prediction Model Accuracy: {accuracy:.4f}")

# Function to predict disease based on symptoms
def predict_disease(symptom_list):
    input_data = np.array(symptom_list).reshape(1, -1)  # Reshape input to match model format
    prediction = gb_model.predict(input_data)
    return le.inverse_transform(prediction)[0]  # Convert number back to disease name

# Example usage
sample_symptoms = [1, 0, 1, 0, 1, 0]  # Example symptoms (1=Yes, 0=No)
predicted_disease = predict_disease(sample_symptoms)
print(f"Predicted Disease: {predicted_disease}")
