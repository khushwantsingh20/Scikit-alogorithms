import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("credit_card_fraud_sample.csv")

# Drop irrelevant columns
df.drop(columns=["TransactionID", "Timestamp"], inplace=True)

# Encode categorical features
label_encoders = {}
categorical_cols = ["Location", "Merchant", "CardType"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store for later use
    print(label_encoders,"label_encoders")

# Separate features and target
X = df.drop(columns=["IsFraud"])
y = df["IsFraud"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, verbose=2)
print(grid_search,"grid_search")
grid_search.fit(X_train, y_train)

# Best model
tuned_rf = grid_search.best_estimator_

# Make predictions
y_pred = tuned_rf.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")
print("Classification Report:\n", report)

# Save trained model and preprocessors
joblib.dump(tuned_rf, "random_forest_fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model and preprocessors saved successfully!")

# Function to test the model with new transactions
def test_new_transaction(transaction):
    """Predict if a transaction is fraudulent or not."""
    model = joblib.load("random_forest_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    
    # Encode categorical features
    for col in ["Location", "Merchant", "CardType"]:
        transaction[col] = label_encoders[col].transform([transaction[col]])[0]
    
    # Convert to NumPy array and scale
    input_data = np.array(list(transaction.values())).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    
    # Make a prediction
    prediction = model.predict(input_data_scaled)
    
    return "Fraudulent Transaction Detected!" if prediction[0] == 1 else "Legitimate Transaction"

# Example test case
new_transaction = {
    "Amount": 250.75,
    "Location": "New York",
    "Merchant": "Amazon",
    "CardType": "Visa"
}

print("Test Result:", test_new_transaction(new_transaction))
