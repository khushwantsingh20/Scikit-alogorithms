import joblib

# Load the saved model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example new email
new_email = ["Free Netflix for 6 months! Sign up now."]

# Convert the new email to TF-IDF features
new_email_tfidf = vectorizer.transform(new_email)

# Predict
prediction = model.predict(new_email_tfidf)
print("Prediction:", "Spam" if prediction[0] == 1 else "Not Spam")