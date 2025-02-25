import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

# ✅ Step 1: Load dataset
df = pd.read_csv("sentiment_data.csv")

# ✅ Step 2: Preprocess text (handle negations)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"n't", " not", text)  # Normalize negations
    words = text.split()
    
    negation_words = {"not", "no", "never", "lie", "isn’t", "wasn’t", "doesn’t"}
    processed_text = []
    i = 0
    
    while i < len(words):
        if words[i] in negation_words and i + 1 < len(words):
            processed_text.append(words[i] + "_" + words[i+1])  # Merge negation with next word
            i += 2  # Skip next word
        else:
            processed_text.append(words[i])
            i += 1
            
    return " ".join(processed_text)

df["text"] = df["text"].apply(preprocess_text)

# ✅ Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(df["text"])
y = df["sentiment"]

# ✅ Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 5: Optimize SVM hyperparameters
param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring="accuracy", verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# ✅ Step 6: Train final model with best params
best_params = grid_search.best_params_
svm_model = SVC(C=best_params["C"], kernel=best_params["kernel"])
svm_model.fit(X_train, y_train)

# ✅ Step 7: Evaluate model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ✅ Step 8: Predict sentiment for new text
def predict_sentiment(text):
    text = preprocess_text(text)
    text_transformed = vectorizer.transform([text])
    prediction = svm_model.predict(text_transformed)
    return "Positive" if prediction[0] == 1 else "Negative"

# Test with a complex sentence
sample_text = "I love this amazing place is lie I hate it"
print(f"\nSentiment for '{sample_text}': {predict_sentiment(sample_text)}")

