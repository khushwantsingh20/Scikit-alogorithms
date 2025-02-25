import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Function to load images and extract HOG features
def load_images_from_folder(folder, label):
    images, labels = [], []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to fixed size
            features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
            images.append(features)
            labels.append(label)
    
    return images, labels

# Load dataset
cat_images, cat_labels = load_images_from_folder("dataset/cat", 0)  # 0 for cat
dog_images, dog_labels = load_images_from_folder("dataset/dog", 1)  # 1 for dog

# Combine data
X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  
    'kernel': ['linear', 'rbf', 'poly'],  
    'gamma': ['scale', 'auto']  
}

# Train SVM model with hyperparameter tuning
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_svm = grid_search.best_estimator_

# Save the trained model
joblib.dump(best_svm, "svm_cat_dog.pkl")

# Evaluate the model
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Function to predict a new image
def predict_image(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    features, _ = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    
    prediction = model.predict([features])
    return "Dog" if prediction[0] == 1 else "Cat"

# Load trained model and predict
svm_model = joblib.load("svm_cat_dog.pkl")
result = predict_image("test_image.jpg", svm_model)
print(f"Predicted: {result}")