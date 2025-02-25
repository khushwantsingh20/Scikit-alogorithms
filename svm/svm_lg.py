import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.stem.wordnet import WordNetLemmatizer

sns.set_context('notebook')
sns.set_style('white')

# Load dataset
df = pd.read_csv("Training.txt", sep="\t", names=['liked', 'text'], encoding="utf-8")

# Preprocessing functions
def tokens(review):
    return TextBlob(review).words

def to_lemmas(review):
    wordss = TextBlob(review.lower()).words
    return [word.lemmatize() for word in wordss]

# Text preprocessing
bow_transformer = CountVectorizer(analyzer=to_lemmas).fit(df['text'])
review_bow = bow_transformer.transform(df['text'])

# TF-IDF transformation
tfidf_transformer = TfidfTransformer().fit(review_bow)
review_tfidf = tfidf_transformer.transform(review_bow)

# Split dataset
text_train, text_test, liked_train, liked_test = train_test_split(df['text'], df['liked'], test_size=0.2, random_state=42)

# Define pipeline
pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=to_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),
])

# Hyperparameter tuning
param_svm = [
  {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
  {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,
    param_grid=param_svm,
    refit=True,
    n_jobs=-1,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
)

# Train model
classifier = grid_svm.fit(text_train, liked_train)

# Model evaluation
print("Best Parameters:", classifier.best_params_)
print("\nClassification Report:\n", classification_report(liked_test, classifier.predict(text_test)))

# Test prediction
print(classifier.predict(["I am feeling very sad today."])[0])

