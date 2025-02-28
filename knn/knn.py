import pandas as pd
import numpy as np
import ast  # To parse JSON-like strings
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Load the dataset
movies = pd.read_csv('tmdb_5000_movies.csv')

# Select relevant features
movies = movies[['id', 'title', 'genres', 'vote_average', 'popularity']]

# Drop missing values
movies.dropna(inplace=True)

# Convert titles to lowercase for uniformity
movies['title'] = movies['title'].str.lower()

# Function to extract genre names from JSON-like string
def extract_genres(genre_str):
    genres = ast.literal_eval(genre_str)
    return [genre['name'] for genre in genres]

# Apply function to extract genres
movies['genres'] = movies['genres'].apply(extract_genres)

# Convert genres to one-hot encoding
genres_df = movies['genres'].str.join('|').str.get_dummies()

# Merge genre data with movies
movies = pd.concat([movies, genres_df], axis=1)

# Drop original genres column
movies.drop(columns=['genres'], inplace=True)

# Normalize numerical features (vote_average, popularity)
scaler = MinMaxScaler()
movies[['vote_average', 'popularity']] = scaler.fit_transform(movies[['vote_average', 'popularity']])

# Define features for KNN
features = movies.drop(columns=['id', 'title'])

# Train the KNN model
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(features)

# Function to recommend movies
def recommend_movie(movie_name):
    movie_name = movie_name.lower().strip()  # Normalize input (lowercase + remove spaces)

    if movie_name not in movies['title'].values:
        print(f"❌ Movie '{movie_name}' not found in the database!")
        print("\n🔍 Available movie titles (first 10):")
        print(movies['title'].head(10).tolist())  # Show some available movies
        return []

    # Get the movie index
    movie_idx = movies[movies['title'] == movie_name].index[0]

    # Find nearest neighbors
    distances, indices = knn.kneighbors([features.iloc[movie_idx]])

    # Get recommended movie titles
    recommended_movies = movies.iloc[indices[0][1:]]['title'].values

    return recommended_movies

# Example: Recommend movies similar to 'John Carter'
print("Movies similar to 'John Carter':", recommend_movie('John Carter'))
