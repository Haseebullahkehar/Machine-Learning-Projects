import pickle
import streamlit as st
import requests

# Function to get movie recommendations from TMDB API
def get_movie_recommendations(api_key, movie_name):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}"
    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            movie_id = data['results'][0]['id']
            recommendations_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={api_key}"
            rec_response = requests.get(recommendations_url)
            if rec_response.status_code == 200:
                return rec_response.json()['results']
            else:
                st.error("Failed to fetch recommendations")
        else:
            st.error("Movie not found")
    else:
        st.error("Failed to fetch movie data")
    return []

# API key
api_key = '232ea3f7b88d22468755a29b84d8b065'

st.header("Movies Recommendation System Using Machine Learning- by Haseeb")

# Corrected pickle loading
with open('artifacts/movie_list.pkl', 'rb') as f:
    movies = pickle.load(f)

with open('artifacts/similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

movie_list = movies['title'].values

selected_movie = st.selectbox(
    'Type or select a movie to get recommendation',
    movie_list
)

# Function to handle recommendations
def recommend(movie_name):
    recommendations = get_movie_recommendations(api_key, movie_name)
    recommended_movies_name = [movie['title'] for movie in recommendations[:5]]  # Limit to 5 movies
    recommended_movies_poster = [movie['poster_path'] for movie in recommendations[:5]]  # Limit to 5 movies
    return recommended_movies_name, recommended_movies_poster

if st.button('Show Recommendation'):
    recommended_movies_name, recommended_movies_poster = recommend(selected_movie)
    st.subheader("Recommended Movies:")

    # Create a single row with columns for each recommended movie
    cols = st.columns(5)  # Create 5 columns

    for col, name, poster in zip(cols, recommended_movies_name, recommended_movies_poster):
        with col:
            st.image(f"https://image.tmdb.org/t/p/w500{poster}", width=150)  # Set image width
            st.write(name)
