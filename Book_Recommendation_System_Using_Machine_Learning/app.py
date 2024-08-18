import requests
import pickle
import streamlit as st
import numpy as np

st.header('Book Recommender System Using Machine Learning')
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/books_name.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))


def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    for book_id in suggestion[0]:
        book_name.append(book_pivot.index[book_id])

    for name in book_name:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        try:
            url = final_rating.iloc[idx]['img_url']
            response = requests.get(url)
            if response.status_code == 200:
                poster_url.append(url)
            else:
                st.warning(
                    f"Failed to load image for book index {idx}: Status Code {response.status_code}")
                # Using a placeholder image
                poster_url.append("https://via.placeholder.com/150")
        except Exception as e:
            st.warning(f"Failed to load image for book index {idx}: {e}")
            # Using a placeholder image
            poster_url.append("https://via.placeholder.com/150")

    return poster_url


def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion[0])):
        books = book_pivot.index[suggestion[0][i]]
        books_list.append(books)
    return books_list, poster_url


selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)
    if len(recommended_books) > 1:
        cols = st.columns(5)
        for i in range(1, 6):
            with cols[i-1]:
                st.text(recommended_books[i])
                st.image(poster_url[i], width=150)
    else:
        st.warning("No recommendations found.")
