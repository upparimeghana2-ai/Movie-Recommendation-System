import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0f0f0f;
        color: white;
    }

    h1 {
        color: #E50914;
        text-align: center;
    }

    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 8px;
        height: 45px;
        width: 150px;
        font-size: 16px;
    }

    .stSelectbox label {
        color: white;
        font-size: 18px;
    }

    .stMarkdown {
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# API KEY (paste your TMDB key)
# -------------------------------
API_KEY = "e348f0913e92c09167feaa141d7d8dba"


# -------------------------------
# Load datasets
# -------------------------------
movies = pd.read_csv(
    "u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

ratings = pd.read_csv(
    "u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)


# -------------------------------
# Create user-movie matrix
# -------------------------------
movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
)

movie_matrix = movie_matrix.fillna(0)


# -------------------------------
# Cosine similarity
# -------------------------------
similarity = cosine_similarity(movie_matrix.T)


similarity_df = pd.DataFrame(
    similarity,
    index=movie_matrix.columns,
    columns=movie_matrix.columns
)


# -------------------------------
# Poster fetch function
# -------------------------------
import requests

def fetch_poster(movie_name):

    try:
        # remove year
        movie_name = movie_name.split("(")[0]

        # fix title formats like "Rock, The"
        if "," in movie_name:
            parts = movie_name.split(",")
            movie_name = parts[1].strip() + " " + parts[0].strip()

        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"

        data = requests.get(url).json()

        if data["results"]:
            poster_path = data["results"][0]["poster_path"]

            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path

        return None

    except:
        return None

# -------------------------------
# Recommendation function
# -------------------------------
def recommend(movie_name):

    movie_id = movies[movies["title"] == movie_name]["movie_id"].values[0]

    similar_scores = list(enumerate(similarity_df[movie_id]))

    similar_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)

    recommended_movies = []

    for i in similar_scores[1:6]:

        recommended_id = similarity_df.index[i[0]]

        movie_title = movies[movies["movie_id"] == recommended_id]["title"].values

        if len(movie_title) > 0:
            recommended_movies.append(movie_title[0])

    return recommended_movies


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("🍿Movie Recommendation System")

movie_list = movies["title"].values

selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Recommend"):

    # create recommendations list
    recommendations = recommend(selected_movie)

    st.subheader("🎬 Recommended Movies")

    cols = st.columns(5)

    for i, movie in enumerate(recommendations):

        poster = fetch_poster(movie)

        with cols[i]:

            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.write("Poster not available")

            st.markdown(f"**{movie}**")