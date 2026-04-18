import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

API_KEY = st.secrets["tmdb_api_key"]

def fetch_poster(title):
    url = f"https://api.themoviedb.org/3/search/multi?api_key={API_KEY}&query={title}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    
    if data['results']:
        poster_path = data['results'][0].get('poster_path')
        
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
    
    return None
# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    
    df = df[['title', 'type', 'listed_in', 'description', 'country']]
    df = df.dropna()
    
    df['title'] = df['title'].str.lower()
    df['content'] = df['listed_in'] + ' ' + df['description']
    
    return df.reset_index(drop=True)

df = load_data()

# -----------------------
# Similarity
# -----------------------
@st.cache_resource
def compute_similarity(data):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(data['content'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_similarity(df)

# -----------------------
# Recommendation Function
# -----------------------
def recommend(title, types=None, countries=None, n=5):
    title = title.lower()
    
    if title not in df['title'].values:
        return None
    
    idx = df[df['title'] == title].index[0]
    
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:]
    
    results = []
    
    for i in scores:
        row = df.iloc[i[0]]
        
        if types and row['type'] not in types:
            continue
        
        if countries and row['country'] not in countries:
            continue
        
        poster = fetch_poster(row['title'])
        
        results.append({
            "title": row['title'].title(),
            "type": row['type'],
            "poster": poster
        })
        
        if len(results) >= n:
            break
    
    return results

# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="Netflix Recommender", layout="centered")

st.title("Netflix Recommendation System")
st.write("Get recommendations with smart filters!")

# Title input
title_input = st.selectbox(
    "Choose a title:",
    df['title'].str.title().unique()
)

# Filters section
st.subheader("Filters (Optional)")

use_filters = st.checkbox("Enable filters")

selected_types = None
selected_countries = None

if use_filters:
    selected_types = st.multiselect(
        "Select Content Type:",
        df['type'].unique()
    )
    
    selected_countries = st.multiselect(
        "Select Country:",
        df['country'].value_counts().head(10).index
    )

# Button
if st.button("Recommend"):
    results = recommend(
        title_input,
        types=selected_types if selected_types else None,
        countries=selected_countries if selected_countries else None
    )
    
    if results:
        st.subheader("Recommendations:")
        
        cols = st.columns(len(results))
        
        for i, rec in enumerate(results):
            with cols[i]:
                if rec['poster']:
                    st.image(rec['poster'])
                else:
                    st.write("No Image")
                
                st.caption(rec['title'])
                st.caption(rec['type'])
    else:
        st.warning("No results found.")
