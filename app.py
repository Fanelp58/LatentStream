import streamlit as st
import pandas as pd
import numpy as np
import implicit
import scipy.sparse as sparse
import pickle
import time
import requests

# ==========================================
# 1. CONFIGURATION & DESIGN (CSS)
# ==========================================
st.set_page_config(page_title="LatentStream Dashboard", layout="wide", page_icon="🎬")

# CSS pour le look "Netflix / Dark Mode" et les Cartes
st.markdown("""
<style>
    /* Fond global */
    .stApp {
        background-color: #141414;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 1px solid #333;
    }
    
    /* Titres */
    h1, h2, h3 {
        color: #E50914 !important; /* Netflix Red */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Métriques dans la sidebar */
    div[data-testid="stMetricValue"] {
        color: #E50914 !important;
        font-size: 24px !important;
    }
    
    /* Boutons */
    .stButton>button {
        background-color: #E50914;
        color: white;
        border-radius: 4px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff0f1f;
        color: white;
    }
    
    /* Cartes de films (Design Custom) */
    .movie-card {
        background-color: #1f1f1f;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 20px;
        transition: transform 0.3s;
        text-align: center;
        height: 100%;
        border: 1px solid #333;
    }
    .movie-card:hover {
        transform: scale(1.05);
        border-color: #E50914;
    }
    .movie-title {
        font-size: 16px;
        font-weight: bold;
        margin-top: 10px;
        height: 40px;
        overflow: hidden;
        color: white;
    }
    .movie-genre {
        font-size: 12px;
        color: #888;
        margin-bottom: 5px;
    }
    .match-score {
        color: #46d369; /* Vert Netflix match */
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Input fields dark */
    .stSelectbox > div > div {
        background-color: #333;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CHARGEMENT DES DONNEES & MODELE
# ==========================================
@st.cache_resource
def load_engine():
    # Chargement des métadonnées
    with open("als_metadata.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Chargement du modèle ALS
    model = implicit.als.AlternatingLeastSquares(factors=64) # Doit matcher ton entraînement
    model = model.load("als_model.npz")
    
    return model, data

try:
    model, metadata = load_engine()
    movies_df = metadata['movies_df']
    movie_to_idx = metadata['movie_to_idx']
    idx_to_movie = metadata['idx_to_movie']
    sparsity = metadata.get('sparsity', 0.9553) # Fallback si pas calculé
except Exception as e:
    st.error(f"Erreur de chargement : {e}. Avez-vous bien généré les fichiers .pkl et .npz ?")
    st.stop()

# ==========================================
# 3. FONCTIONS UTILITAIRES
# ==========================================

@st.cache_data(show_spinner=False) # Mémorise les résultats pour ne pas re-consommer l'API
def get_poster_url(title):
    # REMPLACE PAR TA PROPRE CLE APRES ACTIVATION PAR EMAIL
    api_key = st.secrets["OMDB_API_KEY"] 
    
    try:
        # 1. Nettoyage du titre (MovieLens met souvent l'année à la fin)
        clean_title = title.split('(')[0].strip() 
        
        # 2. Appel API avec un timeout pour ne pas ralentir l'interface
        url = f"http://www.omdbapi.com/?t={clean_title}&apikey={api_key}"
        r = requests.get(url, timeout=2)
        
        if r.status_code == 200:
            data = r.json()
            # 3. Vérification que le poster existe bien dans leur base
            if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
                return data['Poster']
    except Exception as e:
        # En cas d'erreur réseau, on ne bloque pas l'app
        pass
    
    # 4. Fallback : Un placeholder élégant aux couleurs de Netflix (Noir et Rouge)
    # On utilise un service comme 'placehold.co' qui est plus moderne
    return "https://placehold.co/300x450/141414/E50914?text=Latent+Stream"

def recommend_live(user_ratings, model, n=6):
    """Calcule le vecteur utilisateur à la volée (Instant Response)"""
    start_time = time.time()
    
    # Construction vecteur sparse
    movie_indices = [movie_to_idx[mid] for mid in user_ratings.keys() if mid in movie_to_idx]
    ratings_values = list(user_ratings.values())
    
    if not movie_indices:
        return [], 0.0

    # Création matrice (1, N_items)
    user_sparse = sparse.csr_matrix(
        (ratings_values, ([0]*len(movie_indices), movie_indices)),
        shape=(1, model.item_factors.shape[0])
    )
    
    # Inférence ALS (Projection mathématique)
    ids, scores = model.recommend(userid=0, user_items=user_sparse, N=n, recalculate_user=True)
    
    latency = (time.time() - start_time) * 1000 # ms
    
    results = []
    for idx, score in zip(ids, scores):
        real_id = idx_to_movie[idx]
        info = movies_df[movies_df['MovieID'] == real_id].iloc[0]
        results.append({
            'id': real_id,
            'title': info['Title'],
            'genre': info['Genres'].split('|')[0], # Premier genre principal
            'score': score
        })
        
    return results, latency

# ==========================================
# 4. INTERFACE UTILISATEUR
# ==========================================

# --- SIDEBAR (Tableau de Bord Technique) ---
with st.sidebar:
    st.title("LatentStream Dashboard")
    st.caption("The Live Movie Recommender")
    st.markdown("---")
    
    # KPIs
    st.markdown("### ⚡ Inference Latency")
    latency_placeholder = st.empty()
    latency_placeholder.metric(label="", value="0.00 ms")
    st.progress(0, text="Goal: < 100ms")
    
    st.markdown("### 🗄️ Sparsity Rate")
    st.metric(label="MovieLens 1M", value=f"{sparsity:.4%}")
    
    st.markdown("### 🎯 Model Params")
    st.info(f"Factors: {model.factors} \n\n Iterations: {model.iterations}")
    
    st.markdown("---")
    st.markdown("""
    **Projection Info**
    
    New user vectors are calculated in real-time by projecting ratings onto the fixed item latent space ($Q$) using a least squares minimization:
    
    $P_{new} = (Q^T Q + \lambda I)^{-1} Q^T R_{new}$
    """)

    st.markdown("---")
    st.markdown("""
    **Powered with Streamlit by**
    
    HOUEHA Fanès | _Data Scientist and Engineer in Statistics and Economics_
    """)

# --- MAIN PAGE ---
col1, col2 = st.columns([2, 1])
with col1:
    st.title("Live Evaluation of Cold Start Strategy")
    st.markdown("""
    Rate movies to dynamically update your latent user vector. 
    The system uses **real-time Matrix Factorization projection** to adjust recommendations instantly.
    """)

# Gestion de la session (Panier de notes)
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {} # {MovieID: Rating}

# --- ZONE DE SAISIE ---
st.markdown("### 🔍 Select & Rate")
col_input1, col_input2, col_btn = st.columns([3, 2, 1])

with col_input1:
    # Liste déroulante avec recherche
    movie_options = dict(zip(movies_df['Title'], movies_df['MovieID']))
    selected_movie_name = st.selectbox("Search a movie", options=movie_options.keys(), index=None, placeholder="Ex: Toy Story...")

with col_input2:
    # Slider étoiles
    rating = st.slider("Your Rating", 1, 5, 3)

with col_btn:
    st.write("") # Spacer
    st.write("") 
    if st.button("➕ Add"):
        if selected_movie_name:
            movie_id = movie_options[selected_movie_name]
            st.session_state.user_ratings[movie_id] = rating
            st.toast(f"Added: {selected_movie_name} ({rating}★)", icon="✅")

# --- AFFICHAGE DES NOTES ACTUELLES ---
if st.session_state.user_ratings:
    st.markdown("### Your Ratings")
    
    # Affichage sous forme de tags
    cols = st.columns(len(st.session_state.user_ratings) + 1)
    for i, (mid, rat) in enumerate(st.session_state.user_ratings.items()):
        title = movies_df[movies_df['MovieID'] == mid]['Title'].values[0]
        st.info(f"{title}: {rat}★")
    
    if st.button("🗑️ Clear All"):
        st.session_state.user_ratings = {}
        st.rerun()

st.markdown("---")

# --- BOUTON D'ACTION & RESULTATS ---
if st.button("🚀 PREDICT FAVORITES", use_container_width=True, type="primary"):
    
    if not st.session_state.user_ratings:
        st.warning("Please rate at least one movie to initialize the Cold Start strategy.")
    else:
        with st.spinner("Projecting user vector into latent space..."):
            recommendations, lat = recommend_live(st.session_state.user_ratings, model)
            
            # Mise à jour KPI Latence Sidebar
            latency_placeholder.metric(label="", value=f"{lat:.2f} ms")
            
            # Affichage "Top Picks"
            st.markdown("## Top Picks for You")
            
            # Grille de résultats (2 lignes de 3 colonnes)
            cols = st.columns(3)
            for i, movie in enumerate(recommendations):
                with cols[i % 3]:
                    poster = get_poster_url(movie['title'])
                    
                    # HTML Card Injection
                    html_code = f"""
                    <div class="movie-card">
                        <div style="position: absolute; top: 10px; left: 10px; background-color: #E50914; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">
                            #{i+1}
                        </div>
                        <img src="{poster}" style="width: 100%; height: 250px; object-fit: cover; border-radius: 4px;">
                        <div class="movie-title">{movie['title']}</div>
                        <div class="movie-genre">{movie['genre']}</div>
                        <div class="match-score">Match {int(movie['score']*100)}%</div>
                    </div>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
                
                if (i + 1) % 3 == 0:
                    st.write("") # Spacer entre les lignes
