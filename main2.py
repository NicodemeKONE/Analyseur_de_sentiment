import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import re
import json
from datetime import datetime, timedelta
import time
import tweepy
from collections import Counter, defaultdict
import io
import base64
from typing import List, Dict, Any
import hashlib

# Configuration des ressources NLTK - VERSION CORRIGÉE
def setup_nltk_resources():
    """Configure et télécharge les ressources NLTK nécessaires"""
    resources_to_download = [
        'punkt_tab',  # Nouvelle version de punkt
        'punkt',      # Ancienne version pour compatibilité
        'vader_lexicon',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    
    for resource in resources_to_download:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                print(f"Téléchargement de {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Impossible de télécharger {resource}: {e}")
                continue

# Initialiser les ressources NLTK au démarrage
setup_nltk_resources()

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Sentiment Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .positive { color: #28a745; font-weight: bold; }
    .negative { color: #dc3545; font-weight: bold; }
    .neutral { color: #6c757d; font-weight: bold; }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

class RealTimeSentimentAnalyzer:
    def __init__(self):
        """Initialise l'analyseur avec tous les outils nécessaires"""
        self.init_nltk_resources()
        self.sia = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Stop words multilingues
        try:
            self.stop_words = set(
                stopwords.words('french') + 
                stopwords.words('english') + 
                stopwords.words('spanish')
            )
        except:
            self.stop_words = set()
        
        # Dictionnaires de mots personnalisés
        self.positive_words = {
            'excellent', 'fantastique', 'génial', 'parfait', 'superbe', 
            'incroyable', 'merveilleux', 'extraordinaire', 'magnifique',
            'amazing', 'awesome', 'brilliant', 'outstanding', 'wonderful',
            'love', 'best', 'great', 'good', 'nice', 'cool', 'happy'
        }
        
        self.negative_words = {
            'terrible', 'horrible', 'affreux', 'nul', 'catastrophique',
            'décevant', 'pathétique', 'minable', 'désastreux',
            'awful', 'terrible', 'horrible', 'disgusting', 'pathetic',
            'hate', 'worst', 'bad', 'sad', 'angry', 'disappointed'
        }
        
        # Cache pour améliorer les performances
        self.cache = {}
        
    def init_nltk_resources(self):
        """Télécharge les ressources NLTK nécessaires - VERSION AMÉLIORÉE"""
        pass  # Déjà fait dans setup_nltk_resources()
    
    def safe_tokenize(self, text: str) -> List[str]:
        """Tokenisation sécurisée avec fallback"""
        if not text:
            return []
        
        try:
            # Essayer d'abord avec NLTK
            return word_tokenize(text)
        except LookupError:
            # Fallback : tokenisation simple
            words = re.findall(r'\b\w+\b', text.lower())
            return words
        except Exception:
            # Fallback ultime : split simple
            return text.split()
    
    def preprocess_text(self, text: str) -> str:
        """Nettoie et préprocesse le texte"""
        if not text:
            return ""
        
        # Cache pour éviter le retraitement
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Convertir en minuscules
        text = text.lower()
        
        # Supprimer les URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Supprimer les emojis (basique)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Nettoyer les espaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Mettre en cache
        self.cache[text_hash] = text
        return text
    
    def analyze_sentiment_comprehensive(self, text: str) -> Dict[str, Any]:
        """Analyse complète du sentiment avec multiple méthodes"""
        if not text or text.strip() == '':
            return self._get_empty_result()
        
        cleaned_text = self.preprocess_text(text)
        
        # Analyse VADER
        try:
            vader_scores = self.sia.polarity_scores(text)
        except:
            vader_scores = {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
        
        # Analyse TextBlob
        try:
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
        except:
            textblob_polarity = 0
            textblob_subjectivity = 0
        
        # Analyse personnalisée
        custom_result = self._analyze_custom_lexicon(cleaned_text)
        
        # Score consensus (moyenne pondérée)
        consensus_score = (
            vader_scores['compound'] * 0.4 + 
            textblob_polarity * 0.4 + 
            custom_result['compound'] * 0.2
        )
        
        # Classification finale
        classification = self._classify_sentiment(consensus_score)
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'timestamp': datetime.now(),
            'vader': {
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'compound': vader_scores['compound']
            },
            'textblob': {
                'polarity': textblob_polarity,
                'subjectivity': textblob_subjectivity
            },
            'custom': custom_result,
            'consensus': {
                'score': round(consensus_score, 3),
                'classification': classification,
                'confidence': self._calculate_confidence([
                    self._classify_sentiment(vader_scores['compound']),
                    self._classify_sentiment(textblob_polarity),
                    custom_result['classification']
                ])
            }
        }
    
    def _analyze_custom_lexicon(self, text: str) -> Dict[str, Any]:
        """Analyse avec lexique personnalisé - VERSION SÉCURISÉE"""
        if not text:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'classification': 'Neutre'}
        
        # Utiliser la tokenisation sécurisée
        words = self.safe_tokenize(text)
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        if not words:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'classification': 'Neutre'}
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        pos_ratio = positive_count / len(words)
        neg_ratio = negative_count / len(words)
        neu_ratio = 1 - (pos_ratio + neg_ratio)
        compound = pos_ratio - neg_ratio
        
        return {
            'positive': pos_ratio,
            'negative': neg_ratio,
            'neutral': neu_ratio,
            'compound': compound,
            'classification': self._classify_sentiment(compound)
        }
    
    def _classify_sentiment(self, score: float) -> str:
        """Classifie le sentiment basé sur le score"""
        if score >= 0.1:
            return 'Positif'
        elif score <= -0.1:
            return 'Négatif'
        else:
            return 'Neutre'
    
    def _calculate_confidence(self, classifications: List[str]) -> str:
        """Calcule la confiance basée sur la cohérence"""
        counter = Counter(classifications)
        most_common_count = counter.most_common(1)[0][1]
        confidence = most_common_count / len(classifications)
        
        if confidence == 1.0:
            return "Très élevée"
        elif confidence >= 0.67:
            return "Élevée"
        else:
            return "Moyenne"
    
    def _get_empty_result(self):
        """Retourne un résultat vide"""
        return {
            'text': '',
            'cleaned_text': '',
            'timestamp': datetime.now(),
            'consensus': {'score': 0, 'classification': 'Neutre', 'confidence': 'Faible'}
        }
    
    def extract_keywords_tfidf(self, texts: List[str], max_features: int = 20) -> Dict[str, float]:
        """Extrait les mots-clés avec TF-IDF"""
        if not texts:
            return {}
        
        cleaned_texts = [self.preprocess_text(text) for text in texts]
        cleaned_texts = [text for text in cleaned_texts if text]
        
        if not cleaned_texts:
            return {}
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=list(self.stop_words),
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            return dict(zip(feature_names, scores))
        except:
            return {}
    
    def correlate_terms_sentiment(self, results: List[Dict]) -> pd.DataFrame:
        """Corrèle les termes avec les sentiments - VERSION SÉCURISÉE"""
        if not results:
            return pd.DataFrame()
        
        term_sentiment_data = []
        
        for result in results:
            words = self.safe_tokenize(result['cleaned_text'])
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            
            for word in words:
                term_sentiment_data.append({
                    'term': word,
                    'sentiment_score': result['consensus']['score'],
                    'classification': result['consensus']['classification']
                })
        
        if not term_sentiment_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(term_sentiment_data)
        
        # Agrégation par terme
        correlation_df = df.groupby('term').agg({
            'sentiment_score': ['mean', 'count'],
            'classification': lambda x: Counter(x).most_common(1)[0][0]
        }).round(3)
        
        correlation_df.columns = ['avg_sentiment', 'frequency', 'dominant_class']
        correlation_df = correlation_df.reset_index()
        correlation_df = correlation_df[correlation_df['frequency'] >= 2]  # Filtre fréquence minimale
        correlation_df = correlation_df.sort_values('frequency', ascending=False)
        
        return correlation_df

# Initialisation de l'analyseur avec gestion d'erreur
@st.cache_resource
def get_analyzer():
    try:
        return RealTimeSentimentAnalyzer()
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de l'analyseur: {e}")
        st.stop()

analyzer = get_analyzer()

# Interface utilisateur principale
def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Analyseur de Sentiment en Temps Réel</h1>
        <p>Analyse avancée de sentiment avec visualisations interactives</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode d'analyse
        analysis_mode = st.selectbox(
            "Mode d'analyse",
            ["📝 Texte Manuel", "📁 Fichier Upload", "🔄 Simulation Temps Réel"]
        )
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres Avancés"):
            auto_refresh = st.checkbox("Rafraîchissement automatique", value=False)
            refresh_interval = st.slider("Intervalle (secondes)", 5, 60, 10)
            max_results = st.slider("Nombre max de résultats", 10, 1000, 100)
            confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.6)
    
    # Conteneur principal
    main_container = st.container()
    
    with main_container:
        if analysis_mode == "📝 Texte Manuel":
            handle_manual_text_analysis()
        elif analysis_mode == "📁 Fichier Upload":
            handle_file_upload_analysis()
        elif analysis_mode == "🔄 Simulation Temps Réel":
            handle_realtime_simulation()

def handle_manual_text_analysis():
    """Interface pour l'analyse de texte manuel"""
    st.subheader("📝 Analyse de Texte Manuel")
    
    # Zone de saisie
    text_input = st.text_area(
        "Entrez votre texte à analyser:",
        height=100,
        placeholder="Exemple: Ce produit est absolument fantastique ! Je le recommande vivement."
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        analyze_btn = st.button("🔍 Analyser", type="primary")
    
    with col2:
        if st.button("🧹 Effacer"):
            st.rerun()
    
    if analyze_btn and text_input:
        with st.spinner("Analyse en cours..."):
            try:
                result = analyzer.analyze_sentiment_comprehensive(text_input)
                display_single_analysis_result(result)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {e}")

def handle_file_upload_analysis():
    """Interface pour l'analyse de fichiers"""
    st.subheader("📁 Analyse de Fichier")
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier texte",
        type=['txt', 'csv'],
        help="Formats supportés: TXT (une ligne par texte), CSV (colonne 'text')"
    )
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            if uploaded_file.type == "text/plain":
                texts = str(uploaded_file.read(), "utf-8").split('\n')
                texts = [text.strip() for text in texts if text.strip()]
            else:  # CSV
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts = df['text'].dropna().tolist()
                else:
                    st.error("Le fichier CSV doit contenir une colonne 'text'")
                    return
            
            if texts:
                with st.spinner(f"Analyse de {len(texts)} textes..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, text in enumerate(texts):
                        try:
                            result = analyzer.analyze_sentiment_comprehensive(text)
                            results.append(result)
                        except Exception as e:
                            st.warning(f"Erreur lors de l'analyse du texte {i+1}: {e}")
                            continue
                        progress_bar.progress((i + 1) / len(texts))
                    
                    progress_bar.empty()
                    
                    if results:
                        display_batch_analysis_results(results)
                    else:
                        st.error("Aucun texte n'a pu être analysé")
            else:
                st.warning("Aucun texte trouvé dans le fichier")
                
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {str(e)}")

def handle_realtime_simulation():
    """Interface pour la simulation temps réel"""
    st.subheader("🔄 Simulation Temps Réel")
    
    # Textes d'exemple pour simulation
    sample_texts = [
        "Ce produit est absolument fantastique ! Je le recommande.",
        "Service client décevant, très lent à répondre.",
        "Qualité correcte, rien d'exceptionnel mais acceptable.",
        "Livraison rapide et produit conforme à la description.",
        "Prix un peu élevé mais la qualité est au rendez-vous.",
        "Interface utilisateur intuitive et facile à utiliser.",
        "Problème technique récurrent, très frustrant.",
        "Excellent rapport qualité-prix, je recommande !",
        "Fonctionnalités limitées par rapport à la concurrence.",
        "Support technique réactif et professionnel."
    ]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        simulation_active = st.checkbox("🔴 Activer la simulation")
    
    with col2:
        if simulation_active:
            st.info("Simulation active - Mise à jour toutes les 3 secondes")
    
    # Conteneur pour les résultats temps réel
    results_container = st.empty()
    
    if simulation_active:
        # Initialiser session state pour les résultats
        if 'realtime_results' not in st.session_state:
            st.session_state.realtime_results = []
        
        # Simulation temps réel
        for _ in range(10):  # Limiter à 10 itérations pour éviter la boucle infinie
            # Sélectionner un texte aléatoire
            random_text = np.random.choice(sample_texts)
            
            # Analyser
            try:
                result = analyzer.analyze_sentiment_comprehensive(random_text)
                st.session_state.realtime_results.append(result)
                
                # Garder seulement les 50 derniers résultats
                if len(st.session_state.realtime_results) > 50:
                    st.session_state.realtime_results = st.session_state.realtime_results[-50:]
                
                # Afficher les résultats
                with results_container.container():
                    display_realtime_results(st.session_state.realtime_results)
                
                time.sleep(3)
            except Exception as e:
                st.error(f"Erreur lors de la simulation: {e}")
                break

def display_single_analysis_result(result):
    """Affiche le résultat d'une analyse unique"""
    consensus = result['consensus']
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_color = "🟢" if consensus['classification'] == 'Positif' else "🔴" if consensus['classification'] == 'Négatif' else "🟡"
        st.metric("Sentiment", f"{sentiment_color} {consensus['classification']}")
    
    with col2:
        st.metric("Score", f"{consensus['score']:.3f}")
    
    with col3:
        st.metric("Confiance", consensus['confidence'])
    
    with col4:
        st.metric("Longueur", f"{len(result['text'])} chars")
    
    # Détails par méthode
    st.subheader("📊 Détails par Méthode")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**VADER**")
        st.write(f"Positif: {result['vader']['positive']:.2%}")
        st.write(f"Négatif: {result['vader']['negative']:.2%}")
        st.write(f"Neutre: {result['vader']['neutral']:.2%}")
    
    with col2:
        st.write("**TextBlob**")
        st.write(f"Polarité: {result['textblob']['polarity']:.3f}")
        st.write(f"Subjectivité: {result['textblob']['subjectivity']:.3f}")
    
    with col3:
        st.write("**Lexique Personnalisé**")
        st.write(f"Score: {result['custom']['compound']:.3f}")
        st.write(f"Classification: {result['custom']['classification']}")
    
    # Texte préprocessé
    with st.expander("🔍 Texte Préprocessé"):
        st.write(f"**Original:** {result['text']}")
        st.write(f"**Nettoyé:** {result['cleaned_text']}")

def display_batch_analysis_results(results):
    """Affiche les résultats d'analyse en lot"""
    if not results:
        st.warning("Aucun résultat à afficher")
        return
    
    # Statistiques globales
    st.subheader("📊 Statistiques Globales")
    
    classifications = [r['consensus']['classification'] for r in results]
    scores = [r['consensus']['score'] for r in results]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Analysés", len(results))
    
    with col2:
        positive_count = classifications.count('Positif')
        st.metric("Positifs", f"{positive_count} ({positive_count/len(results):.1%})")
    
    with col3:
        negative_count = classifications.count('Négatif')
        st.metric("Négatifs", f"{negative_count} ({negative_count/len(results):.1%})")
    
    with col4:
        neutral_count = classifications.count('Neutre')
        st.metric("Neutres", f"{neutral_count} ({neutral_count/len(results):.1%})")
    
    # Visualisations
    create_visualizations(results)
    
    # Tableau des résultats
    st.subheader("📋 Détails des Résultats")
    
    # Créer DataFrame pour affichage
    df_display = pd.DataFrame([
        {
            'Texte': r['text'][:100] + "..." if len(r['text']) > 100 else r['text'],
            'Sentiment': r['consensus']['classification'],
            'Score': r['consensus']['score'],
            'Confiance': r['consensus']['confidence'],
            'Timestamp': r['timestamp'].strftime("%H:%M:%S")
        }
        for r in results
    ])
    
    st.dataframe(df_display, use_container_width=True)

def display_realtime_results(results):
    """Affiche les résultats en temps réel"""
    if not results:
        st.info("En attente de données...")
        return
    
    # Métriques temps réel
    recent_results = results[-10:]  # 10 derniers
    classifications = [r['consensus']['classification'] for r in recent_results]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Analyses", len(results))
    
    with col2:
        positive_count = classifications.count('Positif')
        st.metric("Positifs (10 derniers)", positive_count)
    
    with col3:
        negative_count = classifications.count('Négatif')
        st.metric("Négatifs (10 derniers)", negative_count)
    
    with col4:
        if results:
            avg_score = np.mean([r['consensus']['score'] for r in recent_results])
            st.metric("Score Moyen", f"{avg_score:.3f}")
    
    # Derniers résultats
    st.subheader("🔄 Derniers Résultats")
    for i, result in enumerate(reversed(recent_results), 1):
        with st.expander(f"{i}. {result['consensus']['classification']} - {result['timestamp'].strftime('%H:%M:%S')}"):
            st.write(f"**Texte:** {result['text']}")
            st.write(f"**Score:** {result['consensus']['score']:.3f}")
            st.write(f"**Confiance:** {result['consensus']['confidence']}")

def create_visualizations(results):
    """Crée les visualisations interactives"""
    st.subheader("📈 Visualisations")
    
    # Préparer les données
    df = pd.DataFrame([
        {
            'classification': r['consensus']['classification'],
            'score': r['consensus']['score'],
            'confidence': r['consensus']['confidence'],
            'timestamp': r['timestamp']
        }
        for r in results
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des sentiments
        fig_pie = px.pie(
            df, 
            names='classification',
            title="🥧 Distribution des Sentiments",
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Histogramme des scores
        fig_hist = px.histogram(
            df,
            x='score',
            color='classification',
            title="📊 Distribution des Scores",
            nbins=20,
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()