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

import nltk

# Vérifie et télécharge 'punkt' si nécessaire
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configuration de la page
st.set_page_config(
    page_title="🎯 Analyseur de Sentiment Pro",
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
        """Télécharge les ressources NLTK nécessaires"""
        resources = ['vader_lexicon', 'punkt', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except:
                    pass
    
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
        vader_scores = self.sia.polarity_scores(text)
        
        # Analyse TextBlob
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
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
        """Analyse avec lexique personnalisé"""
        if not text:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'classification': 'Neutre'}
        
        words = word_tokenize(text)
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
        """Corrèle les termes avec les sentiments"""
        if not results:
            return pd.DataFrame()
        
        term_sentiment_data = []
        
        for result in results:
            words = word_tokenize(result['cleaned_text'])
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

# Initialisation de l'analyseur
@st.cache_resource
def get_analyzer():
    return RealTimeSentimentAnalyzer()

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
            result = analyzer.analyze_sentiment_comprehensive(text_input)
            display_single_analysis_result(result)

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
                        result = analyzer.analyze_sentiment_comprehensive(text)
                        results.append(result)
                        progress_bar.progress((i + 1) / len(texts))
                    
                    progress_bar.empty()
                    display_batch_analysis_results(results)
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
        while simulation_active:
            # Sélectionner un texte aléatoire
            random_text = np.random.choice(sample_texts)
            
            # Analyser
            result = analyzer.analyze_sentiment_comprehensive(random_text)
            st.session_state.realtime_results.append(result)
            
            # Garder seulement les 50 derniers résultats
            if len(st.session_state.realtime_results) > 50:
                st.session_state.realtime_results = st.session_state.realtime_results[-50:]
            
            # Afficher les résultats
            with results_container.container():
                display_realtime_results(st.session_state.realtime_results)
            
            time.sleep(3)

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
    
    # Analyse des mots-clés
    display_keyword_analysis(results)
    
    # Export des résultats
    st.subheader("💾 Export")
    if st.button("📥 Télécharger les résultats (JSON)"):
        json_data = json.dumps([
            {
                'text': r['text'],
                'sentiment': r['consensus']['classification'],
                'score': r['consensus']['score'],
                'confidence': r['consensus']['confidence'],
                'timestamp': r['timestamp'].isoformat()
            }
            for r in results
        ], ensure_ascii=False, indent=2)
        
        st.download_button(
            label="📥 Télécharger JSON",
            data=json_data,
            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

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
    
    # Graphique temps réel
    if len(results) > 1:
        df_time = pd.DataFrame([
            {
                'timestamp': r['timestamp'],
                'score': r['consensus']['score'],
                'classification': r['consensus']['classification']
            }
            for r in results
        ])
        
        fig = px.line(
            df_time, 
            x='timestamp', 
            y='score',
            color='classification',
            title="📈 Évolution du Sentiment en Temps Réel",
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Évolution temporelle si applicable
    if len(results) > 5:
        fig_time = px.scatter(
            df,
            x='timestamp',
            y='score',
            color='classification',
            size='confidence',
            title="⏰ Évolution Temporelle des Sentiments",
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_time, use_container_width=True)

def display_keyword_analysis(results):
    """Affiche l'analyse des mots-clés"""
    st.subheader("🔍 Analyse des Mots-Clés")
    
    # Extraire les textes
    texts = [r['text'] for r in results]
    
    # TF-IDF
    keywords = analyzer.extract_keywords_tfidf(texts, max_features=20)
    
    if keywords:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top mots-clés
            st.write("**🏆 Top Mots-Clés (TF-IDF)**")
            keywords_df = pd.DataFrame(
                list(keywords.items()), 
                columns=['Terme', 'Score TF-IDF']
            ).sort_values('Score TF-IDF', ascending=False).head(10)
            
            fig_bar = px.bar(
                keywords_df,
                x='Score TF-IDF',
                y='Terme',
                orientation='h',
                title="Importance des Termes"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Nuage de mots
            st.write("**☁️ Nuage de Mots**")
            try:
                wordcloud = WordCloud(
                    width=400, 
                    height=300, 
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(keywords)
                
                fig_cloud = go.Figure()
                # Convertir l'image en base64 pour Streamlit
                img_buffer = io.BytesIO()
                wordcloud.to_image().save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                st.markdown(
                    f'<img src="data:image/png;base64,{img_str}" style="width:100%">',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.write("Impossible de générer le nuage de mots")
    
    # Corrélation termes-sentiment
    correlation_df = analyzer.correlate_terms_sentiment(results)
    
    if not correlation_df.empty:
        st.write("**🔗 Corrélation Termes ↔ Sentiments**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Termes les plus positifs
            positive_terms = correlation_df[
                correlation_df['dominant_class'] == 'Positif'
            ].head(10)
            
            if not positive_terms.empty:
                fig_pos = px.bar(
                    positive_terms,
                    x='avg_sentiment',
                    y='term',
                    orientation='h',
                    title="🟢 Termes les Plus Positifs",
                    color='avg_sentiment',
                    color_continuous_scale='Greens'
                )
                st.plotly_chart(fig_pos, use_container_width=True)
        
        with col2:
            # Termes les plus négatifs
            negative_terms = correlation_df[
                correlation_df['dominant_class'] == 'Négatif'
            ].head(10)
            
            if not negative_terms.empty:
                fig_neg = px.bar(
                    negative_terms,
                    x='avg_sentiment',
                    y='term',
                    orientation='h',
                    title="🔴 Termes les Plus Négatifs",
                    color='avg_sentiment',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_neg, use_container_width=True)
        
        # Tableau complet des corrélations
        with st.expander("📊 Tableau Complet des Corrélations"):
            st.dataframe(
                correlation_df.style.format({
                    'avg_sentiment': '{:.3f}',
                    'frequency': '{:.0f}'
                }),
                use_container_width=True
            )

def create_twitter_simulator():
    """Simulateur de données Twitter (remplace l'API réelle)"""
    
    # Tweets d'exemple par catégorie
    tweet_templates = {
        'positive': [
            "Excellente expérience avec {brand}! Service client au top 👍 #satisfaction",
            "Je recommande vivement {brand}, qualité exceptionnelle! 🌟",
            "{brand} a dépassé mes attentes, bravo! 🎉 #happy",
            "Livraison ultra rapide de {brand}, parfait! ⚡ #efficient",
            "Interface {brand} très intuitive, j'adore! 💖 #userexperience"
        ],
        'negative': [
            "Très déçu de {brand}, service client inexistant 😠 #disappointed",
            "Problème récurrent avec {brand}, c'est inadmissible! 😤",
            "Qualité {brand} en chute libre, dommage... 📉 #quality",
            "Attente interminable chez {brand}, très frustrant! ⏰ #waiting",
            "Bug constant sur l'app {brand}, quand est-ce que ça sera réparé? 🐛"
        ],
        'neutral': [
            "Test en cours avec {brand}, on verra bien...",
            "{brand} a lancé une nouvelle fonctionnalité aujourd'hui",
            "Mise à jour {brand} disponible, quelqu'un l'a essayée?",
            "Comparaison entre {brand} et ses concurrents en cours",
            "Webinaire {brand} prévu la semaine prochaine 📅"
        ]
    }
    
    brands = ['TechCorp', 'ServicePlus', 'InnovateCo', 'QualityBrand', 'FastDelivery']
    
    def generate_tweet():
        """Génère un tweet aléatoire"""
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
        template = np.random.choice(tweet_templates[sentiment])
        brand = np.random.choice(brands)
        
        return template.format(brand=brand)
    
    return generate_tweet

def handle_twitter_analysis():
    """Interface pour l'analyse Twitter simulée"""
    st.subheader("🐦 Analyse Twitter (Simulation)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input(
            "Terme de recherche:",
            value="TechCorp",
            placeholder="Entrez une marque, hashtag ou mot-clé"
        )
    
    with col2:
        tweet_count = st.selectbox("Nombre de tweets", [10, 25, 50, 100], index=1)
    
    if st.button("🔍 Analyser les Tweets", type="primary"):
        if search_term:
            with st.spinner(f"Simulation de {tweet_count} tweets pour '{search_term}'..."):
                # Générer des tweets simulés
                tweet_generator = create_twitter_simulator()
                simulated_tweets = []
                
                progress_bar = st.progress(0)
                
                for i in range(tweet_count):
                    tweet = tweet_generator()
                    # Remplacer les marques par le terme recherché
                    for brand in ['TechCorp', 'ServicePlus', 'InnovateCo', 'QualityBrand', 'FastDelivery']:
                        tweet = tweet.replace(brand, search_term)
                    
                    result = analyzer.analyze_sentiment_comprehensive(tweet)
                    result['platform'] = 'Twitter'
                    result['search_term'] = search_term
                    simulated_tweets.append(result)
                    
                    progress_bar.progress((i + 1) / tweet_count)
                    time.sleep(0.1)  # Simulation du temps de traitement
                
                progress_bar.empty()
                
                # Afficher les résultats
                display_social_media_results(simulated_tweets, search_term)

def display_social_media_results(results, search_term):
    """Affiche les résultats d'analyse des réseaux sociaux"""
    st.success(f"✅ Analyse terminée pour '{search_term}' - {len(results)} messages analysés")
    
    # KPIs principaux
    classifications = [r['consensus']['classification'] for r in results]
    scores = [r['consensus']['score'] for r in results]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total", len(results))
    
    with col2:
        positive_count = classifications.count('Positif')
        positive_pct = positive_count / len(results) * 100
        st.metric(
            "🟢 Positifs", 
            f"{positive_count}",
            f"{positive_pct:.1f}%"
        )
    
    with col3:
        negative_count = classifications.count('Négatif')
        negative_pct = negative_count / len(results) * 100
        st.metric(
            "🔴 Négatifs", 
            f"{negative_count}",
            f"{negative_pct:.1f}%"
        )
    
    with col4:
        neutral_count = classifications.count('Neutre')
        neutral_pct = neutral_count / len(results) * 100
        st.metric(
            "🟡 Neutres", 
            f"{neutral_count}",
            f"{neutral_pct:.1f}%"
        )
    
    with col5:
        avg_score = np.mean(scores)
        st.metric(
            "📈 Score Moyen", 
            f"{avg_score:.3f}",
            f"{'📈' if avg_score > 0 else '📉' if avg_score < 0 else '➡️'}"
        )
    
    # Alertes intelligentes
    display_smart_alerts(results, search_term)
    
    # Graphiques avancés
    create_advanced_visualizations(results, search_term)
    
    # Timeline des sentiments
    create_sentiment_timeline(results)
    
    # Analyse des hashtags et mentions
    analyze_hashtags_mentions(results)
    
    # Recommandations
    generate_recommendations(results, search_term)

def display_smart_alerts(results, search_term):
    """Affiche des alertes intelligentes basées sur l'analyse"""
    st.subheader("🚨 Alertes Intelligentes")
    
    classifications = [r['consensus']['classification'] for r in results]
    negative_ratio = classifications.count('Négatif') / len(results)
    positive_ratio = classifications.count('Positif') / len(results)
    
    # Système d'alertes
    alerts = []
    
    if negative_ratio > 0.6:
        alerts.append({
            'type': 'error',
            'message': f"⚠️ ALERTE CRITIQUE: {negative_ratio:.1%} de sentiments négatifs pour '{search_term}'"
        })
    elif negative_ratio > 0.4:
        alerts.append({
            'type': 'warning',
            'message': f"⚠️ Attention: Proportion élevée de sentiments négatifs ({negative_ratio:.1%})"
        })
    
    if positive_ratio > 0.7:
        alerts.append({
            'type': 'success',
            'message': f"🎉 Excellente perception: {positive_ratio:.1%} de sentiments positifs!"
        })
    
    # Détection de mots critiques
    negative_results = [r for r in results if r['consensus']['classification'] == 'Négatif']
    if negative_results:
        critical_words = []
        for result in negative_results:
            words = result['cleaned_text'].split()
            critical_words.extend([w for w in words if w in analyzer.negative_words])
        
        if critical_words:
            most_common_critical = Counter(critical_words).most_common(3)
            alerts.append({
                'type': 'info',
                'message': f"🔍 Mots critiques fréquents: {', '.join([w[0] for w in most_common_critical])}"
            })
    
    # Afficher les alertes
    for alert in alerts:
        if alert['type'] == 'error':
            st.error(alert['message'])
        elif alert['type'] == 'warning':
            st.warning(alert['message'])
        elif alert['type'] == 'success':
            st.success(alert['message'])
        else:
            st.info(alert['message'])

def create_advanced_visualizations(results, search_term):
    """Crée des visualisations avancées"""
    st.subheader("📊 Analyses Avancées")
    
    # Préparer les données
    df = pd.DataFrame([
        {
            'text': r['text'],
            'classification': r['consensus']['classification'],
            'score': r['consensus']['score'],
            'confidence': r['consensus']['confidence'],
            'timestamp': r['timestamp'],
            'text_length': len(r['text']),
            'word_count': len(r['text'].split())
        }
        for r in results
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Relation longueur du texte vs sentiment
        fig_scatter = px.scatter(
            df,
            x='word_count',
            y='score',
            color='classification',
            size='confidence',
            title="📝 Longueur du Message vs Sentiment",
            labels={'word_count': 'Nombre de mots', 'score': 'Score de sentiment'},
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Distribution des scores par classification
        fig_violin = px.violin(
            df,
            y='classification',
            x='score',
            title="🎻 Distribution des Scores par Sentiment",
            color='classification',
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    # Heatmap de confiance
    confidence_sentiment = df.groupby(['classification', 'confidence']).size().reset_index(name='count')
    if not confidence_sentiment.empty:
        fig_heatmap = px.density_heatmap(
            df,
            x='classification',
            y='confidence',
            title="🔥 Heatmap Sentiment × Confiance",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

def create_sentiment_timeline(results):
    """Crée une timeline des sentiments"""
    st.subheader("⏰ Timeline des Sentiments")
    
    # Simuler des timestamps répartis sur la journée
    now = datetime.now()
    timestamps = [
        now - timedelta(hours=23-i, minutes=np.random.randint(0, 60))
        for i in range(len(results))
    ]
    
    # Mettre à jour les timestamps
    for i, result in enumerate(results):
        result['timestamp'] = timestamps[i]
    
    # Créer le DataFrame pour la timeline
    df_timeline = pd.DataFrame([
        {
            'timestamp': r['timestamp'],
            'score': r['consensus']['score'],
            'classification': r['consensus']['classification'],
            'text_preview': r['text'][:50] + "..." if len(r['text']) > 50 else r['text']
        }
        for r in results
    ])
    
    # Grouper par heure pour voir les tendances
    df_timeline['hour'] = df_timeline['timestamp'].dt.hour
    hourly_sentiment = df_timeline.groupby('hour').agg({
        'score': 'mean',
        'classification': lambda x: Counter(x).most_common(1)[0][0]
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution horaire
        fig_hourly = px.line(
            hourly_sentiment,
            x='hour',
            y='score',
            title="📈 Évolution Horaire du Sentiment",
            markers=True
        )
        fig_hourly.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_hourly.update_xaxis(title="Heure de la journée")
        fig_hourly.update_yaxis(title="Score moyen de sentiment")
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Distribution par heure
        hourly_counts = df_timeline.groupby(['hour', 'classification']).size().reset_index(name='count')
        fig_hourly_dist = px.bar(
            hourly_counts,
            x='hour',
            y='count',
            color='classification',
            title="📊 Volume de Messages par Heure",
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        st.plotly_chart(fig_hourly_dist, use_container_width=True)

def analyze_hashtags_mentions(results):
    """Analyse les hashtags et mentions"""
    st.subheader("🏷️ Analyse des Hashtags et Mentions")
    
    all_hashtags = []
    all_mentions = []
    
    for result in results:
        text = result['text']
        hashtags = re.findall(r'#\w+', text)
        mentions = re.findall(r'@\w+', text)
        
        for hashtag in hashtags:
            all_hashtags.append({
                'hashtag': hashtag,
                'sentiment': result['consensus']['classification'],
                'score': result['consensus']['score']
            })
        
        for mention in mentions:
            all_mentions.append({
                'mention': mention,
                'sentiment': result['consensus']['classification'],
                'score': result['consensus']['score']
            })
    
    col1, col2 = st.columns(2)
    
    with col1:
        if all_hashtags:
            st.write("**🏷️ Top Hashtags**")
            hashtag_df = pd.DataFrame(all_hashtags)
            hashtag_counts = hashtag_df['hashtag'].value_counts().head(10)
            
            fig_hashtags = px.bar(
                x=hashtag_counts.values,
                y=hashtag_counts.index,
                orientation='h',
                title="Hashtags les plus utilisés"
            )
            st.plotly_chart(fig_hashtags, use_container_width=True)
        else:
            st.info("Aucun hashtag trouvé dans les données")
    
    with col2:
        if all_mentions:
            st.write("**👤 Top Mentions**")
            mention_df = pd.DataFrame(all_mentions)
            mention_counts = mention_df['mention'].value_counts().head(10)
            
            fig_mentions = px.bar(
                x=mention_counts.values,
                y=mention_counts.index,
                orientation='h',
                title="Comptes les plus mentionnés"
            )
            st.plotly_chart(fig_mentions, use_container_width=True)
        else:
            st.info("Aucune mention trouvée dans les données")

def generate_recommendations(results, search_term):
    """Génère des recommandations basées sur l'analyse"""
    st.subheader("💡 Recommandations Stratégiques")
    
    classifications = [r['consensus']['classification'] for r in results]
    negative_ratio = classifications.count('Négatif') / len(results)
    positive_ratio = classifications.count('Positif') / len(results)
    
    recommendations = []
    
    # Recommandations basées sur le sentiment global
    if negative_ratio > 0.5:
        recommendations.extend([
            "🚨 **Action Urgente**: Mise en place d'une stratégie de gestion de crise",
            "🎯 **Communication**: Répondre publiquement aux préoccupations principales",
            "🔍 **Analyse**: Identifier les causes racines des sentiments négatifs",
            "📞 **Support**: Renforcer l'équipe de service client"
        ])
    elif negative_ratio > 0.3:
        recommendations.extend([
            "⚠️ **Surveillance**: Monitoring accru des mentions négatives",
            "💬 **Engagement**: Répondre de manière proactive aux critiques constructives",
            "📊 **Amélioration**: Analyser les retours pour améliorer le produit/service"
        ])
    
    if positive_ratio > 0.6:
        recommendations.extend([
            "🎉 **Amplification**: Partager et promouvoir les témoignages positifs",
            "🌟 **Témoignages**: Convertir les clients satisfaits en ambassadeurs",
            "📈 **Croissance**: Capitaliser sur la perception positive pour l'expansion"
        ])
    
    # Recommandations basées sur les mots-clés
    negative_results = [r for r in results if r['consensus']['classification'] == 'Négatif']
    if negative_results:
        # Analyser les problèmes récurrents
        negative_words = []
        for result in negative_results:
            words = result['cleaned_text'].split()
            negative_words.extend(words)
        
        common_issues = Counter(negative_words).most_common(5)
        if common_issues:
            recommendations.append(
                f"🔧 **Points d'amélioration prioritaires**: {', '.join([w[0] for w in common_issues[:3]])}"
            )
    
    # Recommandations temporelles
    now = datetime.now()
    recent_results = [r for r in results if (now - r['timestamp']).seconds < 3600]  # Dernière heure
    if recent_results:
        recent_negative = [r for r in recent_results if r['consensus']['classification'] == 'Négatif']
        if len(recent_negative) / len(recent_results) > 0.4:
            recommendations.append("⏰ **Tendance récente**: Augmentation des sentiments négatifs détectée")
    
    # Afficher les recommandations
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.info("Aucune recommandation spécifique. Continuez le monitoring régulier.")
    
    # Tableau de bord exécutif
    with st.expander("📋 Résumé Exécutif"):
        st.markdown(f"""
        **Analyse de '{search_term}'** - {len(results)} messages analysés
        
        **📊 Métriques Clés:**
        - Sentiment positif: {positive_ratio:.1%}
        - Sentiment négatif: {negative_ratio:.1%}
        - Score moyen: {np.mean([r['consensus']['score'] for r in results]):.3f}
        
        **🎯 Status Global:** {"🟢 Positif" if positive_ratio > negative_ratio * 1.5 else "🔴 Critique" if negative_ratio > 0.5 else "🟡 Neutre"}
        
        **⏰ Dernière mise à jour:** {datetime.now().strftime("%d/%m/%Y %H:%M")}
        """)

# Interface principale mise à jour
def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Analyseur de Sentiment Pro - Temps Réel</h1>
        <p>Analyse avancée de sentiment avec IA et visualisations interactives</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode d'analyse
        analysis_mode = st.selectbox(
            "Mode d'analyse",
            [
                "📝 Texte Manuel", 
                "📁 Fichier Upload", 
                "🐦 Twitter Simulation",
                "🔄 Temps Réel"
            ]
        )
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres Avancés"):
            st.session_state.auto_refresh = st.checkbox("Rafraîchissement automatique", value=False)
            st.session_state.refresh_interval = st.slider("Intervalle (secondes)", 5, 60, 10)
            st.session_state.max_results = st.slider("Nombre max de résultats", 10, 1000, 100)
            st.session_state.show_details = st.checkbox("Afficher les détails", value=True)
        
        # Informations système
        with st.expander("ℹ️ Informations"):
            st.markdown("""
            **🤖 Méthodes d'analyse:**
            - VADER (réseaux sociaux)
            - TextBlob (linguistique)
            - Lexique personnalisé
            
            **📊 Fonctionnalités:**
            - Analyse temps réel
            - Corrélation termes-sentiment
            - Alertes intelligentes
            - Export des données
            """)
    
    # Conteneur principal
    main_container = st.container()
    
    with main_container:
        if analysis_mode == "📝 Texte Manuel":
            handle_manual_text_analysis()
        elif analysis_mode == "📁 Fichier Upload":
            handle_file_upload_analysis()
        elif analysis_mode == "🐦 Twitter Simulation":
            handle_twitter_analysis()
        elif analysis_mode == "🔄 Temps Réel":
            handle_realtime_simulation()

if __name__ == "__main__":
    main()