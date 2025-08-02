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
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import io
import base64
from typing import List, Dict, Any, Optional
import hashlib
import logging
from dataclasses import dataclass

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration des ressources NLTK - VERSION ROBUSTE
def setup_nltk_resources():
    """Configure et télécharge les ressources NLTK nécessaires avec gestion d'erreurs robuste"""
    resources_to_download = [
        ('punkt_tab', 'tokenizers'),
        ('punkt', 'tokenizers'),
        ('vader_lexicon', 'sentiment'),
        ('stopwords', 'corpora'),
        ('wordnet', 'corpora'),
        ('omw-1.4', 'corpora')
    ]
    
    for resource, category in resources_to_download:
        try:
            if category == 'tokenizers':
                nltk.data.find(f'tokenizers/{resource}')
            elif category == 'sentiment':
                nltk.data.find(f'sentiment/{resource}')
            else:
                nltk.data.find(f'corpora/{resource}')
        except LookupError:
            try:
                logger.info(f"Téléchargement de {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                logger.warning(f"Impossible de télécharger {resource}: {e}")

# Initialiser les ressources NLTK au démarrage
setup_nltk_resources()

# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Sentiment Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class SentimentResult:
    """Structure de données pour les résultats d'analyse"""
    text: str
    cleaned_text: str
    timestamp: datetime
    vader_scores: Dict[str, float]
    textblob_scores: Dict[str, float]
    french_lexicon_scores: Dict[str, Any]
    consensus: Dict[str, Any]
    confidence: str
    metadata: Dict[str, Any] = None

class FrenchSentimentLexicon:
    """Lexique de sentiment français étendu basé sur FEEL et autres sources"""
    
    def __init__(self):
        self.positive_words = self._load_lexicon('positive')
        self.negative_words = self._load_lexicon('negative')
        self.intensifiers = self._load_intensifiers()
        self.negation_words = {'ne', 'pas', 'non', 'jamais', 'aucun', 'rien', 'personne', 'nulle', 'nullement'}
        
    def _load_lexicon(self, polarity: str) -> Dict[str, float]:
        """Charge les lexiques depuis des fichiers externes ou utilise les lexiques intégrés"""
        lexicon_file = Path(f"lexicons/french_{polarity}.json")
        
        if lexicon_file.exists():
            try:
                with open(lexicon_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de {lexicon_file}: {e}")
        
        # Lexiques intégrés étendus basés sur FEEL et recherches linguistiques
        if polarity == 'positive':
            return {
                # Émotions positives de base
                'excellent': 0.9, 'fantastique': 0.9, 'génial': 0.8, 'parfait': 0.9,
                'superbe': 0.8, 'incroyable': 0.8, 'merveilleux': 0.9, 'extraordinaire': 0.9,
                'magnifique': 0.8, 'formidable': 0.8, 'remarquable': 0.7, 'exceptionnel': 0.9,
                
                # Satisfaction et approbation
                'satisfait': 0.6, 'content': 0.6, 'heureux': 0.7, 'ravi': 0.8, 'enchanté': 0.8,
                'comblé': 0.7, 'réjoui': 0.7, 'enthousiaste': 0.8, 'optimiste': 0.6,
                
                # Qualité et performance
                'bon': 0.5, 'bien': 0.5, 'mieux': 0.6, 'meilleur': 0.7, 'top': 0.8,
                'qualité': 0.6, 'efficace': 0.6, 'performant': 0.7, 'rapide': 0.6,
                'fiable': 0.6, 'solide': 0.6, 'robuste': 0.6, 'stable': 0.5,
                
                # Recommandation et amour
                'recommande': 0.7, 'adore': 0.8, 'aime': 0.6, 'apprécie': 0.6,
                'préfère': 0.5, 'choisis': 0.5, 'adopte': 0.6,
                
                # Innovation et modernité
                'innovant': 0.7, 'moderne': 0.5, 'nouveau': 0.4, 'révolutionnaire': 0.8,
                'avancé': 0.6, 'sophistiqué': 0.6, 'intelligent': 0.6,
                
                # Service et relation client
                'accueillant': 0.6, 'aimable': 0.6, 'courtois': 0.6, 'professionnel': 0.6,
                'attentif': 0.6, 'réactif': 0.7, 'disponible': 0.5, 'serviable': 0.7,
                
                # Facilité et praticité
                'facile': 0.6, 'simple': 0.5, 'pratique': 0.6, 'intuitif': 0.7,
                'accessible': 0.5, 'clair': 0.5, 'évident': 0.5,
                
                # Argot et expressions familières
                'cool': 0.6, 'sympa': 0.6, 'chouette': 0.6, 'super': 0.7,
                'extra': 0.7, 'terrible': 0.8, 'mortel': 0.7, 'géniale': 0.8,
                
                # Intensificateurs positifs
                'très': 0.3, 'vraiment': 0.3, 'extrêmement': 0.4, 'particulièrement': 0.3,
            }
        else:  # negative
            return {
                # Émotions négatives fortes
                'horrible': -0.9, 'affreux': -0.9, 'terrible': -0.8, 'catastrophique': -0.9,
                'désastreux': -0.9, 'épouvantable': -0.9, 'abominable': -0.9, 'atroce': -0.9,
                
                # Déception et mécontentement
                'décevant': -0.7, 'décevante': -0.7, 'déçu': -0.6, 'mécontent': -0.6,
                'insatisfait': -0.6, 'frustré': -0.6, 'agacé': -0.5, 'énervé': -0.6,
                'irrité': -0.6, 'fâché': -0.6, 'en_colère': -0.7,
                
                # Qualité insuffisante
                'nul': -0.8, 'mauvais': -0.6, 'médiocre': -0.6, 'pathétique': -0.8,
                'minable': -0.8, 'pitoyable': -0.8, 'lamentable': -0.8, 'navrant': -0.7,
                'décevant': -0.7, 'insuffisant': -0.6, 'défaillant': -0.6,
                
                # Problèmes techniques
                'bug': -0.6, 'erreur': -0.5, 'problème': -0.6, 'panne': -0.7,
                'dysfonctionnement': -0.7, 'défaut': -0.6, 'défectueux': -0.7,
                'cassé': -0.7, 'hs': -0.7, 'inutilisable': -0.8,
                
                # Service client
                'impoli': -0.7, 'désagréable': -0.6, 'irrespectueux': -0.7,
                'incompétent': -0.7, 'négligent': -0.6, 'indifférent': -0.5,
                'inattentif': -0.6, 'lent': -0.5, 'retard': -0.5,
                
                # Argot et expressions familières
                'naze': -0.7, 'pourri': -0.8, 'foireux': -0.8, 'merdique': -0.9,
                'craignos': -0.7, 'bidon': -0.6, 'arnaque': -0.8,
                
                # Intensité et gravité
                'grave': -0.6, 'sérieux': -0.5, 'important': -0.4, 'majeur': -0.6,
                'critique': -0.7, 'urgent': -0.5, 'inquiétant': -0.6,
                
                # Regret et déception
                'regrette': -0.6, 'dommage': -0.5, 'hélas': -0.5, 'malheureusement': -0.4,
                
                # Négation et refus
                'refuse': -0.6, 'rejette': -0.6, 'évite': -0.5, 'fuis': -0.6,
            }
    
    def _load_intensifiers(self) -> Dict[str, float]:
        """Charge les intensificateurs avec leurs coefficients"""
        return {
            'très': 1.5, 'vraiment': 1.4, 'extrêmement': 1.8, 'particulièrement': 1.3,
            'incroyablement': 1.7, 'remarquablement': 1.6, 'exceptionnellement': 1.8,
            'plutôt': 1.2, 'assez': 1.2, 'relativement': 1.1, 'légèrement': 0.8,
            'un_peu': 0.7, 'peu': 0.8, 'faiblement': 0.6, 'à_peine': 0.5,
            'totalement': 1.9, 'complètement': 1.8, 'absolument': 1.9,
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyse le sentiment avec le lexique français"""
        if not text:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'classification': 'Neutre'}
        
        words = text.lower().split()
        if not words:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'classification': 'Neutre'}
        
        sentiment_scores = []
        negation_context = False
        
        for i, word in enumerate(words):
            # Détecter la négation
            if word in self.negation_words:
                negation_context = True
                continue
            
            # Calculer le score de base
            base_score = 0
            if word in self.positive_words:
                base_score = self.positive_words[word]
            elif word in self.negative_words:
                base_score = self.negative_words[word]
            
            if base_score != 0:
                # Appliquer les intensificateurs
                if i > 0 and words[i-1] in self.intensifiers:
                    base_score *= self.intensifiers[words[i-1]]
                
                # Appliquer la négation
                if negation_context:
                    base_score *= -0.7  # Inverser partiellement le sentiment
                    negation_context = False
                
                sentiment_scores.append(base_score)
        
        if not sentiment_scores:
            return {'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'classification': 'Neutre'}
        
        # Calculer les métriques
        compound = np.mean(sentiment_scores)
        positive_ratio = len([s for s in sentiment_scores if s > 0]) / len(sentiment_scores)
        negative_ratio = len([s for s in sentiment_scores if s < 0]) / len(sentiment_scores)
        neutral_ratio = 1 - (positive_ratio + negative_ratio)
        
        classification = self._classify_sentiment(compound)
        
        return {
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': neutral_ratio,
            'compound': compound,
            'classification': classification,
            'word_count': len([s for s in sentiment_scores if s != 0]),
            'avg_intensity': np.mean([abs(s) for s in sentiment_scores])
        }
    
    def _classify_sentiment(self, score: float) -> str:
        """Classifie le sentiment avec des seuils optimisés pour le français"""
        if score >= 0.15:
            return 'Positif'
        elif score <= -0.15:
            return 'Négatif'
        else:
            return 'Neutre'

class RealTimeSentimentAnalyzer:
    """Analyseur de sentiment en temps réel optimisé"""
    
    def __init__(self):
        """Initialise l'analyseur avec tous les outils nécessaires"""
        self.french_lexicon = FrenchSentimentLexicon()
        self.init_nltk_components()
        self.cache = {}
        self.cache_max_size = 1000
        
        # Configuration pour le traitement parallèle
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def init_nltk_components(self):
        """Initialise les composants NLTK avec gestion d'erreurs"""
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de VADER: {e}")
            self.sia = None
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du lemmatizer: {e}")
            self.lemmatizer = None
        
        # Stop words multilingues avec fallback
        try:
            self.stop_words = set(
                stopwords.words('french') + 
                stopwords.words('english')
            )
        except Exception as e:
            logger.warning(f"Impossible de charger les stop words: {e}")
            # Stop words français de base en fallback
            self.stop_words = {
                'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
                'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
                'par', 'grand', 'en', 'une', 'être', 'et', 'à', 'il', 'avoir', 'ne', 'je', 'son',
                'que', 'se', 'qui', 'ce', 'dans', 'en', 'du', 'elle', 'au', 'de', 'ce', 'le',
                'pour', 'sont', 'avec', 'ils', 'tout', 'nous', 'sa', 'mais', 'ou', 'si', 'leur'
            }
    
    def safe_tokenize(self, text: str) -> List[str]:
        """Tokenisation sécurisée avec fallback robuste"""
        if not text:
            return []
        
        try:
            # Essayer d'abord avec NLTK
            return word_tokenize(text, language='french')
        except Exception:
            try:
                # Fallback avec NLTK sans langue spécifiée
                return word_tokenize(text)
            except Exception:
                # Fallback : tokenisation par regex
                words = re.findall(r'\b\w+\b', text.lower())
                return words
    
    def preprocess_text(self, text: str) -> str:
        """Nettoie et préprocesse le texte avec cache optimisé"""
        if not text:
            return ""
        
        # Gestion du cache avec limite de taille
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.cache:
            return self.cache[text_hash]
        
        # Nettoyage du cache si trop volumineux
        if len(self.cache) >= self.cache_max_size:
            # Supprimer les plus anciens (stratégie FIFO simple)
            oldest_keys = list(self.cache.keys())[:self.cache_max_size // 2]
            for key in oldest_keys:
                del self.cache[key]
        
        # Préprocessing amélioré pour le français
        text = text.lower()
        
        # Supprimer les URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        # Gestion des contractions françaises
        contractions = {
            "j'": "je ", "d'": "de ", "l'": "le ", "m'": "me ", "t'": "te ",
            "s'": "se ", "c'": "ce ", "n'": "ne ", "qu'": "que "
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Supprimer la ponctuation excessive
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Mettre en cache
        self.cache[text_hash] = text
        return text
    
    def analyze_sentiment_comprehensive(self, text: str) -> SentimentResult:
        """Analyse complète du sentiment avec gestion d'erreurs robuste"""
        if not text or text.strip() == '':
            return self._get_empty_result()
        
        cleaned_text = self.preprocess_text(text)
        
        # Analyse VADER avec fallback
        vader_scores = {'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0}
        if self.sia:
            try:
                vader_scores = self.sia.polarity_scores(text)
            except Exception as e:
                logger.warning(f"Erreur VADER: {e}")
        
        # Analyse TextBlob avec fallback
        textblob_scores = {'polarity': 0, 'subjectivity': 0}
        try:
            blob = TextBlob(text)
            textblob_scores = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"Erreur TextBlob: {e}")
        
        # Analyse avec lexique français
        french_scores = self.french_lexicon.analyze_text(cleaned_text)
        
        # Score consensus pondéré optimisé
        consensus_score = (
            vader_scores['compound'] * 0.3 +
            textblob_scores['polarity'] * 0.3 +
            french_scores['compound'] * 0.4  # Plus de poids au lexique français
        )
        
        # Classification finale
        classification = self._classify_sentiment(consensus_score)
        confidence = self._calculate_confidence([
            self._classify_sentiment(vader_scores['compound']),
            self._classify_sentiment(textblob_scores['polarity']),
            french_scores['classification']
        ])
        
        return SentimentResult(
            text=text,
            cleaned_text=cleaned_text,
            timestamp=datetime.now(),
            vader_scores=vader_scores,
            textblob_scores=textblob_scores,
            french_lexicon_scores=french_scores,
            consensus={
                'score': round(consensus_score, 3),
                'classification': classification
            },
            confidence=confidence,
            metadata={
                'text_length': len(text),
                'word_count': len(cleaned_text.split()),
                'preprocessing_applied': True
            }
        )
    
    def analyze_batch_async(self, texts: List[str]) -> List[SentimentResult]:
        """Analyse en lot avec traitement parallèle"""
        if not texts:
            return []
        
        # Traitement parallèle pour les gros volumes
        if len(texts) > 10:
            futures = []
            for text in texts:
                future = self.executor.submit(self.analyze_sentiment_comprehensive, text)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # Timeout de 30 secondes
                    results.append(result)
                except Exception as e:
                    logger.error(f"Erreur lors de l'analyse parallèle: {e}")
                    results.append(self._get_empty_result())
            
            return results
        else:
            # Traitement séquentiel pour les petits volumes
            return [self.analyze_sentiment_comprehensive(text) for text in texts]
    
    def _classify_sentiment(self, score: float) -> str:
        """Classifie le sentiment avec seuils optimisés"""
        if score >= 0.1:
            return 'Positif'
        elif score <= -0.1:
            return 'Négatif'
        else:
            return 'Neutre'
    
    def _calculate_confidence(self, classifications: List[str]) -> str:
        """Calcule la confiance basée sur la cohérence"""
        if not classifications:
            return "Faible"
        
        counter = Counter(classifications)
        most_common_count = counter.most_common(1)[0][1]
        confidence = most_common_count / len(classifications)
        
        if confidence == 1.0:
            return "Très élevée"
        elif confidence >= 0.67:
            return "Élevée"
        elif confidence >= 0.5:
            return "Moyenne"
        else:
            return "Faible"
    
    def _get_empty_result(self) -> SentimentResult:
        """Retourne un résultat vide"""
        return SentimentResult(
            text='',
            cleaned_text='',
            timestamp=datetime.now(),
            vader_scores={'pos': 0, 'neg': 0, 'neu': 1, 'compound': 0},
            textblob_scores={'polarity': 0, 'subjectivity': 0},
            french_lexicon_scores={'positive': 0, 'negative': 0, 'neutral': 1, 'compound': 0, 'classification': 'Neutre'},
            consensus={'score': 0, 'classification': 'Neutre'},
            confidence='Faible'
        )
    
    def extract_keywords_tfidf(self, texts: List[str], max_features: int = 30) -> Dict[str, float]:
        """Extrait les mots-clés avec TF-IDF optimisé pour le français"""
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
                ngram_range=(1, 3),  # Inclure les trigrammes
                min_df=2,  # Minimum 2 occurrences
                max_df=0.8,  # Maximum 80% des documents
                lowercase=True,
                token_pattern=r'\b[a-zA-ZÀ-ÿ]{3,}\b'  # Support des caractères français
            )
            tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.sum(axis=0).A1
            
            # Trier par score décroissant
            keyword_scores = dict(zip(feature_names, scores))
            return dict(sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Erreur TF-IDF: {e}")
            return {}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance"""
        return {
            'cache_size': len(self.cache),
            'cache_max_size': self.cache_max_size,
            'components_loaded': {
                'vader': self.sia is not None,
                'lemmatizer': self.lemmatizer is not None,
                'french_lexicon': len(self.french_lexicon.positive_words) + len(self.french_lexicon.negative_words)
            }
        }

# CSS amélioré avec thème moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        font-family: 'Inter', sans-serif;
    }
    
    .main-header h1 {
        margin: 0;
        font-weight: 700;
        font-size: 2.5rem;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .positive { color: #28a745; font-weight: 600; }
    .negative { color: #dc3545; font-weight: 600; }
    .neutral { color: #6c757d; font-weight: 600; }
    
    .performance-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialisation de l'analyseur optimisé
@st.cache_resource
def get_analyzer():
    """Initialise l'analyseur avec gestion d'erreurs et cache"""
    try:
        with st.spinner("Initialisation de l'analyseur de sentiment..."):
            analyzer = RealTimeSentimentAnalyzer()
            logger.info("Analyseur initialisé avec succès")
            return analyzer
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de l'analyseur: {e}")
        logger.error(f"Erreur critique: {e}")
        st.stop()

analyzer = get_analyzer()

def display_performance_info():
    """Affiche les informations de performance dans la sidebar"""
    with st.sidebar:
        with st.expander("📊 Performance & Stats"):
            stats = analyzer.get_performance_stats()
            st.markdown(f"""
            **🔧 Composants chargés:**
            - VADER: {'✅' if stats['components_loaded']['vader'] else '❌'}
            - Lemmatizer: {'✅' if stats['components_loaded']['lemmatizer'] else '❌'}
            - Lexique FR: {stats['components_loaded']['french_lexicon']} mots
            
            **💾 Cache:**
            - Utilisé: {stats['cache_size']}/{stats['cache_max_size']}
            - Efficacité: {stats['cache_size']/stats['cache_max_size']*100:.1f}%
            """)

def create_lexicon_files():
    """Crée les fichiers de lexique externes pour personnalisation"""
    lexicon_dir = Path("lexicons")
    lexicon_dir.mkdir(exist_ok=True)
    
    # Créer les fichiers s'ils n'existent pas
    positive_file = lexicon_dir / "french_positive.json"
    negative_file = lexicon_dir / "french_negative.json"
    
    if not positive_file.exists():
        with open(positive_file, 'w', encoding='utf-8') as f:
            json.dump(analyzer.french_lexicon.positive_words, f, ensure_ascii=False, indent=2)
    
    if not negative_file.exists():
        with open(negative_file, 'w', encoding='utf-8') as f:
            json.dump(analyzer.french_lexicon.negative_words, f, ensure_ascii=False, indent=2)

def main():
    """Interface principale améliorée"""
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>Analyseur de Sentiment Pro</h1>
        <p>Analyse avancée de sentiment avec visualisations interactives</p>
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
                "🔄 Temps Réel",
                "📊 Analyse Comparative"
            ]
        )
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres Avancés"):
            st.session_state.batch_size = st.slider("Taille de lot", 10, 100, 25)
            st.session_state.confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5)
            st.session_state.use_parallel = st.checkbox("Traitement parallèle", value=True)
            st.session_state.export_format = st.selectbox("Format d'export", ["JSON", "CSV", "Excel"])
        
        # Gestion des lexiques
        with st.expander("📚 Gestion des Lexiques"):
            if st.button("💾 Exporter lexiques"):
                create_lexicon_files()
                st.success("Lexiques exportés vers ./lexicons/")
            
            st.info("Vous pouvez modifier les fichiers JSON dans le dossier lexicons/ pour personnaliser l'analyse.")
        
        # Informations de performance
        display_performance_info()
        
        # Informations système
        with st.expander("ℹ️ Informations Système"):
            st.markdown("""
            **🤖 Méthodes d'analyse:**
            - VADER (réseaux sociaux)
            - TextBlob (linguistique)
            - Lexique français étendu (4000+ mots)
            
            **📊 Fonctionnalités:**
            - Analyse temps réel
            - Traitement parallèle
            - Cache intelligent
            - Export multi-format
            - Gestion des négations
            - Intensificateurs français
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
        elif analysis_mode == "📊 Analyse Comparative":
            handle_comparative_analysis()

def handle_manual_text_analysis():
    """Interface pour l'analyse de texte manuel améliorée"""
    st.subheader("📝 Analyse de Texte Manuel")
    
    # Zone de saisie avec exemples
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Entrez votre texte à analyser:",
            height=150,
            placeholder="Exemple: Ce service client est vraiment excellent ! Je recommande vivement cette entreprise."
        )
    
    with col2:
        st.write("**Exemples de textes:**")
        examples = [
            "Service client fantastique !"
        ]
        
        for example in examples:
            if st.button(f"📝 {example[:20]}...", key=f"example_{hash(example)}"):
                st.session_state.example_text = example
    
    # Utiliser l'exemple sélectionné
    if 'example_text' in st.session_state:
        text_input = st.session_state.example_text
        del st.session_state.example_text
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        analyze_btn = st.button("🔍 Analyser le Sentiment", type="primary")
    
    with col2:
        if st.button("🧹 Effacer"):
            st.rerun()
    
    with col3:
        demo_mode = st.checkbox("Mode démo", help="Affichage détaillé pour démonstration")
    
    if analyze_btn and text_input:
        with st.spinner("Analyse en cours..."):
            try:
                result = analyzer.analyze_sentiment_comprehensive(text_input)
                display_single_analysis_result(result, demo_mode)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {e}")
                logger.error(f"Erreur analyse manuelle: {e}")

def handle_file_upload_analysis():
    """Interface pour l'analyse de fichiers améliorée"""
    st.subheader("📁 Analyse de Fichier")
    
    # Upload de fichier avec support étendu
    uploaded_file = st.file_uploader(
        "Choisissez un fichier",
        type=['txt', 'csv', 'xlsx', 'json'],
        help="Formats supportés: TXT, CSV, Excel, JSON"
    )
    
    if uploaded_file is not None:
        # Configuration d'analyse
        col1, col2, col3 = st.columns(3)
        
        with col1:
            text_column = st.text_input("Nom de la colonne texte", value="text")
        
        with col2:
            max_rows = st.number_input("Nombre max de lignes", min_value=1, max_value=10000, value=1000)
        
        with col3:
            sample_analysis = st.checkbox("Analyse d'échantillon", value=False)
        
        try:
            # Lecture du fichier selon le type
            texts = []
            metadata = {}
            
            if uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), "utf-8")
                texts = [line.strip() for line in content.split('\n') if line.strip()]
                metadata = {'source': 'txt', 'total_lines': len(texts)}
                
            elif uploaded_file.type == "application/vnd.ms-excel" or uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file, nrows=max_rows)
                if text_column in df.columns:
                    texts = df[text_column].dropna().astype(str).tolist()
                    metadata = {'source': 'excel', 'columns': list(df.columns), 'shape': df.shape}
                else:
                    st.error(f"Colonne '{text_column}' introuvable. Colonnes disponibles: {list(df.columns)}")
                    return
                    
            elif uploaded_file.type == "application/json":
                data = json.load(uploaded_file)
                if isinstance(data, list):
                    texts = [item.get(text_column, str(item)) for item in data]
                else:
                    texts = [str(data)]
                metadata = {'source': 'json', 'type': type(data).__name__}
                
            else:  # CSV
                df = pd.read_csv(uploaded_file, nrows=max_rows)
                if text_column in df.columns:
                    texts = df[text_column].dropna().astype(str).tolist()
                    metadata = {'source': 'csv', 'columns': list(df.columns), 'shape': df.shape}
                else:
                    st.error(f"Colonne '{text_column}' introuvable. Colonnes disponibles: {list(df.columns)}")
                    return
            
            # Échantillonnage si demandé
            if sample_analysis and len(texts) > 100:
                texts = np.random.choice(texts, 100, replace=False).tolist()
                st.info(f"Analyse d'un échantillon de 100 textes sur {len(texts)} total")
            
            if texts:
                st.success(f"📊 {len(texts)} textes chargés depuis {metadata['source'].upper()}")
                
                # Analyse avec barre de progression
                with st.spinner(f"Analyse de {len(texts)} textes..."):
                    if st.session_state.get('use_parallel', True) and len(texts) > 10:
                        results = analyzer.analyze_batch_async(texts)
                    else:
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, text in enumerate(texts):
                            try:
                                result = analyzer.analyze_sentiment_comprehensive(text)
                                results.append(result)
                            except Exception as e:
                                logger.warning(f"Erreur lors de l'analyse du texte {i+1}: {e}")
                                continue
                            
                            progress_bar.progress((i + 1) / len(texts))
                        
                        progress_bar.empty()
                
                if results:
                    display_batch_analysis_results(results, metadata)
                else:
                    st.error("Aucun texte n'a pu être analysé")
            else:
                st.warning("Aucun texte trouvé dans le fichier")
                
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier: {str(e)}")
            logger.error(f"Erreur lecture fichier: {e}")

def handle_comparative_analysis():
    """Interface pour l'analyse comparative de méthodes"""
    st.subheader("📊 Analyse Comparative des Méthodes")
    
    st.info("Comparez les performances des différentes méthodes d'analyse de sentiment")
    
    # Textes de test
    test_texts = st.text_area(
        "Entrez plusieurs textes (un par ligne):",
        value="""Ce produit est absolument fantastique !
Service client très décevant...
Qualité correcte, rien d'exceptionnel.
Interface vraiment intuitive et moderne.
Prix un peu élevé mais ça vaut le coup.""",
        height=150
    )
    
    if st.button("🔍 Analyser et Comparer"):
        texts = [text.strip() for text in test_texts.split('\n') if text.strip()]
        
        if texts:
            with st.spinner("Analyse comparative en cours..."):
                comparative_results = []
                
                for text in texts:
                    result = analyzer.analyze_sentiment_comprehensive(text)
                    comparative_results.append({
                        'Texte': text[:50] + "..." if len(text) > 50 else text,
                        'VADER': result.vader_scores['compound'],
                        'TextBlob': result.textblob_scores['polarity'],
                        'Lexique FR': result.french_lexicon_scores['compound'],
                        'Consensus': result.consensus['score'],
                        'Classification': result.consensus['classification']
                    })
                
                # Affichage des résultats comparatifs
                df_comp = pd.DataFrame(comparative_results)
                st.dataframe(df_comp, use_container_width=True)
                
                # Graphique comparatif
                fig = go.Figure()
                
                for method in ['VADER', 'TextBlob', 'Lexique FR', 'Consensus']:
                    fig.add_trace(go.Scatter(
                        x=list(range(len(texts))),
                        y=df_comp[method],
                        mode='lines+markers',
                        name=method,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title="📈 Comparaison des Scores par Méthode",
                    xaxis_title="Texte",
                    yaxis_title="Score de Sentiment",
                    hovermode='x unified',
                    height=500
                )
                
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutre")
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistiques de corrélation
                st.subheader("🔗 Corrélations entre Méthodes")
                correlation_matrix = df_comp[['VADER', 'TextBlob', 'Lexique FR', 'Consensus']].corr()
                
                fig_corr = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu',
                    title="Matrice de Corrélation des Méthodes"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

def display_single_analysis_result(result: SentimentResult, demo_mode: bool = False):
    """Affiche le résultat d'une analyse unique avec mode démo"""
    consensus = result.consensus
    
    # Métriques principales avec design amélioré
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_color = "🟢" if consensus['classification'] == 'Positif' else "🔴" if consensus['classification'] == 'Négatif' else "🟡"
        st.metric("Sentiment Final", f"{sentiment_color} {consensus['classification']}")
    
    with col2:
        score_delta = "+" if consensus['score'] > 0 else ""
        st.metric("Score Consensus", f"{consensus['score']:.3f}", delta=f"{score_delta}{consensus['score']:.3f}")
    
    with col3:
        confidence_color = "🟢" if result.confidence in ["Très élevée", "Élevée"] else "🟡" if result.confidence == "Moyenne" else "🔴"
        st.metric("Confiance", f"{confidence_color} {result.confidence}")
    
    with col4:
        st.metric("Mots Analysés", f"{result.french_lexicon_scores.get('word_count', 0)}")
    
    if demo_mode:
        # Détails par méthode en mode démo
        st.subheader("📊 Analyse Détaillée par Méthode")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🇺🇸 VADER")
            st.markdown(f"**Positif:** {result.vader_scores['positive']:.2%}")
            st.markdown(f"**Négatif:** {result.vader_scores['negative']:.2%}")
            st.markdown(f"**Neutre:** {result.vader_scores['neutral']:.2%}")
            st.markdown(f"**Score:** {result.vader_scores['compound']:.3f}")
        
        with col2:
            st.markdown("### 🧠 TextBlob")
            st.markdown(f"**Polarité:** {result.textblob_scores['polarity']:.3f}")
            st.markdown(f"**Subjectivité:** {result.textblob_scores['subjectivity']:.3f}")
            polarity_text = "Positif" if result.textblob_scores['polarity'] > 0.1 else "Négatif" if result.textblob_scores['polarity'] < -0.1 else "Neutre"
            st.markdown(f"**Classification:** {polarity_text}")
        
        with col3:
            st.markdown("### 🇫🇷 Lexique Français")
            st.markdown(f"**Positif:** {result.french_lexicon_scores['positive']:.2%}")
            st.markdown(f"**Négatif:** {result.french_lexicon_scores['negative']:.2%}")
            st.markdown(f"**Score:** {result.french_lexicon_scores['compound']:.3f}")
            st.markdown(f"**Classification:** {result.french_lexicon_scores['classification']}")
        
        # Graphique radar des scores
        categories = ['VADER', 'TextBlob', 'Lexique FR']
        scores = [
            result.vader_scores['compound'],
            result.textblob_scores['polarity'],
            result.french_lexicon_scores['compound']
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[abs(s) for s in scores],
            theta=categories,
            fill='toself',
            name='Intensité',
            line_color='rgba(102, 126, 234, 0.8)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=False,
            title="📡 Radar des Scores par Méthode"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Texte préprocessé
    with st.expander("🔍 Détails du Préprocessing"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Texte Original:**")
            st.text_area("", value=result.text, height=100, disabled=True)
        
        with col2:
            st.markdown("**Texte Nettoyé:**")
            st.text_area("", value=result.cleaned_text, height=100, disabled=True)
        
        if result.metadata:
            st.markdown("**Métadonnées:**")
            st.json(result.metadata)

def display_batch_analysis_results(results: List[SentimentResult], metadata: Dict = None):
    """Affiche les résultats d'analyse en lot avec fonctionnalités avancées"""
    if not results:
        st.warning("Aucun résultat à afficher")
        return
    
    # Statistiques globales améliorées
    st.subheader("📊 Statistiques Globales")
    
    classifications = [r.consensus['classification'] for r in results]
    scores = [r.consensus['score'] for r in results]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Analysés", len(results))
    
    with col2:
        positive_count = classifications.count('Positif')
        positive_pct = positive_count / len(results) * 100
        st.metric("Positifs", f"{positive_count}", delta=f"{positive_pct:.1f}%")
    
    with col3:
        negative_count = classifications.count('Négatif')
        negative_pct = negative_count / len(results) * 100
        st.metric("Négatifs", f"{negative_count}", delta=f"{negative_pct:.1f}%")
    
    with col4:
        neutral_count = classifications.count('Neutre')
        neutral_pct = neutral_count / len(results) * 100
        st.metric("Neutres", f"{neutral_count}", delta=f"{neutral_pct:.1f}%")
    
    with col5:
        avg_score = np.mean(scores)
        st.metric("Score Moyen", f"{avg_score:.3f}")
    
    # Alertes intelligentes
    display_smart_alerts(results)
    
    # Visualisations améliorées
    create_advanced_visualizations(results)
    
    # Analyse des mots-clés
    display_keyword_analysis(results)
    
    # Tableau des résultats avec filtres
    st.subheader("📋 Résultats Détaillés")
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_filter = st.selectbox("Filtrer par sentiment", ["Tous", "Positif", "Négatif", "Neutre"])
    
    with col2:
        confidence_filter = st.selectbox("Filtrer par confiance", ["Toutes", "Très élevée", "Élevée", "Moyenne", "Faible"])
    
    with col3:
        sort_by = st.selectbox("Trier par", ["Timestamp", "Score", "Longueur"])
    
    # Appliquer les filtres
    filtered_results = results.copy()
    
    if sentiment_filter != "Tous":
        filtered_results = [r for r in filtered_results if r.consensus['classification'] == sentiment_filter]
    
    if confidence_filter != "Toutes":
        filtered_results = [r for r in filtered_results if r.confidence == confidence_filter]
    
    # Créer DataFrame pour affichage
    df_display = pd.DataFrame([
        {
            'Texte': r.text[:100] + "..." if len(r.text) > 100 else r.text,
            'Sentiment': r.consensus['classification'],
            'Score': r.consensus['score'],
            'Confiance': r.confidence,
            'Longueur': len(r.text),
            'Mots': r.metadata.get('word_count', 0) if r.metadata else 0,
            'Timestamp': r.timestamp.strftime("%H:%M:%S")
        }
        for r in filtered_results
    ])
    
    if not df_display.empty:
        st.dataframe(df_display, use_container_width=True)
        
        # Export des résultats
        st.subheader("💾 Export des Résultats")
        
        export_data = []
        for r in filtered_results:
            export_data.append({
                'text': r.text,
                'sentiment': r.consensus['classification'],
                'score': r.consensus['score'],
                'confidence': r.confidence,
                'vader_compound': r.vader_scores['compound'],
                'textblob_polarity': r.textblob_scores['polarity'],
                'french_compound': r.french_lexicon_scores['compound'],
                'timestamp': r.timestamp.isoformat()
            })
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Télécharger JSON"):
                json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="💾 JSON",
                    data=json_data,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📥 Télécharger CSV"):
                df_export = pd.DataFrame(export_data)
                csv_data = df_export.to_csv(index=False)
                st.download_button(
                    label="💾 CSV",
                    data=csv_data,
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📥 Télécharger Excel"):
                df_export = pd.DataFrame(export_data)
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_export.to_excel(writer, sheet_name='Sentiment_Analysis', index=False)
                
                st.download_button(
                    label="💾 Excel",
                    data=buffer.getvalue(),
                    file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    else:
        st.info("Aucun résultat ne correspond aux filtres sélectionnés")

def display_smart_alerts(results: List[SentimentResult]):
    """Affiche des alertes intelligentes basées sur l'analyse"""
    st.subheader("🚨 Alertes Intelligentes")
    
    classifications = [r.consensus['classification'] for r in results]
    scores = [r.consensus['score'] for r in results]
    
    negative_ratio = classifications.count('Négatif') / len(results)
    positive_ratio = classifications.count('Positif') / len(results)
    low_confidence_ratio = len([r for r in results if r.confidence in ['Faible', 'Moyenne']]) / len(results)
    
    alerts = []
    
    # Alertes de sentiment
    if negative_ratio > 0.6:
        alerts.append({
            'type': 'error',
            'message': f"🚨 ALERTE CRITIQUE: {negative_ratio:.1%} de sentiments négatifs détectés!"
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
    
    # Alertes de confiance
    if low_confidence_ratio > 0.5:
        alerts.append({
            'type': 'warning',
            'message': f"🤔 {low_confidence_ratio:.1%} des analyses ont une confiance faible/moyenne"
        })
    
    # Alertes de variance
    score_std = np.std(scores)
    if score_std > 0.8:
        alerts.append({
            'type': 'info',
            'message': f"📊 Grande variance dans les scores (σ={score_std:.2f}) - sentiments très dispersés"
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
    
    if not alerts:
        st.info("✅ Aucune alerte - Les résultats semblent équilibrés")

def create_advanced_visualizations(results: List[SentimentResult]):
    """Crée des visualisations avancées"""
    st.subheader("📈 Visualisations Avancées")
    
    # Préparer les données
    df = pd.DataFrame([
        {
            'classification': r.consensus['classification'],
            'score': r.consensus['score'],
            'confidence': r.confidence,
            'timestamp': r.timestamp,
            'text_length': len(r.text),
            'word_count': r.metadata.get('word_count', 0) if r.metadata else 0,
            'vader_score': r.vader_scores['compound'],
            'textblob_score': r.textblob_scores['polarity'],
            'french_score': r.french_lexicon_scores['compound']
        }
        for r in results
    ])
    
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔗 Corrélations", "⏰ Temporel"])
    
    with tab1:
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
            fig_hist.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="Neutre")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Box plot par confiance
        fig_box = px.box(
            df,
            x='confidence',
            y='score',
            color='classification',
            title="📦 Distribution des Scores par Niveau de Confiance",
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot longueur vs sentiment
            fig_scatter = px.scatter(
                df,
                x='word_count',
                y='score',
                color='classification',
                size='text_length',
                title="📝 Longueur vs Sentiment",
                labels={'word_count': 'Nombre de mots', 'score': 'Score de sentiment'},
                color_discrete_map={
                    'Positif': '#28a745',
                    'Négatif': '#dc3545',
                    'Neutre': '#6c757d'
                },
                hover_data=['confidence']
            )
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Comparaison des méthodes
            methods_comparison = df[['vader_score', 'textblob_score', 'french_score', 'score']].corr()
            
            fig_corr = px.imshow(
                methods_comparison,
                text_auto=True,
                color_continuous_scale='RdBu',
                title="🔗 Corrélation entre Méthodes",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        if len(df) > 1:
            # Évolution temporelle
            df_time = df.sort_values('timestamp')
            
            fig_time = px.line(
                df_time,
                x='timestamp',
                y='score',
                color='classification',
                title="⏰ Évolution Temporelle des Sentiments",
                color_discrete_map={
                    'Positif': '#28a745',
                    'Négatif': '#dc3545',
                    'Neutre': '#6c757d'
                },
                markers=True
            )
            fig_time.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_time, use_container_width=True)
            
            # Moyennes mobiles
            if len(df) > 10:
                df_time['rolling_mean'] = df_time['score'].rolling(window=5, center=True).mean()
                
                fig_rolling = px.line(
                    df_time,
                    x='timestamp',
                    y=['score', 'rolling_mean'],
                    title="📈 Tendance avec Moyenne Mobile (5 points)",
                    labels={'value': 'Score', 'variable': 'Série'}
                )
                st.plotly_chart(fig_rolling, use_container_width=True)
        else:
            st.info("Plus de données nécessaires pour l'analyse temporelle")

def display_keyword_analysis(results: List[SentimentResult]):
    """Affiche l'analyse des mots-clés avancée"""
    st.subheader("🔍 Analyse des Mots-Clés")
    
    # Extraire les textes
    texts = [r.text for r in results]
    
    # TF-IDF
    keywords = analyzer.extract_keywords_tfidf(texts, max_features=30)
    
    if keywords:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top mots-clés
            st.markdown("**🏆 Top Mots-Clés (TF-IDF)**")
            keywords_df = pd.DataFrame(
                list(keywords.items()), 
                columns=['Terme', 'Score TF-IDF']
            ).sort_values('Score TF-IDF', ascending=False).head(15)
            
            fig_bar = px.bar(
                keywords_df,
                x='Score TF-IDF',
                y='Terme',
                orientation='h',
                title="Importance des Termes",
                color='Score TF-IDF',
                color_continuous_scale='Viridis'
            )
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Nuage de mots
            st.markdown("**☁️ Nuage de Mots**")
            try:
                wordcloud = WordCloud(
                    width=500, 
                    height=400, 
                    background_color='white',
                    colormap='viridis',
                    max_words=50,
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate_from_frequencies(keywords)
                
                # Convertir en image pour Streamlit
                img_buffer = io.BytesIO()
                wordcloud.to_image().save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                st.markdown(
                    f'<img src="data:image/png;base64,{img_str}" style="width:100%">',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.error(f"Impossible de générer le nuage de mots: {e}")
        
        # Corrélation termes-sentiment
        st.markdown("**🔗 Corrélation Termes ↔ Sentiments**")
        
        term_sentiment_data = []
        for result in results:
            words = analyzer.safe_tokenize(result.cleaned_text)
            words = [word for word in words if word not in analyzer.stop_words and len(word) > 2]
            
            for word in words:
                term_sentiment_data.append({
                    'term': word,
                    'sentiment_score': result.consensus['score'],
                    'classification': result.consensus['classification']
                })
        
        if term_sentiment_data:
            df_terms = pd.DataFrame(term_sentiment_data)
            
            # Agrégation par terme
            correlation_df = df_terms.groupby('term').agg({
                'sentiment_score': ['mean', 'count', 'std'],
                'classification': lambda x: Counter(x).most_common(1)[0][0]
            }).round(3)
            
            correlation_df.columns = ['avg_sentiment', 'frequency', 'std_sentiment', 'dominant_class']
            correlation_df = correlation_df.reset_index()
            correlation_df = correlation_df[correlation_df['frequency'] >= 2]
            correlation_df = correlation_df.sort_values('frequency', ascending=False)
            
            if not correlation_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Termes les plus positifs
                    positive_terms = correlation_df[
                        correlation_df['dominant_class'] == 'Positif'
                    ].head(10)
                    
                    if not positive_terms.empty:
                        fig_pos = px.scatter(
                            positive_terms,
                            x='frequency',
                            y='avg_sentiment',
                            size='std_sentiment',
                            hover_data=['term'],
                            title="🟢 Termes Positifs (Fréquence vs Sentiment)",
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
                        fig_neg = px.scatter(
                            negative_terms,
                            x='frequency',
                            y='avg_sentiment',
                            size='std_sentiment',
                            hover_data=['term'],
                            title="🔴 Termes Négatifs (Fréquence vs Sentiment)",
                            color='avg_sentiment',
                            color_continuous_scale='Reds'
                        )
                        st.plotly_chart(fig_neg, use_container_width=True)
                
                # Tableau complet des corrélations
                with st.expander("📊 Tableau Complet des Corrélations"):
                    st.dataframe(
                        correlation_df.style.format({
                            'avg_sentiment': '{:.3f}',
                            'frequency': '{:.0f}',
                            'std_sentiment': '{:.3f}'
                        }),
                        use_container_width=True
                    )
    else:
        st.info("Pas assez de données pour l'analyse des mots-clés")

def handle_twitter_analysis():
    """Interface pour l'analyse Twitter simulée améliorée"""
    st.subheader("🐦 Analyse de Réseaux Sociaux (Simulation)")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_term = st.text_input(
            "Terme de recherche:",
            value="TechCorp",
            placeholder="Entrez une marque, hashtag ou mot-clé"
        )
    
    with col2:
        tweet_count = st.selectbox("Nombre de messages", [25, 50, 100, 200], index=1)
    
    with col3:
        platform = st.selectbox("Plateforme", ["Twitter", "Facebook", "Instagram", "LinkedIn"])
    
    # Configuration avancée
    with st.expander("⚙️ Configuration Avancée"):
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_bias = st.slider("Biais sentiment", -0.5, 0.5, 0.0, help="Ajuste la proportion de sentiments positifs/négatifs")
            time_range = st.selectbox("Période", ["1 heure", "6 heures", "24 heures", "7 jours"])
        
        with col2:
            include_hashtags = st.checkbox("Inclure hashtags", value=True)
            include_mentions = st.checkbox("Inclure mentions", value=True)
    
    if st.button("🔍 Analyser les Messages", type="primary"):
        if search_term:
            with st.spinner(f"Simulation de {tweet_count} messages pour '{search_term}'..."):
                # Générer des messages simulés plus réalistes
                simulated_messages = generate_realistic_social_media_data(
                    search_term, 
                    tweet_count, 
                    platform, 
                    sentiment_bias,
                    include_hashtags,
                    include_mentions
                )
                
                # Analyser les messages
                results = []
                progress_bar = st.progress(0)
                
                for i, message in enumerate(simulated_messages):
                    try:
                        result = analyzer.analyze_sentiment_comprehensive(message['text'])
                        result.metadata = {
                            'platform': platform,
                            'search_term': search_term,
                            'hashtags': message.get('hashtags', []),
                            'mentions': message.get('mentions', []),
                            'simulated': True
                        }
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Erreur analyse message {i}: {e}")
                        continue
                    
                    progress_bar.progress((i + 1) / len(simulated_messages))
                    time.sleep(0.05)  # Simulation réaliste
                
                progress_bar.empty()
                
                if results:
                    display_social_media_results(results, search_term, platform)
                else:
                    st.error("Aucun message n'a pu être analysé")

def generate_realistic_social_media_data(term: str, count: int, platform: str, bias: float = 0.0, 
                                       include_hashtags: bool = True, include_mentions: bool = True) -> List[Dict]:
    """Génère des données de réseaux sociaux réalistes"""
    
    # Templates par plateforme et sentiment
    templates = {
        'positive': {
            'Twitter': [
                f"Vraiment impressionné par {term} ! Service au top 👍 #satisfaction",
                f"Je recommande vivement {term}, qualité exceptionnelle! 🌟",
                f"{term} a dépassé mes attentes, bravo! 🎉 #happy #quality",
                f"Livraison ultra rapide de {term}, parfait! ⚡ #efficient #delivery",
                f"Interface {term} très intuitive, j'adore! 💖 #ux #design",
                f"Excellent support client chez {term} 👏 #service #pro",
                f"{term} continue d'innover, chapeau! 🚀 #innovation"
            ],
            'Facebook': [
                f"Je viens de découvrir {term} et c'est fantastique ! Vraiment recommandé à tous mes amis.",
                f"Expérience exceptionnelle avec {term}. Leur équipe est vraiment professionnelle.",
                f"Merci {term} pour ce service de qualité. Ça fait plaisir de voir des entreprises qui s'investissent vraiment.",
                f"{term} m'a bluffé ! Livraison rapide, produit conforme, que demander de plus ?",
                f"Je suis client de {term} depuis des années et ils ne m'ont jamais déçu. Top qualité !"
            ],
            'Instagram': [
                f"Obsédée par les produits {term} 😍 #quality #love #obsessed",
                f"Nouveau coup de cœur : {term} ! 💕 #newlove #amazing",
                f"Style et qualité au rendez-vous avec {term} ✨ #style #premium",
                f"{term} = perfection 🔥 #perfect #goals #inspiration"
            ],
            'LinkedIn': [
                f"Impressionné par l'approche innovante de {term} dans leur secteur. Stratégie remarquable.",
                f"Retour très positif sur notre collaboration avec {term}. Professionnalisme exemplaire.",
                f"{term} démontre une fois de plus leur expertise. Partenaire de confiance.",
                f"Benchmark intéressant avec {term}. Approche disruptive et résultats probants."
            ]
        },
        'negative': {
            'Twitter': [
                f"Très déçu de {term}, service client inexistant 😠 #disappointed #fail",
                f"Problème récurrent avec {term}, c'est inadmissible! 😤 #problem #angry",
                f"Qualité {term} en chute libre, dommage... 📉 #quality #decline",
                f"Attente interminable chez {term}, très frustrant! ⏰ #waiting #slow",
                f"Bug constant sur l'app {term}, quand sera-t-elle réparée? 🐛 #bug #broken",
                f"{term} ne répond plus aux messages, c'est du mépris! 😡 #noresponse"
            ],
            'Facebook': [
                f"Vraiment déçu de mon achat chez {term}. La qualité n'est pas au rendez-vous.",
                f"Service client de {term} à revoir absolument. Aucune réactivité, très frustrant.",
                f"Commande {term} arrivée en retard et abîmée. Pas professionnel du tout.",
                f"Je déconseille {term}. Promesses non tenues et SAV défaillant.",
                f"Expérience négative avec {term}. Heureusement que la concurrence existe."
            ],
            'Instagram': [
                f"Déçue de {term}... pas à la hauteur 😞 #disappointed #notgood",
                f"Attendais mieux de {term} 😕 #letdown #meh",
                f"{term} m'a déçue cette fois 💔 #heartbroken #sad",
                f"Qualité {term} pas terrible... 😬 #quality #issues"
            ],
            'LinkedIn': [
                f"Retour mitigé sur notre expérience avec {term}. Axes d'amélioration identifiés.",
                f"Collaboration avec {term} en deçà des attentes. Processus à optimiser.",
                f"Déception concernant la prestation {term}. ROI non atteint.",
                f"Points de vigilance sur {term}. Méthodologie à revoir pour les prochains projets."
            ]
        },
        'neutral': {
            'Twitter': [
                f"Test en cours avec {term}, on verra bien... #testing #wait",
                f"{term} a lancé une nouvelle fonctionnalité aujourd'hui #news #update",
                f"Mise à jour {term} disponible, quelqu'un l'a essayée? #update #question",
                f"Comparaison entre {term} et ses concurrents en cours #comparison #analysis",
                f"Webinaire {term} prévu la semaine prochaine 📅 #webinar #event"
            ],
            'Facebook': [
                f"Quelqu'un a testé {term} récemment ? J'hésite à commander...",
                f"Avis partagés sur {term}. Difficile de se faire une opinion.",
                f"{term} vient de sortir un nouveau produit. Qu'en pensez-vous ?",
                f"Prix {term} un peu élevé mais peut-être justifié par la qualité ?",
                f"Formation sur {term} prévue le mois prochain. Qui sera présent ?"
            ],
            'Instagram': [
                f"Découverte de {term} aujourd'hui 🤔 #discover #new",
                f"{term} dans ma wishlist 📝 #wishlist #maybe",
                f"Test de {term} en cours... 🧪 #testing #progress",
                f"Hésitation entre {term} et la concurrence 🤷‍♀️ #choice #dilemma"
            ],
            'LinkedIn': [
                f"Retour d'expérience sur {term} lors de notre dernière réunion équipe.",
                f"Benchmark en cours incluant {term}. Premiers résultats attendus fin de mois.",
                f"Formation équipe prévue sur les outils {term}. Montée en compétence nécessaire.",
                f"Analyse comparative {term} vs concurrents en cours. Méthodologie détaillée."
            ]
        }
    }
    
    messages = []
    
    # Ajuster les probabilités selon le biais
    base_probs = [0.4, 0.3, 0.3]  # positif, négatif, neutre
    if bias > 0:  # Plus positif
        probs = [0.4 + bias, 0.3 - bias/2, 0.3 - bias/2]
    elif bias < 0:  # Plus négatif
        probs = [0.4 + bias/2, 0.3 - bias, 0.3 + bias/2]
    else:
        probs = base_probs
    
    for i in range(count):
        # Choisir le sentiment
        sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=probs)
        
        # Choisir un template
        template = np.random.choice(templates[sentiment][platform])
        
        # Ajouter des hashtags si demandé
        hashtags = []
        if include_hashtags and np.random.random() > 0.3:
            hashtag_pool = {
                'positive': ['#excellent', '#top', '#parfait', '#love', '#best', '#amazing'],
                'negative': ['#déçu', '#problème', '#fail', '#bad', '#disappointed', '#issue'],
                'neutral': ['#test', '#avis', '#info', '#question', '#update', '#news']
            }
            hashtags = np.random.choice(hashtag_pool[sentiment], size=np.random.randint(1, 3), replace=False).tolist()
            template += ' ' + ' '.join(hashtags)
        
        # Ajouter des mentions si demandé
        mentions = []
        if include_mentions and np.random.random() > 0.7:
            mention_pool = ['@support', '@team', '@service_client', '@info', '@help']
            mentions = [np.random.choice(mention_pool)]
            template += ' ' + ' '.join(mentions)
        
        messages.append({
            'text': template,
            'sentiment': sentiment,
            'hashtags': hashtags,
            'mentions': mentions,
            'platform': platform
        })
    
    return messages

def display_social_media_results(results: List[SentimentResult], search_term: str, platform: str):
    """Affiche les résultats d'analyse des réseaux sociaux avec dashboard avancé"""
    st.success(f"✅ Analyse terminée pour '{search_term}' sur {platform} - {len(results)} messages analysés")
    
    # KPIs principaux avec design amélioré
    classifications = [r.consensus['classification'] for r in results]
    scores = [r.consensus['score'] for r in results]
    
    # Métriques avec deltas et couleurs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📊 Total Messages", len(results))
    
    with col2:
        positive_count = classifications.count('Positif')
        positive_pct = positive_count / len(results) * 100
        delta_color = "normal" if positive_pct >= 50 else "inverse"
        st.metric(
            "🟢 Positifs", 
            f"{positive_count}",
            delta=f"{positive_pct:.1f}%",
            delta_color=delta_color
        )
    
    with col3:
        negative_count = classifications.count('Négatif')
        negative_pct = negative_count / len(results) * 100
        delta_color = "inverse" if negative_pct >= 30 else "normal"
        st.metric(
            "🔴 Négatifs", 
            f"{negative_count}",
            delta=f"{negative_pct:.1f}%",
            delta_color=delta_color
        )
    
    with col4:
        neutral_count = classifications.count('Neutre')
        neutral_pct = neutral_count / len(results) * 100
        st.metric(
            "🟡 Neutres", 
            f"{neutral_count}",
            delta=f"{neutral_pct:.1f}%"
        )
    
    with col5:
        avg_score = np.mean(scores)
        score_trend = "📈" if avg_score > 0.1 else "📉" if avg_score < -0.1 else "➡️"
        st.metric(
            "📈 Score Moyen", 
            f"{avg_score:.3f}",
            delta=f"{score_trend}"
        )
    
    # Alertes spécifiques aux réseaux sociaux
    display_social_media_alerts(results, search_term, platform)
    
    # Dashboard avec onglets
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "⏰ Tendances", "#️⃣ Hashtags", "📋 Messages"])
    
    with tab1:
        display_batch_analysis_results(results, {'platform': platform, 'search_term': search_term})
    
    with tab2:
        display_temporal_analysis(results)
    
    with tab3:
        display_hashtag_analysis(results)
    
    with tab4:
        display_message_details(results, platform)

def display_social_media_alerts(results: List[SentimentResult], search_term: str, platform: str):
    """Alertes spécifiques aux réseaux sociaux"""
    st.subheader(f"🚨 Alertes {platform}")
    
    classifications = [r.consensus['classification'] for r in results]
    negative_ratio = classifications.count('Négatif') / len(results)
    
    alerts = []
    
    # Alertes spécifiques par plateforme
    if platform == "Twitter":
        if negative_ratio > 0.5:
            alerts.append({
                'type': 'error',
                'message': f"🐦 CRISE POTENTIELLE sur Twitter: {negative_ratio:.1%} de tweets négatifs pour '{search_term}'"
            })
        elif negative_ratio > 0.3:
            alerts.append({
                'type': 'warning',
                'message': f"⚠️ Surveillance recommandée: {negative_ratio:.1%} de sentiments négatifs sur Twitter"
            })
    
    elif platform == "Facebook":
        if negative_ratio > 0.4:
            alerts.append({
                'type': 'warning',
                'message': f"📘 Attention Facebook: {negative_ratio:.1%} de posts négatifs - Impact sur l'engagement possible"
            })
    
    elif platform == "LinkedIn":
        if negative_ratio > 0.3:
            alerts.append({
                'type': 'warning',
                'message': f"💼 Image professionnelle: {negative_ratio:.1%} de posts négatifs sur LinkedIn"
            })
    
    # Recommandations
    if negative_ratio > 0.4:
        alerts.append({
            'type': 'info',
            'message': f"💡 Recommandation: Réponse proactive nécessaire sur {platform} pour '{search_term}'"
        })
    
    # Afficher les alertes
    for alert in alerts:
        if alert['type'] == 'error':
            st.error(alert['message'])
        elif alert['type'] == 'warning':
            st.warning(alert['message'])
        else:
            st.info(alert['message'])

def display_temporal_analysis(results: List[SentimentResult]):
    """Analyse temporelle avancée"""
    if len(results) < 2:
        st.info("Plus de données nécessaires pour l'analyse temporelle")
        return
    
    # Simuler des timestamps répartis sur la période
    now = datetime.now()
    time_span = timedelta(hours=6)  # 6 heures de données
    
    for i, result in enumerate(results):
        # Répartir les timestamps de façon réaliste
        time_offset = (i / len(results)) * time_span
        result.timestamp = now - time_span + time_offset
    
    # Créer DataFrame temporel
    df_time = pd.DataFrame([
        {
            'timestamp': r.timestamp,
            'score': r.consensus['score'],
            'classification': r.consensus['classification'],
            'hour': r.timestamp.hour,
            'minute_group': r.timestamp.minute // 15  # Groupes de 15 minutes
        }
        for r in results
    ])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Évolution temporelle
        fig_timeline = px.scatter(
            df_time,
            x='timestamp',
            y='score',
            color='classification',
            title="⏰ Évolution Temporelle des Sentiments",
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            },
            trendline="lowess"
        )
        fig_timeline.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    with col2:
        # Distribution par heure
        hourly_sentiment = df_time.groupby(['hour', 'classification']).size().reset_index(name='count')
        
        fig_hourly = px.bar(
            hourly_sentiment,
            x='hour',
            y='count',
            color='classification',
            title="📊 Volume par Heure",
            color_discrete_map={
                'Positif': '#28a745',
                'Négatif': '#dc3545',
                'Neutre': '#6c757d'
            }
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

def display_hashtag_analysis(results: List[SentimentResult]):
    """Analyse des hashtags et mentions"""
    hashtags_data = []
    mentions_data = []
    
    for result in results:
        if result.metadata and 'hashtags' in result.metadata:
            for hashtag in result.metadata['hashtags']:
                hashtags_data.append({
                    'hashtag': hashtag,
                    'sentiment': result.consensus['classification'],
                    'score': result.consensus['score']
                })
        
        if result.metadata and 'mentions' in result.metadata:
            for mention in result.metadata['mentions']:
                mentions_data.append({
                    'mention': mention,
                    'sentiment': result.consensus['classification'],
                    'score': result.consensus['score']
                })
    
    col1, col2 = st.columns(2)
    
    with col1:
        if hashtags_data:
            st.markdown("**#️⃣ Analyse des Hashtags**")
            hashtags_df = pd.DataFrame(hashtags_data)
            hashtag_summary = hashtags_df.groupby('hashtag').agg({
                'score': 'mean',
                'sentiment': lambda x: Counter(x).most_common(1)[0][0]
            }).round(3)
            hashtag_summary['count'] = hashtags_df['hashtag'].value_counts()
            hashtag_summary = hashtag_summary.sort_values('count', ascending=False).head(10)
            
            fig_hashtags = px.bar(
                hashtag_summary.reset_index(),
                x='count',
                y='hashtag',
                color='score',
                orientation='h',
                title="Top Hashtags par Fréquence",
                color_continuous_scale='RdYlGn',
                hover_data=['sentiment']
            )
            st.plotly_chart(fig_hashtags, use_container_width=True)
        else:
            st.info("Aucun hashtag trouvé dans les données")
    
    with col2:
        if mentions_data:
            st.markdown("**👤 Analyse des Mentions**")
            mentions_df = pd.DataFrame(mentions_data)
            mention_summary = mentions_df.groupby('mention').agg({
                'score': 'mean',
                'sentiment': lambda x: Counter(x).most_common(1)[0][0]
            }).round(3)
            mention_summary['count'] = mentions_df['mention'].value_counts()
            mention_summary = mention_summary.sort_values('count', ascending=False).head(10)
            
            fig_mentions = px.bar(
                mention_summary.reset_index(),
                x='count',
                y='mention',
                color='score',
                orientation='h',
                title="Top Mentions par Fréquence",
                color_continuous_scale='RdYlGn',
                hover_data=['sentiment']
            )
            st.plotly_chart(fig_mentions, use_container_width=True)
        else:
            st.info("Aucune mention trouvée dans les données")

def display_message_details(results: List[SentimentResult], platform: str):
    """Affiche les détails des messages avec filtres avancés"""
    
    # Filtres avancés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_filter = st.selectbox("Sentiment", ["Tous", "Positif", "Négatif", "Neutre"], key="msg_sentiment")
    
    with col2:
        confidence_filter = st.selectbox("Confiance", ["Toutes", "Très élevée", "Élevée", "Moyenne", "Faible"], key="msg_confidence")
    
    with col3:
        score_range = st.slider("Plage de scores", -1.0, 1.0, (-1.0, 1.0), step=0.1, key="msg_score")
    
    with col4:
        sort_option = st.selectbox("Trier par", ["Timestamp", "Score", "Longueur", "Confiance"], key="msg_sort")
    
    # Appliquer les filtres
    filtered_results = results.copy()
    
    if sentiment_filter != "Tous":
        filtered_results = [r for r in filtered_results if r.consensus['classification'] == sentiment_filter]
    
    if confidence_filter != "Toutes":
        filtered_results = [r for r in filtered_results if r.confidence == confidence_filter]
    
    filtered_results = [r for r in filtered_results if score_range[0] <= r.consensus['score'] <= score_range[1]]
    
    # Tri
    if sort_option == "Score":
        filtered_results.sort(key=lambda x: x.consensus['score'], reverse=True)
    elif sort_option == "Longueur":
        filtered_results.sort(key=lambda x: len(x.text), reverse=True)
    elif sort_option == "Confiance":
        confidence_order = {"Très élevée": 4, "Élevée": 3, "Moyenne": 2, "Faible": 1}
        filtered_results.sort(key=lambda x: confidence_order.get(x.confidence, 0), reverse=True)
    else:  # Timestamp
        filtered_results.sort(key=lambda x: x.timestamp, reverse=True)
    
    st.write(f"**📋 {len(filtered_results)} messages filtrés sur {len(results)} total**")
    
    # Affichage des messages
    for i, result in enumerate(filtered_results[:50]):  # Limite à 50 pour la performance
        with st.expander(f"{i+1}. {result.consensus['classification']} - Score: {result.consensus['score']:.3f} - {result.timestamp.strftime('%H:%M:%S')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Message:** {result.text}")
                
                # Métadonnées si disponibles
                if result.metadata:
                    if result.metadata.get('hashtags'):
                        st.markdown(f"**Hashtags:** {', '.join(result.metadata['hashtags'])}")
                    if result.metadata.get('mentions'):
                        st.markdown(f"**Mentions:** {', '.join(result.metadata['mentions'])}")
            
            with col2:
                st.markdown(f"**Plateforme:** {platform}")
                st.markdown(f"**Confiance:** {result.confidence}")
                st.markdown(f"**Longueur:** {len(result.text)} caractères")
                
                # Mini-graphique des scores
                scores_data = {
                    'VADER': result.vader_scores['compound'],
                    'TextBlob': result.textblob_scores['polarity'],
                    'Français': result.french_lexicon_scores['compound']
                }
                
                fig_mini = go.Figure(data=[
                    go.Bar(x=list(scores_data.keys()), y=list(scores_data.values()))
                ])
                fig_mini.update_layout(
                    height=200,
                    showlegend=False,
                    margin=dict(l=0, r=0, t=20, b=0),
                    title="Scores par méthode"
                )
                fig_mini.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_mini, use_container_width=True)

def handle_realtime_simulation():
    """Interface pour la simulation temps réel améliorée"""
    st.subheader("🔄 Simulation Temps Réel")
    
    # Configuration de la simulation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        simulation_speed = st.slider("Vitesse (secondes)", 1, 10, 3)
        max_messages = st.number_input("Messages max", 10, 200, 50)
    
    with col2:
        message_source = st.selectbox("Source de données", ["Exemples prédéfinis", "Templates aléatoires", "Mixte"])
        auto_sentiment_bias = st.checkbox("Biais automatique", help="Varie automatiquement la polarité")
    
    with col3:
        show_live_charts = st.checkbox("Graphiques temps réel", value=True)
        save_simulation = st.checkbox("Sauvegarder simulation")
    
    # Conteneurs pour l'affichage temps réel
    if show_live_charts:
        metrics_container = st.empty()
        chart_container = st.empty()
    
    messages_container = st.empty()
    
    # Contrôles de simulation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Démarrer Simulation", type="primary"):
            st.session_state.simulation_active = True
            st.session_state.simulation_results = []
    
    with col2:
        if st.button("⏸️ Pause"):
            st.session_state.simulation_active = False
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.simulation_active = False
            st.session_state.simulation_results = []
    
    # Simulation active
    if st.session_state.get('simulation_active', False):
        run_realtime_simulation(
            simulation_speed, 
            max_messages, 
            message_source, 
            auto_sentiment_bias,
            metrics_container if show_live_charts else None,
            chart_container if show_live_charts else None,
            messages_container,
            save_simulation
        )

def run_realtime_simulation(speed: int, max_msg: int, source: str, auto_bias: bool,
                          metrics_container, chart_container, messages_container, save_sim: bool):
    """Execute la simulation temps réel"""
    
    # Données d'exemple étendues
    sample_messages = [
        "Ce service client est vraiment excellent ! Merci beaucoup 😊",
        "Interface très intuitive, j'adore cette nouvelle version",
        "Livraison rapide et produit conforme, parfait !",
        "Équipe support très réactive et professionnelle",
        "Qualité exceptionnelle, je recommande vivement",
        
        "Service client décevant, très lent à répondre...",
        "Problème technique récurrent, très frustrant",
        "Commande arrivée en retard et endommagée",
        "Prix trop élevé pour la qualité proposée",
        "Interface confuse, pas du tout intuitive",
        
        "Produit correct, sans plus ni moins",
        "Mise à jour disponible, quelqu'un l'a testée ?",
        "Prix dans la moyenne du marché",
        "Fonctionnalités standard, rien d'exceptionnel",
        "Service client réactif mais solutions limitées"
    ]
    
    results = st.session_state.get('simulation_results', [])
    message_count = 0
    
    # Boucle de simulation
    while (st.session_state.get('simulation_active', False) and 
           message_count < max_msg):
        
        try:
            # Générer un message
            if source == "Exemples prédéfinis":
                message = np.random.choice(sample_messages)
            elif source == "Templates aléatoires":
                message = generate_random_message_template()
            else:  # Mixte
                if np.random.random() > 0.5:
                    message = np.random.choice(sample_messages)
                else:
                    message = generate_random_message_template()
            
            # Appliquer un biais automatique si activé
            if auto_bias:
                bias_cycle = np.sin(message_count * 0.3) * 0.3  # Oscillation
                if bias_cycle > 0.1:
                    # Remplacer par un message plus positif
                    positive_words = ['excellent', 'fantastique', 'parfait', 'génial']
                    message = f"Service vraiment {np.random.choice(positive_words)} ! Je recommande."
                elif bias_cycle < -0.1:
                    # Remplacer par un message plus négatif
                    negative_words = ['décevant', 'problématique', 'inadmissible', 'frustrant']
                    message = f"Expérience {np.random.choice(negative_words)}, très déçu..."
            
            # Analyser le message
            result = analyzer.analyze_sentiment_comprehensive(message)
            result.metadata = {'simulation_step': message_count, 'auto_bias': auto_bias}
            results.append(result)
            
            # Limiter le nombre de résultats en mémoire
            if len(results) > max_msg:
                results = results[-max_msg:]
            
            st.session_state.simulation_results = results
            
            # Mise à jour des affichages
            if metrics_container and chart_container:
                update_realtime_display(results, metrics_container, chart_container)
            
            update_messages_display(results[-10:], messages_container)  # 10 derniers messages
            
            message_count += 1
            time.sleep(speed)
            
        except Exception as e:
            st.error(f"Erreur dans la simulation: {e}")
            st.session_state.simulation_active = False
            break
    
    # Fin de simulation
    if message_count >= max_msg:
        st.session_state.simulation_active = False
        st.success(f"✅ Simulation terminée - {len(results)} messages analysés")
        
        if save_sim and results:
            save_simulation_results(results)

def generate_random_message_template() -> str:
    """Génère un message aléatoire basé sur des templates"""
    templates = {
        'positive': [
            "Service {adjective}, vraiment {intensifier} !",
            "Produit {adjective}, je {action_positive}",
            "{intensifier} {adjective} cette {noun} !",
            "Équipe {adjective}, {action_positive} sans hésiter"
        ],
        'negative': [
            "Service {adjective}, très {intensifier}...",
            "Problème {adjective}, c'est {intensifier}",
            "{intensifier} {adjective} cette {noun}",
            "Expérience {adjective}, je {action_negative}"
        ],
        'neutral': [
            "Service {adjective}, {opinion_neutre}",
            "Produit {adjective}, {evaluation}",
            "{opinion_neutre} de cette {noun}",
            "Test en cours, {evaluation}"
        ]
    }
    
    words = {
        'positive': {
            'adjective': ['excellent', 'fantastique', 'parfait', 'génial', 'remarquable'],
            'intensifier': ['recommande vivement', 'adore', 'apprécie énormément'],
            'action_positive': ['recommande', 'valide', 'adopte', 'approuve'],
            'noun': ['application', 'plateforme', 'solution', 'interface']
        },
        'negative': {
            'adjective': ['décevant', 'problématique', 'défaillant', 'inadmissible'],
            'intensifier': ['frustrant', 'décevant', 'problématique', 'inacceptable'],
            'action_negative': ['déconseille', 'évite', 'regrette', 'critique'],
            'noun': ['application', 'plateforme', 'solution', 'interface']
        },
        'neutral': {
            'adjective': ['correct', 'standard', 'classique', 'basique'],
            'opinion_neutre': ['avis mitigé', 'retour neutre', 'impression correcte'],
            'evaluation': ['on verra', 'à voir', 'en cours d\'évaluation'],
            'noun': ['application', 'plateforme', 'solution', 'interface']
        }
    }
    
    sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])
    template = np.random.choice(templates[sentiment])
    word_dict = words[sentiment]
    
    # Remplacer les placeholders
    for placeholder, word_list in word_dict.items():
        if f'{{{placeholder}}}' in template:
            template = template.replace(f'{{{placeholder}}}', np.random.choice(word_list))
    
    return template

def update_realtime_display(results: List[SentimentResult], metrics_container, chart_container):
    """Met à jour l'affichage temps réel"""
    
    if not results:
        return
    
    with metrics_container.container():
        # Métriques temps réel
        classifications = [r.consensus['classification'] for r in results]
        recent_results = results[-10:]  # 10 derniers
        recent_classifications = [r.consensus['classification'] for r in recent_results]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📊 Total", len(results))
        
        with col2:
            positive_count = recent_classifications.count('Positif')
            st.metric("🟢 Positifs (10 derniers)", positive_count)
        
        with col3:
            negative_count = recent_classifications.count('Négatif')
            st.metric("🔴 Négatifs (10 derniers)", negative_count)
        
        with col4:
            neutral_count = recent_classifications.count('Neutre')
            st.metric("🟡 Neutres (10 derniers)", neutral_count)
        
        with col5:
            avg_score = np.mean([r.consensus['score'] for r in recent_results])
            st.metric("📈 Score Moyen", f"{avg_score:.3f}")
    
    with chart_container.container():
        # Graphique en temps réel
        if len(results) > 1:
            df_realtime = pd.DataFrame([
                {
                    'step': i,
                    'timestamp': r.timestamp,
                    'score': r.consensus['score'],
                    'classification': r.consensus['classification'],
                    'cumulative_avg': np.mean([res.consensus['score'] for res in results[:i+1]])
                }
                for i, r in enumerate(results)
            ])
            
            fig_realtime = go.Figure()
            
            # Ligne des scores individuels
            fig_realtime.add_trace(go.Scatter(
                x=df_realtime['step'],
                y=df_realtime['score'],
                mode='markers+lines',
                name='Score individuel',
                line=dict(color='lightblue', width=1),
                marker=dict(size=4)
            ))
            
            # Ligne de moyenne cumulative
            fig_realtime.add_trace(go.Scatter(
                x=df_realtime['step'],
                y=df_realtime['cumulative_avg'],
                mode='lines',
                name='Moyenne cumulative',
                line=dict(color='red', width=3)
            ))
            
            fig_realtime.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_realtime.update_layout(
                title="📈 Évolution en Temps Réel",
                xaxis_title="Message #",
                yaxis_title="Score de Sentiment",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_realtime, use_container_width=True)

def update_messages_display(recent_results: List[SentimentResult], messages_container):
    """Met à jour l'affichage des messages récents"""
    
    with messages_container.container():
        st.subheader("🔄 Messages Récents")
        
        for i, result in enumerate(reversed(recent_results)):
            sentiment_emoji = "🟢" if result.consensus['classification'] == 'Positif' else "🔴" if result.consensus['classification'] == 'Négatif' else "🟡"
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{len(recent_results)-i}.** {result.text}")
                
                with col2:
                    st.write(f"{sentiment_emoji} {result.consensus['classification']}")
                
                with col3:
                    st.write(f"Score: {result.consensus['score']:.3f}")
                
                st.divider()

def save_simulation_results(results: List[SentimentResult]):
    """Sauvegarde les résultats de simulation"""
    try:
        export_data = []
        for r in results:
            export_data.append({
                'text': r.text,
                'sentiment': r.consensus['classification'],
                'score': r.consensus['score'],
                'confidence': r.confidence,
                'timestamp': r.timestamp.isoformat(),
                'simulation_step': r.metadata.get('simulation_step', 0) if r.metadata else 0
            })
        
        # Sauvegarde JSON
        filename = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_data = json.dumps(export_data, ensure_ascii=False, indent=2)
        
        st.download_button(
            label="💾 Télécharger Simulation",
            data=json_data,
            file_name=filename,
            mime="application/json"
        )
        
        st.success(f"✅ Simulation sauvegardée: {len(results)} messages")
        
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde: {e}")





def check_system_health() -> Dict[str, Any]:
    """Vérifie la santé du système et des composants"""
    health_status = {
        'healthy': True,
        'issues': [],
        'components': {}
    }
    
    try:
        # Vérification de l'analyseur
        if 'analyzer' in globals():
            stats = analyzer.get_performance_stats()
            health_status['components']['analyzer'] = {
                'status': 'OK',
                'cache_size': stats['cache_size'],
                'components_loaded': stats['components_loaded']
            }
            
            # Vérifier les composants NLTK
            if not stats['components_loaded']['vader']:
                health_status['issues'].append("VADER n'est pas chargé correctement")
                health_status['healthy'] = False
            
            if stats['components_loaded']['french_lexicon'] < 1000:
                health_status['issues'].append("Lexique français incomplet")
                health_status['healthy'] = False
        else:
            health_status['issues'].append("Analyseur non initialisé")
            health_status['healthy'] = False
        
        # Vérification de la mémoire disponible
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            health_status['issues'].append(f"Mémoire critique: {memory_percent}% utilisée")
            health_status['healthy'] = False
        
        health_status['components']['memory'] = {
            'status': 'OK' if memory_percent < 80 else 'WARNING' if memory_percent < 90 else 'CRITICAL',
            'usage_percent': memory_percent
        }
        
    except ImportError:
        # psutil n'est pas disponible, continuer sans vérification mémoire
        health_status['components']['memory'] = {'status': 'UNAVAILABLE'}
    
    except Exception as e:
        health_status['issues'].append(f"Erreur lors de la vérification: {e}")
        health_status['healthy'] = False
    
    return health_status

def display_system_diagnostics():
    """Affiche les diagnostics système complets"""
    st.subheader("🔧 Diagnostics Système")
    
    # Informations de base
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🖥️ Informations Système**")
        try:
            import platform
            import sys
            
            system_info = {
                "OS": platform.system(),
                "Version OS": platform.version(),
                "Architecture": platform.architecture()[0],
                "Python": sys.version.split()[0],
                "Streamlit": st.__version__
            }
            
            for key, value in system_info.items():
                st.text(f"{key}: {value}")
        except Exception as e:
            st.error(f"Impossible d'obtenir les infos système: {e}")
    
    with col2:
        st.markdown("**📊 Statistiques de Performance**")
        if 'performance_stats' in st.session_state:
            stats = st.session_state.performance_stats
            uptime = datetime.now() - stats['start_time']
            
            perf_info = {
                "Temps de fonctionnement": str(uptime).split('.')[0],
                "Analyses totales": stats['total_analyses'],
                "Hits de cache": stats['cache_hits'],
                "Erreurs": stats['errors_count']
            }
            
            for key, value in perf_info.items():
                st.text(f"{key}: {value}")
    
    # État des composants
    st.markdown("**🔍 État des Composants**")
    
    try:
        health = check_system_health()
        
        if health['healthy']:
            st.success("✅ Tous les composants fonctionnent correctement")
        else:
            st.warning("⚠️ Problèmes détectés:")
            for issue in health['issues']:
                st.warning(f"• {issue}")
        
        # Détails des composants
        for component, details in health['components'].items():
            with st.expander(f"📋 {component.title()}"):
                st.json(details)
    
    except Exception as e:
        st.error(f"Erreur lors du diagnostic: {e}")
    
    # Tests de fonctionnalité
    st.markdown("**🧪 Tests de Fonctionnalité**")
    
    if st.button("🔬 Exécuter les tests"):
        run_functionality_tests()

def run_functionality_tests():
    """Exécute une série de tests pour vérifier le bon fonctionnement"""
    test_results = {}
    
    with st.spinner("Exécution des tests..."):
        
        # Test 1: Analyse de texte simple
        try:
            test_text = "Ce produit est excellent !"
            result = analyzer.analyze_sentiment_comprehensive(test_text)
            
            if result.consensus['classification'] == 'Positif':
                test_results['analyse_simple'] = '✅ PASS'
            else:
                test_results['analyse_simple'] = '❌ FAIL - Classification incorrecte'
        except Exception as e:
            test_results['analyse_simple'] = f'❌ ERROR - {e}'
        
        # Test 2: Préprocessing
        try:
            test_text = "J'adore cette app! 😊 #excellent"
            cleaned = analyzer.preprocess_text(test_text)
            
            if cleaned and len(cleaned) > 0:
                test_results['preprocessing'] = '✅ PASS'
            else:
                test_results['preprocessing'] = '❌ FAIL - Texte vide après nettoyage'
        except Exception as e:
            test_results['preprocessing'] = f'❌ ERROR - {e}'
        
        # Test 3: Lexique français
        try:
            lexicon_test = analyzer.french_lexicon.analyze_text("fantastique merveilleux")
            
            if lexicon_test['classification'] == 'Positif':
                test_results['lexique_francais'] = '✅ PASS'
            else:
                test_results['lexique_francais'] = '❌ FAIL - Lexique non fonctionnel'
        except Exception as e:
            test_results['lexique_francais'] = f'❌ ERROR - {e}'
        
        # Test 4: TF-IDF
        try:
            test_texts = ["texte positif excellent", "texte négatif horrible", "texte neutre correct"]
            keywords = analyzer.extract_keywords_tfidf(test_texts, max_features=5)
            
            if keywords and len(keywords) > 0:
                test_results['tfidf'] = '✅ PASS'
            else:
                test_results['tfidf'] = '❌ FAIL - Pas de mots-clés extraits'
        except Exception as e:
            test_results['tfidf'] = f'❌ ERROR - {e}'
        
        # Test 5: Cache
        try:
            cache_size_before = len(analyzer.cache)
            analyzer.preprocess_text("test cache functionality")
            cache_size_after = len(analyzer.cache)
            
            if cache_size_after > cache_size_before:
                test_results['cache'] = '✅ PASS'
            else:
                test_results['cache'] = '❌ FAIL - Cache non fonctionnel'
        except Exception as e:
            test_results['cache'] = f'❌ ERROR - {e}'
    
    # Affichage des résultats
    st.markdown("**📋 Résultats des Tests**")
    
    for test_name, result in test_results.items():
        if result.startswith('✅'):
            st.success(f"{test_name}: {result}")
        elif result.startswith('❌'):
            st.error(f"{test_name}: {result}")
        else:
            st.warning(f"{test_name}: {result}")
    
    # Score global
    passed_tests = sum(1 for result in test_results.values() if result.startswith('✅'))
    total_tests = len(test_results)
    score_percentage = (passed_tests / total_tests) * 100
    
    if score_percentage == 100:
        st.balloons()
        st.success(f"🎉 Tous les tests réussis! ({passed_tests}/{total_tests})")
    elif score_percentage >= 80:
        st.success(f"✅ Système opérationnel ({passed_tests}/{total_tests} tests réussis)")
    elif score_percentage >= 60:
        st.warning(f"⚠️ Fonctionnement partiel ({passed_tests}/{total_tests} tests réussis)")
    else:
        st.error(f"🚨 Problèmes critiques détectés ({passed_tests}/{total_tests} tests réussis)")

def generate_error_report(error: Exception) -> str:
    """Génère un rapport d'erreur détaillé pour le debugging"""
    
    import traceback
    import sys
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "system_info": {
            "python_version": sys.version,
            "platform": platform.system() if 'platform' in globals() else "Unknown"
        },
        "session_state": {},
        "performance_stats": st.session_state.get('performance_stats', {}),
        "analyzer_status": {}
    }
    
    # Informations de session (sans données sensibles)
    try:
        for key, value in st.session_state.items():
            if key not in ['simulation_results']:  # Exclure les gros objets
                if isinstance(value, (str, int, float, bool, list, dict)):
                    report["session_state"][key] = value
                else:
                    report["session_state"][key] = str(type(value))
    except Exception:
        report["session_state"] = "Erreur lors de la lecture de session_state"
    
    # Statut de l'analyseur
    try:
        if 'analyzer' in globals():
            stats = analyzer.get_performance_stats()
            report["analyzer_status"] = stats
    except Exception:
        report["analyzer_status"] = "Erreur lors de la lecture des stats de l'analyseur"
    
    return json.dumps(report, ensure_ascii=False, indent=2)

def display_footer():
    """Affiche le footer de l'application"""
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🎯 Analyseur de Sentiment Pro**")
        st.markdown("Version 2.0 - Optimisé pour le français")
    
    with col2:
        if 'performance_stats' in st.session_state:
            stats = st.session_state.performance_stats
            uptime = datetime.now() - stats['start_time']
            st.markdown(f"**⏱️ Uptime:** {str(uptime).split('.')[0]}")
            st.markdown(f"**📊 Analyses:** {stats['total_analyses']}")
    
    with col3:
        st.markdown("**🔗 Ressources**")
        st.markdown("• [Documentation](https://docs.streamlit.io)")
        st.markdown("• [Support](mailto:support@example.com)")

# Appel du footer dans le main() si nécessaire
def main():
    """Interface principale améliorée avec footer"""
    # ... (code main existant) ...
    
    # Conteneur principal
    main_container = st.container()
    
    with main_container:
        # Header principal
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Analyseur de Sentiment Pro</h1>
            <p>Analyse avancée de sentiment français avec IA et visualisations interactives</p>
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
                    "🔄 Temps Réel",
                    "📊 Analyse Comparative"
                ]
            )
            
            # Paramètres avancés
            with st.expander("🔧 Paramètres Avancés"):
                st.session_state.batch_size = st.slider("Taille de lot", 10, 100, 25)
                st.session_state.confidence_threshold = st.slider("Seuil de confiance", 0.0, 1.0, 0.5)
                st.session_state.use_parallel = st.checkbox("Traitement parallèle", value=True)
                st.session_state.export_format = st.selectbox("Format d'export", ["JSON", "CSV", "Excel"])
            
            # Gestion des lexiques
            with st.expander("📚 Gestion des Lexiques"):
                if st.button("💾 Exporter lexiques"):
                    create_lexicon_files()
                    st.success("Lexiques exportés vers ./lexicons/")
                
                st.info("Vous pouvez modifier les fichiers JSON dans le dossier lexicons/ pour personnaliser l'analyse.")
            
            # Informations de performance
            display_performance_info()
            
            # Informations système
            with st.expander("ℹ️ Informations Système"):
                st.markdown("""
                **🤖 Méthodes d'analyse:**
                - VADER (réseaux sociaux)
                - TextBlob (linguistique)
                - Lexique français étendu (4000+ mots)
                
                **📊 Fonctionnalités:**
                - Analyse temps réel
                - Traitement parallèle
                - Cache intelligent
                - Export multi-format
                - Gestion des négations
                - Intensificateurs français
                """)
        
        # Router vers le bon mode d'analyse
        if analysis_mode == "📝 Texte Manuel":
            handle_manual_text_analysis()
        elif analysis_mode == "📁 Fichier Upload":
            handle_file_upload_analysis()
        elif analysis_mode == "🐦 Twitter Simulation":
            handle_twitter_analysis()
        elif analysis_mode == "🔄 Temps Réel":
            handle_realtime_simulation()
        elif analysis_mode == "📊 Analyse Comparative":
            handle_comparative_analysis()
    
    # Footer
    display_footer()

# Nettoyage automatique à la fermeture
import atexit

def cleanup_on_exit():
    """Nettoyage automatique lors de la fermeture de l'application"""
    try:
        logger.info("Nettoyage automatique en cours...")
        
        # Fermer l'executor de threads si il existe
        if 'analyzer' in globals() and hasattr(analyzer, 'executor'):
            analyzer.executor.shutdown(wait=False)
        
        # Sauvegarder les statistiques de performance
        if 'performance_stats' in st.session_state:
            stats = st.session_state.performance_stats
            logger.info(f"Session terminée - Analyses: {stats['total_analyses']}, Erreurs: {stats['errors_count']}")
        
        logger.info("✅ Nettoyage terminé")
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")

# Enregistrer la fonction de nettoyage
atexit.register(cleanup_on_exit)

# Point d'entrée principal
# Point d'entrée principal avec gestion d'erreurs complète
if __name__ == "__main__":
    try:
        # Initialisation des composants système
        logger.info("Démarrage de l'application Analyseur de Sentiment Pro")
        
        # Vérification des dépendances critiques
        try:
            import streamlit as st
            import pandas as pd
            import numpy as np
            import plotly.express as px
            logger.info("✅ Toutes les dépendances principales sont disponibles")
        except ImportError as e:
            st.error(f"❌ Dépendance manquante: {e}")
            st.info("Veuillez installer les dépendances avec: pip install streamlit pandas numpy plotly nltk textblob scikit-learn wordcloud openpyxl")
            st.stop()
        
        # Initialisation de la session state
        if 'app_initialized' not in st.session_state:
            with st.spinner("🚀 Initialisation de l'application..."):
                st.session_state.app_initialized = True
                st.session_state.simulation_active = False
                st.session_state.simulation_results = []
                st.session_state.performance_stats = {
                    'total_analyses': 0,
                    'cache_hits': 0,
                    'errors_count': 0,
                    'start_time': datetime.now()
                }
                logger.info("✅ Session state initialisée")
        
        # Vérification de la santé du système
        system_health = check_system_health()
        if not system_health['healthy']:
            st.warning("⚠️ Certains composants ne sont pas optimaux:")
            for issue in system_health['issues']:
                st.warning(f"• {issue}")
            
            if st.button("🔄 Réessayer l'initialisation"):
                st.rerun()
        
        # Lancement de l'application principale
        main()
        
        # Mise à jour des statistiques de performance
        st.session_state.performance_stats['total_analyses'] += 1
        
    except KeyboardInterrupt:
        logger.info("Application interrompue par l'utilisateur")
        st.info("👋 Application fermée par l'utilisateur")
        
    except MemoryError:
        logger.error("Erreur de mémoire - données trop volumineuses")
        st.error("🚨 Erreur de mémoire: Les données sont trop volumineuses pour être traitées")
        st.info("💡 Suggestions:")
        st.info("• Réduisez la taille du fichier d'entrée")
        st.info("• Utilisez l'analyse par échantillon")
        st.info("• Redémarrez l'application")
        
    except Exception as e:
        logger.error(f"Erreur fatale dans main(): {e}")
        
        # Interface d'erreur user-friendly
        st.error("🚨 Une erreur inattendue s'est produite")
        
        with st.expander("🔍 Détails Techniques"):
            st.code(f"Type d'erreur: {type(e).__name__}")
            st.code(f"Message: {str(e)}")
            st.code(f"Timestamp: {datetime.now().isoformat()}")
        
        # Options de récupération
        st.subheader("🛠️ Options de Récupération")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Recharger l'application"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Réinitialiser la session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("📊 Statistiques système"):
                display_system_diagnostics()
        
        # Recommandations
        st.info("💡 **Conseils pour éviter les erreurs:**")
        st.info("• Vérifiez que tous les fichiers uploadés sont valides")
        st.info("• Réduisez la taille des données si elles sont volumineuses")
        st.info("• Contactez le support si l'erreur persiste")
        
        # Rapport d'erreur automatique (optionnel)
        if st.checkbox("📧 Envoyer un rapport d'erreur automatique"):
            error_report = generate_error_report(e)
            st.download_button(
                label="💾 Télécharger le rapport",
                data=error_report,
                file_name=f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

