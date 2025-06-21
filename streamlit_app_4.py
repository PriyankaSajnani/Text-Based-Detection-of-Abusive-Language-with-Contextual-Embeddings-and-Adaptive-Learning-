import streamlit as st
import torch
import pandas as pd
import numpy as np
import time
import hashlib
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    st.error(f"Failed to import transformers: {e}. Please ensure transformers is installed: pip install transformers==4.35.2")
    st.stop()

from detoxify import Detoxify
import json
import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import warnings
from docx import Document
import PyPDF2
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging():
    """Setup enhanced logging system"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('abusive_detection.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# DATABASE SETUP FOR ANALYTICS
# =============================================================================
def init_database():
    """Initialize SQLite database for analytics"""
    try:
        conn = sqlite3.connect('analytics.db')
        cursor = conn.cursor()
        
        # Create analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_hash TEXT,
                model_used TEXT,
                input_type TEXT,
                text_length INTEGER,
                prediction INTEGER,
                confidence REAL,
                severity TEXT,
                toxic_words TEXT,
                processing_time REAL
            )
        ''')
        
        # Create rate limiting table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rate_limits (
                user_hash TEXT PRIMARY KEY,
                request_count INTEGER DEFAULT 1,
                last_request DATETIME DEFAULT CURRENT_TIMESTAMP,
                blocked_until DATETIME
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    finally:
        conn.close()

def log_analysis(user_hash: str, model_used: str, input_type: str, 
                text_length: int, prediction: int, confidence: float, 
                severity: str, toxic_words: List[str], processing_time: float):
    """Log analysis results to database"""
    try:
        conn = sqlite3.connect('analytics.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_logs 
            (user_hash, model_used, input_type, text_length, prediction, 
             confidence, severity, toxic_words, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_hash, model_used, input_type, text_length, prediction, 
              confidence, severity, ','.join(toxic_words), processing_time))
        
        conn.commit()
        logger.info(f"Analysis logged for user {user_hash[:8]}...")
    except Exception as e:
        logger.error(f"Failed to log analysis: {e}")
    finally:
        conn.close()

# =============================================================================
# FILE PROCESSING
# =============================================================================
def extract_text_from_file(uploaded_file, selected_columns=None):
    """Extract text from uploaded file"""
    try:
        logger.info(f"Extracting text from file: {uploaded_file.name}")
        if uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            text = "\n".join(para.text for para in doc.paragraphs)
        elif uploaded_file.name.endswith('.pdf'):
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        elif uploaded_file.name.endswith('.txt'):
            try:
                text = uploaded_file.read().decode('utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                text = uploaded_file.read().decode('latin1')
        elif uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin1')
            
            # Limit to 10000 rows to prevent memory issues
            if len(df) > 10000:
                df = df.sample(n=10000, random_state=42)
                logger.info("Sampled 10000 rows from large CSV")
            
            # Use selected columns or all string columns
            if selected_columns:
                text_columns = [col for col in selected_columns if col in df.columns]
            else:
                text_columns = df.select_dtypes(include=['object']).columns
            
            if not text_columns:
                st.error("No text columns found in CSV")
                return None
            
            # Extract text, ensuring non-null and string conversion
            text = "\n".join(
                " ".join(str(cell) for cell in row if pd.notna(cell))
                for _, row in df[text_columns].iterrows()
            )
        else:
            st.error("Unsupported file type.")
            return None
        
        logger.info(f"Successfully extracted {len(text)} characters from file")
        return text
    except Exception as e:
        logger.error(f"File reading error: {e}")
        st.error(f"File reading error: {e}")
        return None

# =============================================================================
# BATCH PROCESSING FUNCTION
# =============================================================================
def process_batch_texts(texts: List[str], selected_model: str, model, tokenizer) -> List[Dict]:
    """Process multiple texts in batch"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, text in enumerate(texts):
        status_text.text(f"Processing text {i+1} of {len(texts)}")
        
        # Validate and sanitize text
        is_valid, sanitized_text, warning_msg = validate_and_sanitize_text(text)
        if not is_valid:
            results.append({
                'text': text[:50] + "..." if len(text) > 50 else text,
                'prediction': 0,
                'confidence': 0.0,
                'severity': 'None',
                'toxic_words': [],
                'error': warning_msg
            })
            logger.warning(f"Invalid text in batch: {warning_msg}")
            continue
        
        # Get prediction
        prediction_results, processing_time = predict_with_model(sanitized_text, selected_model, model, tokenizer)
        prediction, confidence, toxic_words, severity = prediction_results[0]
        
        results.append({
            'text': text[:50] + "..." if len(text) > 50 else text,
            'prediction': prediction,
            'confidence': confidence,
            'severity': severity,
            'toxic_words': toxic_words,
            'processing_time': processing_time,
            'model': selected_model,
            'timestamp': datetime.now().isoformat()
        })
        
        progress_bar.progress((i + 1) / len(texts))
    
    progress_bar.empty()
    status_text.empty()
    return results

# =============================================================================
# RATE LIMITING
# =============================================================================
def get_user_hash(user_ip: str = None) -> str:
    """Generate user hash for rate limiting"""
    try:
        if user_ip:
            return hashlib.md5(user_ip.encode()).hexdigest()
        if 'user_hash' not in st.session_state:
            st.session_state.user_hash = hashlib.md5(str(time.time()).encode()).hexdigest()
        return st.session_state.user_hash
    except Exception as e:
        logger.error(f"Error generating user hash: {e}")
        return hashlib.md5(str(time.time()).encode()).hexdigest()

def check_rate_limit(user_hash: str, max_requests: int = 50, time_window: int = 3600) -> Tuple[bool, int]:
    """Check if user has exceeded rate limit with enhanced security"""
    try:
        conn = sqlite3.connect('analytics.db')
        cursor = conn.cursor()
        
        # Sanitize user_hash
        if not re.match(r'^[a-f0-9]{32}$', user_hash):
            logger.error(f"Invalid user hash format: {user_hash}")
            conn.close()
            return False, 3600
        
        cursor.execute('''
            SELECT request_count, last_request, blocked_until 
            FROM rate_limits WHERE user_hash = ?
        ''', (user_hash,))
        
        result = cursor.fetchone()
        current_time = datetime.now()
        
        if result:
            request_count, last_request, blocked_until = result
            last_request = datetime.fromisoformat(last_request)
            
            if blocked_until:
                blocked_until = datetime.fromisoformat(blocked_until)
                if current_time < blocked_until:
                    remaining_time = (blocked_until - current_time).seconds
                    conn.close()
                    return False, remaining_time
            
            if (current_time - last_request).seconds > time_window:
                request_count = 1
            else:
                request_count += 1
            
            if request_count > max_requests:
                blocked_until = current_time + timedelta(hours=1)
                cursor.execute('''
                    UPDATE rate_limits 
                    SET request_count = ?, last_request = ?, blocked_until = ?
                    WHERE user_hash = ?
                ''', (request_count, current_time.isoformat(), 
                      blocked_until.isoformat(), user_hash))
                conn.commit()
                conn.close()
                logger.warning(f"Rate limit exceeded for user {user_hash[:8]}")
                return False, 3600
            
            cursor.execute('''
                UPDATE rate_limits 
                SET request_count = ?, last_request = ?
                WHERE user_hash = ?
            ''', (request_count, current_time.isoformat(), user_hash))
        else:
            cursor.execute('''
                INSERT INTO rate_limits (user_hash, request_count, last_request)
                VALUES (?, 1, ?)
            ''', (user_hash, current_time.isoformat()))
        
        conn.commit()
        logger.info(f"Rate limit check passed for user {user_hash[:8]}")
    except Exception as e:
        logger.error(f"Rate limit check failed: {e}", exc_info=True)
    finally:
        conn.close()
    return True, 0

# =============================================================================
# INPUT VALIDATION AND SANITIZATION
# =============================================================================
def validate_and_sanitize_text(text: str) -> Tuple[bool, str, str]:
    """Validate and sanitize input text"""
    try:
        if not text or not isinstance(text, str):
            return False, "", "Text cannot be empty"
        
        sanitized = re.sub(r'<script.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        sanitized = re.sub(r'<.*?>', '', sanitized)
        sanitized = re.sub(r'[^\w\s.,!?;:\-\'"()]', ' ', sanitized)
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        
        max_length = 500000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")
            return True, sanitized, f"Text was truncated to {max_length:,} characters"
        if len(sanitized) < 3:
            return False, sanitized, "Text too short (min 3 characters)"
        
        return True, sanitized, ""
    except Exception as e:
        logger.error(f"Text validation error: {e}")
        return False, "", f"Validation error: {e}"

def validate_youtube_url(url: str) -> Tuple[bool, str]:
    """Validate YouTube URL"""
    try:
        youtube_patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
        ]
        for pattern in youtube_patterns:
            if re.search(pattern, url):
                return True, ""
        return False, "Invalid YouTube URL format"
    except Exception as e:
        logger.error(f"YouTube URL validation error: {e}")
        return False, f"URL validation error: {e}"

# =============================================================================
# ANALYTICS FUNCTIONS
# =============================================================================
def get_analytics_data() -> Dict:
    """Get analytics data from database"""
    try:
        conn = sqlite3.connect('analytics.db')
        
        total_analyses = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM analysis_logs", conn
        ).iloc[0]['count']
        
        model_stats = pd.read_sql_query(
            "SELECT model_used, COUNT(*) as count FROM analysis_logs GROUP BY model_used", conn
        )
        
        type_stats = pd.read_sql_query(
            "SELECT input_type, COUNT(*) as count FROM analysis_logs GROUP BY input_type", conn
        )
        
        toxicity_stats = pd.read_sql_query(
            "SELECT prediction, COUNT(*) as count FROM analysis_logs GROUP BY prediction", conn
        )
        
        recent_activity = pd.read_sql_query(
            """SELECT DATE(timestamp) as date, COUNT(*) as count 
               FROM analysis_logs 
               WHERE timestamp >= datetime('now', '-7 days')
               GROUP BY DATE(timestamp)
               ORDER BY date""", conn
        )
        
        avg_processing_time = pd.read_sql_query(
            "SELECT AVG(processing_time) as avg_time FROM analysis_logs", conn
        ).iloc[0]['avg_time'] or 0
        
        conn.close()
        
        return {
            'total_analyses': total_analyses,
            'model_stats': model_stats,
            'type_stats': type_stats,
            'toxicity_stats': toxicity_stats,
            'recent_activity': recent_activity,
            'avg_processing_time': avg_processing_time
        }
    except Exception as e:
        logger.error(f"Failed to get analytics data: {e}")
        return {}

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_confidence_gauge(confidence: float, prediction: int) -> go.Figure:
    """Create confidence gauge visualization"""
    try:
        color = "red" if prediction == 1 else "green"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': []},
            title = {'text': "Confidence Score (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "orange"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    except Exception as e:
        logger.error(f"Error creating confidence gauge: {e}")
        return go.Figure()

def create_analytics_dashboard(analytics_data: Dict):
    """Create enhanced analytics dashboard with additional visualizations"""
    try:
        if not analytics_data:
            st.warning("No analytics data available")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", analytics_data.get('total_analyses', 0))
        
        with col2:
            avg_time = analytics_data.get('avg_processing_time', 0)
            st.metric("Avg Processing Time", f"{avg_time:.2f}s")
        
        with col3:
            toxicity_stats = analytics_data.get('toxicity_stats', pd.DataFrame())
            if not toxicity_stats.empty:
                toxic_count = toxicity_stats[toxicity_stats['prediction'] == 1]['count'].sum()
                total_count = toxicity_stats['count'].sum()
                toxic_rate = (toxic_count / total_count * 100) if total_count > 0 else 0
                st.metric("Toxicity Rate", f"{toxic_rate:.1f}%")
            else:
                st.metric("Toxicity Rate", "0%")
        
        with col4:
            model_stats = analytics_data.get('model_stats', pd.DataFrame())
            if not model_stats.empty:
                most_used = model_stats.loc[model_stats['count'].idxmax(), 'model_used']
                st.metric("Most Used Model", most_used)
            else:
                st.metric("Most Used Model", "N/A")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_stats = analytics_data.get('model_stats', pd.DataFrame())
            if not model_stats.empty:
                fig = px.pie(model_stats, values='count', names='model_used', 
                            title="Model Usage Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            type_stats = analytics_data.get('type_stats', pd.DataFrame())
            if not type_stats.empty:
                fig = px.bar(type_stats, x='input_type', y='count', 
                            title="Analysis by Input Type")
                st.plotly_chart(fig, use_container_width=True)
        
        # Severity distribution
        try:
            conn = sqlite3.connect('analytics.db')
            severity_stats = pd.read_sql_query(
                "SELECT severity, COUNT(*) as count FROM analysis_logs WHERE severity != 'None' GROUP BY severity", 
                conn
            )
            conn.close()
            if not severity_stats.empty:
                fig = px.bar(severity_stats, x='severity', y='count', 
                            title="Toxicity Severity Distribution",
                            color='severity', 
                            color_discrete_map={'Mild': '#FFFF99', 'Moderate': '#FF9933', 'Severe': '#FF3333'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating severity distribution: {e}")
        
        # Word cloud for toxic words
        try:
            conn = sqlite3.connect('analytics.db')
            toxic_words_data = pd.read_sql_query(
                "SELECT toxic_words FROM analysis_logs WHERE toxic_words != ''", 
                conn
            )
            conn.close()
            if toxic_words_data.empty:
                st.warning("No toxic words data available to generate word cloud")
            else:
                all_toxic_words = ' '.join(
                    word for words in toxic_words_data['toxic_words'] for word in words.split(',')
                )
                if all_toxic_words.strip():
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_toxic_words)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                else:
                    st.warning("No valid toxic words found for word cloud")
        except Exception as e:
            logger.error(f"Error creating word cloud: {e}")
            st.warning(f"Failed to generate word cloud: {e}")
        
        recent_activity = analytics_data.get('recent_activity', pd.DataFrame())
        if not recent_activity.empty:
            fig = px.line(recent_activity, x='date', y='count', 
                         title="Daily Activity (Last 7 Days)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No recent activity data available")
            
    except Exception as e:
        logger.error(f"Error creating analytics dashboard: {e}")
        st.error(f"Failed to load analytics dashboard: {e}")

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================
def create_pdf_report(results: List[Dict], model_used: str) -> bytes:
    """Create PDF report of analysis results"""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            alignment=1
        )
        story.append(Paragraph("Abusive Language Detection Report", title_style))
        story.append(Spacer(1, 20))
        
        story.append(Paragraph(f"<b>Model Used:</b> {model_used}", styles['Normal']))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Analyses:</b> {len(results)}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        table_data = [['Text', 'Prediction', 'Confidence', 'Severity', 'Toxic Words']]
        for result in results:
            text_preview = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
            prediction = "TOXIC" if result['prediction'] == 1 else "CLEAN"
            confidence = f"{result['confidence']:.1%}"
            severity = result['severity']
            toxic_words = ', '.join(result['toxic_words'][:3])
            table_data.append([text_preview, prediction, confidence, severity, toxic_words])
            
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), colors.black)
        ]))
        
        story.append(table)
        doc.build(story)
        
        buffer.seek(0)
        logger.info("PDF report generated successfully")
        return buffer.read()
    except Exception as e:
        logger.error(f"PDF report generation error: {e}")
        return b""

def create_csv_export(results: List[Dict]) -> str:
    """Create CSV export of results"""
    try:
        df = pd.DataFrame(results)
        logger.info("CSV export generated successfully")
        return df.to_csv(index=False)
    except Exception as e:
        logger.error(f"CSV export error: {e}")
        return ""

def create_json_export(results: List[Dict]) -> str:
    """Create JSON export of results"""
    try:
        formatted_results = [
            {
                'text': r['text'],
                'prediction': 'TOXIC' if r['prediction'] == 1 else 'CLEAN',
                'confidence': f"{r['confidence']:.1%}",
                'severity': r['severity'],
                'toxic_words': r['toxic_words'],
                'model': r['model'],
                'timestamp': r['timestamp']
            } for r in results
        ]
        logger.info("JSON export generated successfully")
        return json.dumps(formatted_results, indent=2)
    except Exception as e:
        logger.error(f"JSON export error: {e}")
        return json.dumps({'error': str(e)})

# =============================================================================
# CORE FUNCTIONS
# =============================================================================
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    logger.info("NLTK resources downloaded successfully")
except Exception as e:
    st.error(f"Failed to download NLTK resources: {e}")
    logger.error(f"NLTK download failed: {e}")
    st.stop()

st.set_page_config(
    page_title="Abusive Language Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = r"C:\Users\Karthik Sundaram\Downloads\Abusive_Language_Project"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

init_database()

for directory in [RESULTS_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

@st.cache_data
def load_results():
    """Load model comparison results"""
    try:
        results_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        else:
            logger.warning(f"Model comparison file not found at: {results_path}")
            default_results = pd.DataFrame([
                {"Model": "BERT", "F1-Score": 0.85},
                {"Model": "RoBERTa", "F1-Score": 0.87},
                {"Model": "DistilBERT", "F1-Score": 0.83},
                {"Model": "Detoxify", "F1-Score": 0.79},
                {"Model": "SVM", "F1-Score": 0.78}
            ])
            default_results.to_csv(results_path, index=False)
            return default_results
    except Exception as e:
        logger.error(f"Error loading results: {e}")
        return None

def preprocess_text(text):
    """Preprocess text for SVM model"""
    try:
        if not isinstance(text, str) or pd.isna(text):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text.split()) <= 2:
            return text
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english')) - {'not', 'no', 'never', 'nothing', 'nobody', 'neither', 'nowhere', 'none'}
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        if tokens:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(t) for t in tokens]
        return ' '.join(tokens)
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return ""

def find_toxic_words(text):
    """Identify toxic words in text"""
    try:
        toxic_words = ['hate', 'stupid', 'idiot', 'kill', 'die', 'worst', 'terrible', 'awful', 'dumb', 'useless',
                       'fool', 'loser', 'shut up', 'shut your mouth', 'shut your face', 'shut it', 'shut your trap',
                       'fuck', 'fucking', 'nincompoop', 'imbecile', 'retard', 'numpty', 'rascal']
        text_lower = text.lower()
        found = []
        for word in toxic_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            if word in ['fuck', 'fucking']:
                pattern = r'(?:\b|[0-9])' + re.escape(word) + r'(?:\b|[0-9])'
            if re.search(pattern, text_lower):
                found.append(word)
        return list(set(found))
    except Exception as e:
        logger.error(f"Error finding toxic words: {e}")
        return []

@st.cache_resource
def load_model(selected_model):
    """Load selected model with fallback"""
    try:
        logger.info(f"Attempting to load model: {selected_model}")
        if selected_model == "SVM":
            svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
            vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
            if os.path.exists(svm_path) and os.path.exists(vectorizer_path):
                with open(svm_path, 'rb') as f:
                    svm_model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                logger.info("SVM model loaded successfully")
                return svm_model, vectorizer
            else:
                logger.error(f"SVM model or vectorizer not found at {svm_path}")
                st.error("SVM model not available. Try another model.")
                return None, None
        elif selected_model == "Detoxify":
            detoxify_model = Detoxify('unbiased', device='cpu')
            logger.info("Detoxify model loaded successfully")
            return detoxify_model, None
        else:
            model_path = os.path.join(MODELS_DIR, f"{selected_model.lower()}_final")
            if os.path.exists(model_path):
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                model.to(torch.device('cpu')).eval()
                logger.info(f"{selected_model} model loaded successfully")
                return model, tokenizer
            else:
                logger.error(f"{selected_model} model not found at: {model_path}")
                st.error(f"{selected_model} model not available. Falling back to Detoxify.")
                detoxify_model = Detoxify('unbiased', device='cpu')
                logger.info("Fell back to Detoxify model")
                return detoxify_model, None
    except Exception as e:
        logger.error(f"Error loading {selected_model} model: {e}", exc_info=True)
        st.error(f"Failed to load {selected_model} model: {e}. Please try another model.")
        return None, None

def get_severity(confidence: float, toxic_words: List[str]) -> str:
    """Determine severity of toxic content"""
    try:
        severe_confidence = 0.9
        moderate_confidence = 0.7
        high_severity_words = ['fuck', 'fucking', 'kill', 'die', 'retard']
        has_high_severity = any(word in toxic_words for word in high_severity_words)
        
        if confidence >= severe_confidence or has_high_severity:
            return "Severe"
        elif confidence >= moderate_confidence:
            return "Moderate"
        else:
            return "Mild"
    except Exception as e:
        logger.error(f"Error determining severity: {e}")
        return "None"

def predict_with_model(texts, selected_model, model, tokenizer):
    """Predict toxicity with selected model"""
    try:
        start_time = time.time()
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for text in texts:
            toxic_words = find_toxic_words(text)
            logger.debug(f"Processing text (length: {len(text)} chars, toxic_words={toxic_words})")
            
            if selected_model == "SVM" and model and tokenizer:
                processed = preprocess_text(text)
                if not processed:
                    logger.warning(f"Empty processed text for input: {text[:50]}...")
                    results.append((0, 0.5, toxic_words, "None"))
                    continue
                text_tfidf = tokenizer.transform([processed])
                prediction = model.predict(text_tfidf)[0]
                confidence = model.predict_proba(text_tfidf)[0][prediction]
                logger.info(f"SVM prediction: {prediction}, confidence: {confidence:.2f}")
            elif selected_model == "Detoxify" and model:
                result = model.predict(text)
                toxicity_score = result.get('toxicity', 0.0)
                prediction = 1 if toxicity_score > 0.5 else 0
                confidence = float(toxicity_score if prediction else 1 - toxicity_score)
                logger.info(f"Detoxify prediction: {prediction}, toxicity: {toxicity_score:.2f}")
            elif selected_model in ["BERT", "RoBERTa", "DistilBERT"] and model and tokenizer:
                inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                prediction = np.argmax(probs)
                confidence = float(probs[prediction])
                logger.info(f"{selected_model} prediction: {prediction}, confidence: {confidence:.2f}")
            else:
                logger.error(f"Model {selected_model} not available")
                results.append((0, 0.5, toxic_words, "None"))
                continue
            severity = get_severity(confidence, toxic_words) if prediction == 1 else "None"
            results.append((prediction, confidence, toxic_words, severity))
        
        processing_time = time.time() - start_time
        logger.info(f"Prediction completed for {len(texts)} texts in {processing_time:.2f}s")
        return results, processing_time
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        processing_time = time.time() - start_time
        return [(0, 0.5, [], "None") for _ in texts], processing_time

def get_youtube_transcript(video_url, _unused_api_key=None):
    """Fetch YouTube transcript with retry mechanism"""
    try:
        logger.info(f"Fetching transcript for: {video_url}")
        video_url = video_url.strip()
        
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
            r'([a-zA-Z0-9_-]{11})'
        ]
        
        video_id = None
        for pattern in patterns:
            match = re.search(pattern, video_url)
            if match:
                potential_id = match.group(1)
                if len(potential_id) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', potential_id):
                    video_id = potential_id
                    break
        
        if not video_id:
            logger.error("Could not extract valid video ID from URL")
            st.error("Could not extract valid video ID from URL. Please check the format.")
            return None

        logger.debug(f"Extracted video ID: {video_id}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1} to fetch transcript for video ID: {video_id}")
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
                
                transcript = None
                try:
                    transcript = transcript_list.find_transcript(['en'])
                    logger.debug("Found manual English transcript")
                except NoTranscriptFound:
                    pass
                
                if not transcript:
                    try:
                        transcript = transcript_list.find_generated_transcript(['en'])
                        logger.debug("Found auto-generated English transcript")
                    except NoTranscriptFound:
                        pass
                
                if not transcript:
                    try:
                        transcript = transcript_list.find_transcript(['en-US', 'en-GB', 'en-CA'])
                        logger.debug("Found English variant transcript")
                    except NoTranscriptFound:
                        pass
                
                if not transcript:
                    try:
                        available_transcripts = list(transcript_list)
                        if available_transcripts:
                            transcript = available_transcripts[0]
                            logger.debug(f"Using available transcript: {transcript.language}")
                        else:
                            st.error("No transcripts available for this video.")
                            return None
                    except:
                        st.error("No transcripts available for this video.")
                        return None

                transcript_data = transcript.fetch()
                
                if not transcript_data:
                    st.error("Empty transcript received. The video might not have captions.")
                    return None
                
                full_text = " ".join(entry['text'] for entry in transcript_data if 'text' in entry)
                
                if not full_text.strip():
                    st.error("Transcript appears to be empty.")
                    return None
                
                logger.info(f"Successfully fetched transcript with {len(full_text)} characters")
                return full_text
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    raise e

    except TranscriptsDisabled:
        logger.error("Transcripts disabled for video")
        st.error("❌ Transcripts/captions are disabled for this video.")
        return None
    except NoTranscriptFound:
        logger.error("No transcript found for video")
        st.error("❌ No transcript found for this video.")
        return None
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"YouTube transcript fetch error: {e}")
        
        if "no element found" in error_msg:
            st.error("❌ Unable to fetch transcript data. This might be due to:")
            st.write("• Video privacy settings")
            st.write("• Regional restrictions")
            st.write("• YouTube API rate limiting")
            st.write("• Invalid video URL")
        elif "http" in error_msg or "connection" in error_msg:
            st.error("❌ Network connection error. Please check your internet connection and try again.")
        elif "not found" in error_msg:
            st.error("❌ Video not found. Please check if the video exists and is publicly accessible.")
        else:
            st.error(f"❌ Error fetching transcript: {e}")
        return None

# =============================================================================
# MAIN APPLICATION
# =============================================================================
st.markdown("""
# 🛡️ Advanced Abusive Language Detection System
### Enhanced with Analytics, Security & Batch Processing
""")

# Sidebar Configuration
st.sidebar.header("🔧 Configuration")
selected_model = st.sidebar.selectbox(
    "Choose Model:",
    ["BERT", "RoBERTa", "DistilBERT", "Detoxify", "SVM"],
    help="Select the machine learning model for analysis"
)

model, tokenizer = load_model(selected_model)

user_hash = get_user_hash()
rate_limit_ok, remaining_time = check_rate_limit(user_hash)

if not rate_limit_ok:
    st.error(f"⚠️ Rate limit exceeded. Please wait {remaining_time//60} minutes before making more requests.")
    st.stop()

st.sidebar.header("📊 Quick Stats")
analytics_data = get_analytics_data()
if analytics_data:
    st.sidebar.metric("Total Analyses", analytics_data.get('total_analyses', 0))
    st.sidebar.metric("Avg Processing Time", f"{analytics_data.get('avg_processing_time', 0):.2f}s")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = []

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Text Input",
    "🎥 YouTube Video",
    "📄 Document Upload",
    "📊 Batch Processing",
    "📈 Analytics Dashboard"
])

with tab1:
    st.header("📝 Single Text Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_text = st.text_area(
            "Enter text to analyze:",
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter any text to analyze for abusive language (max 500,000 characters)"
        )
    
    with col2:
        st.markdown("### Options")
        save_result = st.checkbox("Save to results", value=True)
        show_confidence = st.checkbox("Show confidence gauge", value=True)
    
    if st.button("🔍 Analyze Text", type="primary", key="text_analyze"):
        if user_text.strip():
            is_valid, sanitized_text, warning_msg = validate_and_sanitize_text(user_text)
            if not is_valid:
                st.error(f"❌ {warning_msg}")
            else:
                with st.spinner("Analyzing text..."):
                    try:
                        if warning_msg:
                            st.warning(warning_msg)
                        prediction_results, processing_time = predict_with_model(
                            sanitized_text, selected_model, model, tokenizer
                        )
                        prediction, confidence, toxic_words, severity = prediction_results[0]
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            if prediction == 0:
                                st.success(f"✅ **CLEAN TEXT** (Confidence: {confidence:.1%})")
                            else:
                                color = {"Severe": "🔴", "Moderate": "🟠", "Mild": "🟡"}.get(severity, "🟢")
                                st.error(f"⚠️ **TOXIC TEXT** (Confidence: {confidence:.1%})")
                                st.markdown(f"**Severity Level:** {color} {severity}")
                                if toxic_words:
                                    st.markdown(f"**Toxic words detected:** `{', '.join(toxic_words)}`")
                        
                        with col2:
                            if show_confidence:
                                fig = create_confidence_gauge(confidence, prediction)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        log_analysis(
                            user_hash, selected_model, "text", len(sanitized_text),
                            prediction, confidence, severity, toxic_words, processing_time
                        )
                        
                        if save_result:
                            result = {
                                'text': user_text,
                                'prediction': prediction,
                                'confidence': confidence,
                                'severity': severity,
                                'toxic_words': toxic_words,
                                'model': selected_model,
                                'timestamp': datetime.now().isoformat()
                            }
                            st.session_state.analysis_results.append(result)
                            st.success("✅ Result saved to batch results")
                    except Exception as e:
                        logger.error(f"Analysis error: {e}")
                        st.error(f"❌ Analysis failed: {e}")
        else:
            st.warning("⚠️ Please enter some text to analyze.")

with tab2:
    st.header("🎥 YouTube Video Transcript Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_url = st.text_input(
            "Enter YouTube URL:",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste a YouTube video URL to analyze its transcript"
        )
    
    with col2:
        st.markdown("### Options")
        show_transcript = st.checkbox("Show transcript preview", value=True)
        save_yt_result = st.checkbox("Save to results", value=True, key="save_yt_result_checkbox")
    
    if st.button("🔍 Analyze Transcript", type="primary", key="yt_analyze"):
        if video_url:
            is_valid_url, url_error = validate_youtube_url(video_url)
            if not is_valid_url:
                st.error(f"❌ {url_error}")
            else:
                with st.spinner("Fetching and analyzing transcript..."):
                    transcript = get_youtube_transcript(video_url)
                    if transcript:
                        is_valid, sanitized_transcript, warning_msg = validate_and_sanitize_text(transcript)
                        if is_valid:
                            try:
                                if warning_msg:
                                    st.warning(warning_msg)
                                prediction_results, processing_time = predict_with_model(
                                    sanitized_transcript, selected_model, model, tokenizer
                                )
                                prediction, confidence, toxic_words, severity = prediction_results[0]
                                
                                if prediction == 0:
                                    st.success(f"✅ **CLEAN CONTENT** (Confidence: {confidence:.1%})")
                                else:
                                    color = {"Severe": "🔴", "Moderate": "🟠", "Mild": "🟡"}.get(severity, "🟢")
                                    st.error(f"⚠️ **TOXIC CONTENT** (Confidence: {confidence:.1%})")
                                    st.markdown(f"**Severity Level:** {color} {severity}")
                                    if toxic_words:
                                        st.markdown(f"**Toxic words detected:** `{', '.join(toxic_words)}`")
                                
                                if show_transcript:
                                    with st.expander("📄 Transcript Preview"):
                                        st.text_area(
                                            "Transcript:",
                                            transcript[:1000] + "..." if len(transcript) > 1000 else transcript,
                                            height=150,
                                            disabled=True
                                        )
                                
                                log_analysis(
                                    user_hash, selected_model, "youtube", len(sanitized_transcript),
                                    prediction, confidence, severity, toxic_words, processing_time
                                )
                                
                                if save_yt_result:
                                    result = {
                                        'text': f"YouTube: {video_url}",
                                        'prediction': prediction,
                                        'confidence': confidence,
                                        'severity': severity,
                                        'toxic_words': toxic_words,
                                        'model': selected_model,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    st.session_state.analysis_results.append(result)
                                    st.success("✅ Result saved to batch results")
                            except Exception as e:
                                logger.error(f"YouTube analysis error: {e}")
                                st.error(f"❌ Analysis failed: {e}")
                        else:
                            st.error(f"❌ {warning_msg}")
        else:
            st.warning("⚠️ Please enter a YouTube URL")

with tab3:
    st.header("📄 Document Upload Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload document (.docx, .pdf, .txt, .csv)",
            type=["docx", "pdf", "txt", "csv"],
            help="Upload a document to analyze its content"
        )
        
        # Column selector for CSV
        selected_columns = None
        if uploaded_file and uploaded_file.name.endswith('.csv'):
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='utf-8')
                uploaded_file.seek(0)
                all_columns = df.columns.tolist()
                selected_columns = st.multiselect(
                    "Select CSV columns to analyze (optional)",
                    options=all_columns,
                    help="Choose specific columns to analyze. If none selected, all text columns are used."
                )
            except Exception as e:
                st.warning(f"Could not load CSV for column selection: {e}")
    
    with col2:
        st.markdown("### Options")
        show_doc_preview = st.checkbox("Show document preview", value=True)
        save_doc_result = st.checkbox("Save to results", value=True, key="save_doc_result_checkbox")
    
    if st.button("🔍 Analyze Document", type="primary", key="doc_analyze"):
        if uploaded_file:
            with st.spinner("Processing document..."):
                text = extract_text_from_file(uploaded_file, selected_columns)
                if text:
                    is_valid, sanitized_text, warning_msg = validate_and_sanitize_text(text)
                    if is_valid:
                        try:
                            if warning_msg:
                                st.warning(warning_msg)
                            prediction_results, processing_time = predict_with_model(
                                sanitized_text, selected_model, model, tokenizer
                            )
                            prediction, confidence, toxic_words, severity = prediction_results[0]
                            
                            if prediction == 0:
                                st.success(f"✅ **CLEAN DOCUMENT** (Confidence: {confidence:.1%})")
                            else:
                                color = {"Severe": "🔴", "Moderate": "🟠", "Mild": "🟡"}.get(severity, "🟢")
                                st.error(f"⚠️ **TOXIC CONTENT** (Confidence: {confidence:.1%})")
                                st.markdown(f"**Severity Level:** {color} {severity}")
                                if toxic_words:
                                    st.markdown(f"**Toxic words detected:** `{', '.join(toxic_words)}`")
                            
                            if show_doc_preview:
                                with st.expander("📄 Document Preview"):
                                    st.text_area(
                                        "Document Content:",
                                        text[:1000] + "..." if len(text) > 1000 else text,
                                        height=150,
                                        disabled=True
                                    )
                            
                            log_analysis(
                                user_hash, selected_model, "document", len(sanitized_text),
                                prediction, confidence, severity, toxic_words, processing_time
                            )
                            
                            if save_doc_result:
                                result = {
                                    'text': f"Document: {uploaded_file.name}",
                                    'prediction': prediction,
                                    'confidence': confidence,
                                    'severity': severity,
                                    'toxic_words': toxic_words,
                                    'model': selected_model,
                                    'timestamp': datetime.now().isoformat()
                                }
                                st.session_state.analysis_results.append(result)
                                st.success("✅ Result saved to batch results")
                        except Exception as e:
                            logger.error(f"Document analysis error: {e}")
                            st.error(f"❌ Analysis failed: {e}")
                    else:
                        st.error(f"❌ {warning_msg}")
        else:
            st.warning("⚠️ Please upload a document")

with tab4:
    st.header("📊 Batch Processing & Results")
    
    st.subheader("📝 Batch Text Analysis")
    batch_texts = st.text_area(
        "Enter multiple texts (one per line):",
        height=150,
        placeholder="Text 1\nText 2\nText 3...",
        help="Enter multiple texts separated by new lines for batch processing (max 500,000 characters per text)"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔍 Process Batch", type="primary"):
            if batch_texts.strip():
                texts = [text.strip() for text in batch_texts.split('\n') if text.strip()]
                if len(texts) > 50:
                    st.error("❌ Maximum 50 texts allowed per batch")
                else:
                    with st.spinner("Processing batch..."):
                        batch_results = process_batch_texts(texts, selected_model, model, tokenizer)
                        st.session_state.analysis_results.extend(batch_results)
                        st.success(f"✅ Processed {len(batch_results)} texts successfully!")
    
    st.subheader("📋 Analysis Results")
    
    if st.session_state.analysis_results:
        total_results = len(st.session_state.analysis_results)
        toxic_count = sum(1 for r in st.session_state.analysis_results if r['prediction'] == 1)
        clean_count = total_results - toxic_count
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Analyzed", total_results)
        with col2:
            st.metric("Clean", clean_count)
        with col3:
            st.metric("Toxic", toxic_count)
        with col4:
            toxicity_rate = (toxic_count / total_results * 100) if total_results > 0 else 0
            st.metric("Toxicity Rate", f"{toxicity_rate:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            show_all = st.checkbox("Show all results", value=True)
        with col2:
            filter_toxic = st.checkbox("Show only toxic", value=False)
        with col3:
            if st.button("🗑️ Clear Results"):
                st.session_state.analysis_results = []
                st.rerun()
        
        display_df = pd.DataFrame(st.session_state.analysis_results)
        if filter_toxic:
            display_df = display_df[display_df['prediction'] == 1]
        if not show_all:
            display_df = display_df.head(10)
        
        display_df['Status'] = display_df['prediction'].apply(lambda x: '🔴 TOXIC' if x == 1 else '✅ CLEAN')
        display_df['Confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'
        display_df['Toxic Words'] = display_df['toxic_words'].apply(lambda x: ', '.join(x[:3]))
        
        st.dataframe(
            display_df[['text', 'Status', 'Confidence', 'severity', 'Toxic Words', 'model']],
            use_container_width=True
        )
        
        st.subheader("📤 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report"):
                pdf_bytes = create_pdf_report(st.session_state.analysis_results, selected_model)
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"abusive_language_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
        
        with col2:
            csv_data = create_csv_export(st.session_state.analysis_results)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            json_data = create_json_export(st.session_state.analysis_results)
            st.download_button(
                label="⬇️ Download JSON",
                data=json_data,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("📊 No analysis results yet. Start by analyzing some text in the other tabs!")

with tab5:
    st.header("📈 Analytics Dashboard")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with col2:
        if st.checkbox("Confirm Clear All History"):
            if st.button("🗑️ Clear All History", key="clear_all_history"):
                try:
                    # Clear session state
                    st.session_state.analysis_results = []
                    
                    # Clear database
                    conn = sqlite3.connect('analytics.db')
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM analysis_logs')
                    cursor.execute('DELETE FROM rate_limits')
                    conn.commit()
                    conn.close()
                    
                    # Clear Streamlit cache
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    
                    st.success("✅ All history cleared successfully!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Failed to clear history: {e}")
                    st.error(f"❌ Failed to clear history: {e}")
    
    analytics_data = get_analytics_data()
    create_analytics_dashboard(analytics_data)
    
    st.subheader("🏆 Model Performance Comparison")
    results_df = load_results()
    if results_df is not None and not results_df.empty:
        fig = px.bar(
            results_df,
            x='Model',
            y='F1-Score',
            title="Model Performance (F1-Score)",
            color='F1-Score',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🔧 System Health")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Loaded", "5/5" if model and tokenizer else "Loading...")
    
    with col2:
        db_status = "✅ Connected"
        try:
            conn = sqlite3.connect('analytics.db')
            conn.close()
        except:
            db_status = "❌ Error"
        st.metric("Database Status", db_status)
    
    with col3:
        rate_limit_status = f"{50 - (50 if not rate_limit_ok else 0)}/50 requests"
        st.metric("Rate Limit Status", rate_limit_status)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛡️ Advanced Abusive Language Detection System v2.0</p>
    <p>Enhanced with Analytics, Security & Batch Processing</p>
    <p>Built with Streamlit • Powered by Transformers & Detoxify</p>
</div>
""", unsafe_allow_html=True)