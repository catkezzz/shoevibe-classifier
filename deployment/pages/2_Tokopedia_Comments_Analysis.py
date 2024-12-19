from helper import text_preprocessing, scrape_reviews_and_ratings
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import ast
# Pre-processing & Feature Engineering
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
import keras
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
try:
    from nltk.corpus import stopwords
    stpwds_id = list(set(stopwords.words('indonesian')))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stpwds_id = list(set(stopwords.words('indonesian')))
nltk.download('punkt_tab')
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

st.title("Tokopedia Comments Analysis 💬")
# ---- Custom CSS Styling ----
st.markdown("""
    <style>
    /* Header Style */
    .main-header {
        text-align: center;
        font-family: Arial, sans-serif;
        font-size: 35px;
        font-weight: bold;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    /* Input Label Style */
    .stTextInput > label > div > p {
        font-size: 20px;
        font-weight: 600;
    }
    /* Card Style */
    .sentiment-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sentiment-positive {
        color: #28a745;
    }
    .sentiment-negative {
        color: #dc3545;
    }
    /* Button Style */
    .stButton > button {
        font-size: 18px;
        padding: 10px 20px;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# ---- Title ----
st.markdown("<div class='main-header'>Tokopedia Shoe Sentiment Analysis</div>", unsafe_allow_html=True)

# ---- User Input ----
user_input = st.text_input("Enter Tokopedia Men Shoe product link for sentiment analysis")

# Stopwords and Stemmer
stpwds_id = list(set(stopwords.words('indonesian')))
stemmer = StemmerFactory().create_stemmer()
with open('slangwords_indonesian.txt') as f:
    slangwords_indonesian = ast.literal_eval(f.read())

# Load Model
model = tf.keras.models.load_model('nlp_model')

if st.button('Submit'):
    with st.spinner("Analyzing... Please wait!"):
        # Scrape and Preprocess
        the_data = scrape_reviews_and_ratings(user_input)
        the_data['Review_processed'] = the_data['Review'].apply(lambda x: text_preprocessing(x, slangwords_indonesian))
        y_pred_inf = model(the_data['Review_processed'])
        y_pred_inf = np.argmax(y_pred_inf.numpy(), axis=1)

        # Combine Results
        inffinal = pd.DataFrame({
            'Review_processed': the_data['Review_processed'],
            'Sentiment': y_pred_inf,
            'Sentiment_meaning': ['Positive' if x else 'Negative' for x in y_pred_inf]
        })
        st.session_state['inffinal'] = inffinal

        # Sentiment Count
        pos_count = sum(y_pred_inf)
        neg_count = len(y_pred_inf) - pos_count
        st.balloons()
        
        # ---- Results Cards ----
        col1, col2 = st.columns(2)
        
        # Positive Sentiment Card
        col1.markdown(f"""
        <div class='sentiment-card' style="text-align: center; padding: 20px; border: 1px solid #d4d4d4; border-radius: 10px; background-color: #f9f9f9;">
            <h3 style="color: #28a745; font-size: 24px; font-weight: bold;">Positive</h3>
            <p style="color: black; font-size: 20px; font-weight: bold;">{pos_count}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Negative Sentiment Card
        col2.markdown(f"""
        <div class='sentiment-card' style="text-align: center; padding: 20px; border: 1px solid #d4d4d4; border-radius: 10px; background-color: #f9f9f9;">
            <h3 style="color: black; font-size: 24px; font-weight: bold;">Negatives</h3>
            <p style="color: #dc3545; font-size: 20px; font-weight: bold;">{neg_count}</p>
        </div>
        """, unsafe_allow_html=True)

        # Overall Sentiment
        overall_sentiment = 'Positive' if pos_count > neg_count else 'Negative'
        st.subheader(f"Overall Sentiment: {overall_sentiment}")

# ---- Word Analysis Function ----
def generate_word_analysis(reviews, sentiment_label):
    # Count word frequencies
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X = vectorizer.fit_transform(reviews)
    word_freq = dict(zip(vectorizer.get_feature_names_out(), X.toarray().sum(axis=0)))
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    # Top 5 words
    st.subheader(f"Top 5 Words ({sentiment_label} Reviews):")
    top_5_words = sorted_word_freq[:5]
    for word, freq in top_5_words:
        st.write(f"**{word}**: {freq}")

    # Generate WordCloud
    st.subheader(f"WordCloud for {sentiment_label} Reviews")
    wc = WordCloud(width=800, height=400, stopwords=STOPWORDS, background_color='white').generate_from_frequencies(word_freq)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")  # Hide axes

    # Save the figure
    fig.savefig('wordcloud.png', format='png')

    # Display the figure in Streamlit
    st.pyplot(fig)


# ---- Word Analysis ----
try:
    if 'inffinal' in st.session_state:
        inffinal = st.session_state['inffinal']
        st.header("Word Frequency Analysis")

        pos_reviews = inffinal[inffinal['Sentiment'] == 1]['Review_processed']
        if not pos_reviews.empty:
            generate_word_analysis(pos_reviews, "Positive")
        
        neg_reviews = inffinal[inffinal['Sentiment'] == 0]['Review_processed']
        if not neg_reviews.empty:
            generate_word_analysis(neg_reviews, "Negative")
    else:
        st.error("Run sentiment analysis first!")
except Exception as e:
    st.error(f"Error: {e}")
