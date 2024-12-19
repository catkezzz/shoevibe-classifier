import re
import os
os.system('bash setup.sh')
import ast
import json
import warnings
from time import sleep

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import tensorflow_hub as tf_hub
from wordcloud import STOPWORDS, WordCloud
import nltk
from nltk.corpus import stopwords
try:
    stpwds_id = list(set(stopwords.words('indonesian')))
except LookupError:
    nltk.download('stopwords')
    stpwds_id = list(set(stopwords.words('indonesian')))
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt_tab')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, TextVectorization, Reshape, Input, LSTM, Dropout, Dense, Bidirectional
from keras.callbacks import EarlyStopping

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Define Stopwords
stpwds_id = list(set(stopwords.words('indonesian')))
# Define Stemming
stemmer = StemmerFactory().create_stemmer()
# Open slangwords_indonesian.txt
with open('slangwords_indonesian.txt') as f:
    data = f.read()
slangwords_indonesian =  ast.literal_eval(data)


# Membuat suatu fungsi yang berisi full preprocessing step
def text_preprocessing(text, slangwords_indonesian):
    # Fungsi untuk mengubah teks menjadi huruf kecil
    def lower(text):
        return text.lower()
    # Fungsi untuk mengganti abbreviation
    def check_slang(text):
        temp = []
        for slang in text.split():
            if slang in slangwords_indonesian:
                temp.append(slangwords_indonesian[slang])
            else:
                temp.append(slang)
        return " ".join(temp)
    # Fungsi untuk menghapus tanda baca, newlines, dan whitespace ekstra
    def check_punctuation(text):
        # Non-letter removal (seperti emoticon, symbol (seperti μ, $, 兀), dan lain-lain
        text = re.sub("[^a-zA-Z]", ' ', text)
        # Hashtags removal
        text = re.sub("#[A-Za-z0-9_]+", " ", text)
        # Mention removal
        text = re.sub("@[A-Za-z0-9_]+", " ", text)
        # Menghapus teks yang ada di dalam tanda kurung siku ([...]) dalam document
        text = re.sub('\[[^]]*\]', ' ', text)
        # Mengganti setiap baris baru (newline) dengan spasi
        text = re.sub(r"\\n", " ", text)
        # Menghapus whitespace ekstra di awal dan akhir token
        text = text.strip()
        # Menghapus spasi berlebih di antara kata-kata (hanya menyisakan satu spasi antar kata)
        text = ' '.join(text.split())
        return text
    # Fungsi untuk tokenisasi, menghapus stopwords, dan stemming
    def token_stopwords_stem(text):
        # Tokenization
        tokens = word_tokenize(text)
        # Stopwords removal
        tokens = [word for word in tokens if word not in stpwds_id]
        # Stemming
        tokens = [stemmer.stem(word) for word in tokens]
        # Combining Tokens
        text = ' '.join(tokens)  # Menggunakan 'text' untuk menggabungkan kembali tokens
        return text
    # Proses Preprocessing
    text = lower(text)
    text = check_slang(text)
    text = check_punctuation(text)
    text = token_stopwords_stem(text)
    return text

# Fungsi Untuk Scrape Review Tokopedia
def scrape_reviews_and_ratings(product_urls):
    reviews = []
    ratings = []
    options = webdriver.ChromeOptions()
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--headless")  # Run browser in headless mode
    options.add_argument("--disable-gpu")  # Disable GPU (recommended for headless mode)
    options.add_argument("--enable-automation")
    options.add_argument("--useAutomationExtension")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-extensions")
    options.add_argument("--dns-prefetch-disable")
    options.add_argument("--disable-dev-shm-usage")  # Fix shared memory issues in some environments
    options.add_argument("window-size=1920,1080")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(product_urls)
        sleep(3)
        current_page = 1  # Initialize current page
        while True:
            # Scroll the page to load all reviews
            for _ in range(20):
                driver.execute_script("window.scrollBy(0, 250)")
                sleep(1)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            # Extract reviews and ratings
            for product in soup.find_all('div', {"class": "css-1k41fl7"}):
                review_element = product.find('span', {"data-testid": "lblItemUlasan"})
                reviews.append(review_element.get_text() if review_element else 'None')

                rating_element = product.find('div', {"class": "rating"})
                ratings.append(rating_element.get('aria-label') if rating_element else 'None')
            # Break if the maximum number of pages (e.g., 2) is reached
            if current_page >= 3:
                break
            # Check if "Next" button exists and is enabled
            try:
                next_button_container = driver.find_element(By.CLASS_NAME, "css-1xqkwi8")
                next_button = next_button_container.find_element(
                    By.XPATH, './/button[contains(@class, "css-16uzo3v-unf-pagination-item") and @aria-label="Laman berikutnya"]'
                )
                is_disabled = next_button.get_attribute("disabled")  # Check if button is disabled
                if is_disabled:
                    break
                # Scroll to and click the next button
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
                driver.execute_script("arguments[0].click();", next_button)
                sleep(2)
                current_page += 1  # Increment current page
            except (NoSuchElementException, TimeoutException):
                break
    finally:
        driver.quit()
    print(f"Scraped {len(reviews)} reviews from {current_page} pages.")
    data = pd.DataFrame({'Review': reviews, 'Rating': ratings})
    return data