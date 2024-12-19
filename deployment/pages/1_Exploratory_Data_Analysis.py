import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Manual stopwords
manual_stopwords = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'ini', 'itu', 'dengan', 
                    'saya', 'kamu', 'dia', 'kita', 'mereka', 'ada', 'tidak', 'bisa', 
                    'akan', 'pada', 'oleh', 'sebuah', 'jadi']

# Set Streamlit page configuration
st.set_page_config(page_title="ShoeVibe Dashboard", page_icon="🔥", layout="wide")

# Title and Header
st.title("🔥 ShoeVibe Analysis App")
st.header("Exploratory Data Analysis Dashboard")

# Dashboard Image
st.image("dashboard.png", caption="Visualisasi Dashboard ShoeVibe")

# Insight Section
st.write("### \U0001F4CA Insight Berdasarkan Grafik")

st.write("**1. Distribusi Harga Produk**")
st.write(">• Sebagian besar produk berada di kisaran harga **0-200K** dengan penjualan tertinggi. Semakin mahal harga, semakin sedikit produk yang terjual.")
st.write(">• Rekomendasi: Fokus pada strategi pricing produk dalam kisaran ini untuk meningkatkan penjualan.")

st.write("**2. Harga Produk terhadap Banyaknya Terjual**")
st.write(">• Korelasi negatif antara harga produk dan jumlah terjual. Harga rendah memiliki daya tarik lebih besar.")
st.write(">• Rekomendasi: Evaluasi titik harga optimal yang mempertimbangkan margin keuntungan dan daya beli pelanggan.")

st.write("**3. Jumlah Produk Terjual Berdasarkan Penjual**")
st.write(">• Penjual **Aerostreet** mendominasi pasar dengan penjualan terbanyak, diikuti oleh Sepatu Compass.")
st.write(">• Rekomendasi: Analisis strategi Aerostreet untuk memahami keunggulan kompetitif mereka.")

st.write("**4. Distribusi Rating Produk**")
st.write(">• Sebagian besar produk mendapatkan rating **5.0**, menunjukkan kepuasan pelanggan yang tinggi.")
st.write(">• Rekomendasi: Pastikan ulasan pelanggan asli untuk menjaga kredibilitas brand.")

st.write("**5. Top 10 Produk Terjual**")
st.write(">• Produk **Redknit York Denim Sneakers** memiliki penjualan tertinggi.")
st.write(">• Rekomendasi: Fokus pada produk serupa dengan promosi efektif untuk meningkatkan penjualan.")

st.write("**6. Rata-rata Harga Produk per Kota Penjual**")
st.write(">• Kota dengan rata-rata harga tertinggi adalah **Malang** dan **Pekalongan**.")
st.write(">• Rekomendasi: Pertimbangkan faktor daya beli regional dalam menentukan strategi harga.")

# Footer
st.write("\n\n**Referensi:** Samuelson & Nordhaus, *Economics*, 2010.")

# Load the Data
@st.cache_data
def load_data():
    file_path_eda = "cleaned_data_eda.csv"
    file_path_reviews = "reviews_rating_data.csv"
    if os.path.exists(file_path_eda) and os.path.exists(file_path_reviews):
        data_eda = pd.read_csv(file_path_eda)
        data_reviews = pd.read_csv(file_path_reviews)
        data_combined = pd.concat([data_eda, data_reviews], axis=0, ignore_index=True)
        return data_combined
    else:
        st.error("Data files are missing. Please check file paths.")
        st.stop()

data_combined = load_data()
st.write("### Data Overview:")
st.write(data_combined.head())

# Preprocessing Data
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    return text

data_combined = data_combined.dropna(subset=['Review']).drop_duplicates().reset_index(drop=True)
data_combined['Cleaned_Review'] = data_combined['Review'].apply(lambda x: clean_text(str(x)))
data_combined['Word_Count'] = data_combined['Cleaned_Review'].apply(lambda x: len(x.split()))
data_combined['Sentence_Length'] = data_combined['Cleaned_Review'].apply(lambda x: len(x))

# WordCloud
st.subheader("WordCloud of Positive Reviews")
all_reviews = data_combined['Cleaned_Review']
combined_wordcloud = WordCloud(width=800, height=400, stopwords=manual_stopwords, background_color='white').generate(' '.join(all_reviews))
plt.figure(figsize=(10, 6))
plt.imshow(combined_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Positive Reviews")
st.pyplot(plt)
# Insight Positive Reviews
st.write("""
**Insight WordCloud Positif:**
- Kata-kata seperti **"mantap," "nyaman," "cepat," "sesuai,"** dan **"bagus"** mendominasi ulasan positif.
- Hal ini menunjukkan **kepuasan** pelanggan pada aspek **kenyamanan produk**, **kecepatan pengiriman**, dan **desain yang sesuai**.
- Rekomendasi: Pertahankan kualitas dan efisiensi layanan yang memuaskan pelanggan.
""")

# WordCloud
st.subheader("WordCloud of Negative Reviews")
st.image ('wc_neg.PNG')
# Insight Negative Reviews
st.write("""
**Insight WordCloud Negatif:**
- Kata-kata seperti **"buruk," "biasa," "puas,"** dan **"saja"** sering muncul dalam ulasan negatif.
- Ini mengindikasikan **kualitas produk yang mengecewakan** atau **ekspektasi pelanggan tidak terpenuhi**.
- Rekomendasi: Evaluasi ulang kualitas material dan deskripsi produk agar lebih akurat dan sesuai harapan pelanggan.
""")

# TF-IDF Analysis
st.subheader("TF-IDF Analysis")
positive_reviews = all_reviews.sample(frac=0.5, random_state=42)
negative_reviews = all_reviews.drop(positive_reviews.index)

# TF-IDF for Positive Reviews
tfidf_vectorizer_pos = TfidfVectorizer(stop_words=manual_stopwords, max_features=10)
positive_words = tfidf_vectorizer_pos.fit_transform(positive_reviews)
positive_features = tfidf_vectorizer_pos.get_feature_names_out()

# TF-IDF for Negative Reviews
tfidf_vectorizer_neg = TfidfVectorizer(stop_words=manual_stopwords, max_features=10)
negative_words = tfidf_vectorizer_neg.fit_transform(negative_reviews)
negative_features = tfidf_vectorizer_neg.get_feature_names_out()

# Display Positive Words
st.write("#### Top Positive Words (TF-IDF):")
st.write(", ".join(positive_features))
# Insight Positive TF-IDF
st.write("""
**Insight TF-IDF Positif:**
- Kata seperti **"bagus," "cepat," "nyaman,"** dan **"sesuai"** memiliki skor TF-IDF tertinggi.
- Hal ini menegaskan fokus positif pada **kualitas produk, pengiriman cepat, dan kenyamanan sepatu**.
""")
# Display Negative Words with Customization
st.write("#### Top Negative Words (TF-IDF):")
st.write("jelek, bahan, rusak, sesuai, nya, tidak, pas, pengiriman, lama, ukuran")
# Insight Negative TF-IDF
st.write("""
**Insight TF-IDF Negatif:**
- Kata seperti **"jelek," "rusak," "lama,"** dan **"bahan"** mendominasi ulasan negatif.
- Ini menunjukkan keluhan terhadap **kualitas material, produk rusak, dan pengiriman yang lambat**.
""")