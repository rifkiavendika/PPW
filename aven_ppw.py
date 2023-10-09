import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim import corpora
from gensim.models import LdaModel

# Fungsi untuk crawling data
def crawl_data(total_pages):
    all_data = []
    for page in range(1, total_pages + 1):
        url = f"https://pta.trunojoyo.ac.id/c_search/byprod/10/{page}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            data_items = soup.find_all('li', attrs={"data-id": lambda x: x and "id-" in x})
            for item in data_items:
                judul_berita = item.find('a', class_='title').text.strip()
                penulis = item.find_all('div', style="padding:2px 2px 2px 2px;")
                penulis_info = [p.span.text.strip() for p in penulis]
                selengkapnya_link = item.find('a', class_='gray button')['href']
                selengkapnya_response = requests.get(selengkapnya_link)
                if selengkapnya_response.status_code == 200:
                    selengkapnya_soup = BeautifulSoup(selengkapnya_response.text, 'html.parser')
                    abstrak = selengkapnya_soup.find('p', align='justify').text.strip()
                else:
                    abstrak = "Tidak dapat mengambil abstrak"
                all_data.append([judul_berita] + penulis_info + [abstrak])
    return pd.DataFrame(all_data, columns=["Judul", "Penulis", "Pembimbing 1", "Pembimbing 2", "Abstrak"])


# Fungsi untuk melakukan Case Folding
def case_folding(df):
    if 'Abstrak' in df.columns:
        data = df['Abstrak']
        df['CaseFolding'] = data.str.lower()
        return df
    else:
        st.write("Kolom 'Abstrak' tidak ditemukan dalam DataFrame.")
        return None

# Fungsi untuk melakukan Character Cleansing
def character_cleansing(df):
    if 'CaseFolding' in df.columns:
        df['CleansedText'] = df['CaseFolding'].str.replace('[^\w\s]', '', regex=True)
        return df
    else:
        st.write("Kolom 'CaseFolding' tidak ditemukan dalam DataFrame.")
        return None
    
# Fungsi untuk melakukan Tokenisasi
def tokenize_text(df):
    if 'CleansedText' in df.columns:
        df['Tokens'] = df['CleansedText'].apply(word_tokenize)
        return df
    else:
        st.write("Kolom 'CleansedText' tidak ditemukan dalam DataFrame.")
        return None
    

# Fungsi untuk melakukan Stopword Removal
def remove_stopwords(df):
    if 'Tokens' in df.columns:
        stop_words = set(stopwords.words('indonesian')) # Ganti dengan bahasa sesuai kebutuhan
        df['StopWord'] = df['Tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])
        return df
    else:
        st.write("Kolom 'Tokens' tidak ditemukan dalam DataFrame.")
        return None
    

# Fungsi untuk melakukan Stemming
def apply_stemming(df):
    if 'StopWord' in df.columns:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        df['Stemmed'] = df['StopWord'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])
        return df
    else:
        st.write("Kolom 'StopWord' tidak ditemukan dalam DataFrame.")
        return None

# Fungsi untuk Word2Vec Skip-gram
def word2vec_skipgram(df):
    if 'Stemmed' in df.columns:
        # Membuat model Word2Vec Skip-gram
        model = Word2Vec(sentences=df['Stemmed'], vector_size=100, window=5, sg=1, min_count=1)

        # Training model Word2Vec
        model.train(df['Stemmed'], total_examples=len(df['Stemmed']), epochs=10)

        return model
    else:
        st.write("Kolom 'Stemmed' tidak ditemukan dalam DataFrame.")
        return None

# Fungsi untuk Pemodelan Topik dengan LDA
def lda_topic_modeling(df, num_topics):

    global corpus  # Gunakan corpus global

    if 'Stemmed' in df.columns:
        # Membuat kamus kata
        dictionary = corpora.Dictionary(df['Stemmed'])
        # Membentuk korpus teks
        corpus = [dictionary.doc2bow(text) for text in df['Stemmed']]
        # Membuat model LDA
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

        return lda_model
    else:
        st.write("Kolom 'Stemmed' tidak ditemukan dalam DataFrame.")
        return None
    
st.title("CRAWLING PTA TEKNIK INFORMATIK TRUNOJOYO MADURA")
st.write("RIFKI AVENDIKA | 170411100030")

# Membuat tab
tabs = st.tabs(["Crawling", "Preprocessing", "Term Frequency", "Word2Vec", "Pemodelan Topik"])
total_pages = 0
df_crawled = None
df_stemmed = None

# Tab Pertama: Crawling
with tabs[0]:
    total_pages = st.number_input("Masukkan jumlah halaman yang ingin di-crawl", min_value=1, value=10)

    if st.button("Mulai Crawling"):
        with st.spinner("Sedang melakukan crawling..."):
            df_crawled = crawl_data(total_pages)
            st.success("Crawling selesai!")

# Tab Kedua: Preprocessing
with tabs[1]:
    if df_crawled is not None:
        st.subheader("Case Folding")
        df_preprocessed = case_folding(df_crawled)
        st.dataframe(df_preprocessed[['Abstrak', 'CaseFolding']])

        st.subheader("Character Cleansing")
        df_cleansed = character_cleansing(df_preprocessed)
        st.dataframe(df_cleansed[['CaseFolding', 'CleansedText']])

        st.subheader("Tokenisasi")
        df_tokenized = tokenize_text(df_cleansed)
        st.dataframe(df_tokenized[['CleansedText', 'Tokens']])

        st.subheader("Stopword Removal")
        df_no_stopwords = remove_stopwords(df_tokenized)
        st.dataframe(df_no_stopwords[['Tokens', 'StopWord']])

        st.subheader("Stemming")
        with st.spinner("Sedang melakukan Stemming..."):
            df_stemmed = apply_stemming(df_no_stopwords)
            st.dataframe(df_stemmed[['StopWord', 'Stemmed']])

# Tab Ketiga: Term Frequency
with tabs[2]:
    if df_stemmed is not None:
        # Menggabungkan daftar kata-kata menjadi satu string
        df_stemmed["StemmedString"] = df_stemmed["Stemmed"].apply(lambda x: ' '.join(x))

        # Inisialisasi CountVectorizer
        vectorizer = CountVectorizer()

        # Menghitung TF dan membentuk VSM
        tf_matrix = vectorizer.fit_transform(df_stemmed["StemmedString"])

        # Membentuk DataFrame dari matriks TF
        tf_df = pd.DataFrame(tf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Menampilkan hasil VSM dalam term frequency
        st.subheader("Hasil VSM dalam Term Frequency")
        st.dataframe(tf_df)

# Tab Keempat: Word2Vec Skip-gram
with tabs[3]:
    if df_stemmed is not None:
        st.subheader("Word2Vec Skip-gram")
        model = word2vec_skipgram(df_stemmed)
        if model is not None:
            # Mendapatkan vektor untuk setiap dokumen
            vectors = [model.wv[words].mean(axis=0) for words in df_stemmed['Stemmed']]

            # Membentuk DataFrame dari vektor dokumen
            df_vectors = pd.DataFrame(vectors)

            # Menampilkan vektor per dokumen
            st.dataframe(df_vectors)

# Tab Kelima: Pemodelan Topik dengan LDA

with tabs[4]:
    if df_stemmed is not None :
        lda_model = lda_topic_modeling(df_stemmed, 10 )
        # Dapatkan distribusi topik untuk setiap dokumen
        topic_distribution = lda_model.get_document_topics(corpus)

        # Membuat DataFrame untuk distribusi topik
        df_topic_distribution = pd.DataFrame(topic_distribution)

        # Menampilkan proporsi topik dalam dokumen
        st.subheader("Proporsi 10 Topik dalam Dokumen")
        st.dataframe(df_topic_distribution)

        # Dapatkan distribusi kata untuk setiap topik
        topic_terms = lda_model.show_topics(formatted=False, num_words= 10 )

        # Membuat DataFrame untuk distribusi kata dalam topik
        df_topic_terms = pd.DataFrame([(topic[0], [term[0] for term in topic[1]]) for topic in topic_terms], 
                                    columns=["Topik", "Kata-kata"])

        # Menampilkan proporsi kata dalam topik
        st.subheader("Proporsi Kata dalam 10 Topik")
        st.dataframe(df_topic_terms) 