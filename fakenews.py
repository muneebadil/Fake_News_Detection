import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^"]\w\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def load_and_prepare_data():
    df_real = pd.read_csv("real_news.csv")
    df_fake = pd.read_csv("fake_news.csv")

    df_real['label'] = 1  
    df_fake['label'] = 0  

    df = pd.concat([df_real, df_fake], axis=0).reset_index(drop=True)
    if 'text' not in df.columns:
        raise ValueError("Expected a 'text' column in both CSVs.")

    df['clean_text'] = df['text'].apply(clean_text)
    return df

@st.cache_resource
def train_model():
    df = load_and_prepare_data()
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, acc


st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
        }
        .stApp {
            background-color: #1e1e1e;
            color: white;
        }
        .stTextArea textarea {
            background-color: #2c2c2c;
            color: white;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
        }
        .stAlert {
            background-color: #333;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown("""
    <h1 style='text-align: center; color: #00FFAA;'>🧠 Fake News Detector</h1>
    <p style='text-align: center;'>Built with Machine Learning - Enter news content and detect if it's Real or Fake.</p>
""", unsafe_allow_html=True)

model, vectorizer, accuracy = train_model()
st.markdown(f"<h4 style='color: #00FFAA;'>✅ Model Accuracy: {accuracy:.2%}</h4>", unsafe_allow_html=True)

user_input = st.text_area("✍️ Paste News Article Text Here", height=200)

if st.button("🚀 Detect Now"):
    if user_input.strip() == "":
        st.warning("Please enter some news content!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("✅ This appears to be a REAL news article.")
        else:
            st.error("❌ This appears to be a FAKE news article.")