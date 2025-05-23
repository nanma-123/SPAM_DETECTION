import streamlit as st
import joblib
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Downloads
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

# Preprocessing (must match training)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized)

# Now load the model
model = joblib.load("spam_classifier.pkl")

# Streamlit UI
st.title("📧 Spam Message Classifier")
user_input = st.text_area("Enter a message to classify:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        processed_input = preprocess(user_input)
        prediction = model.predict([processed_input])[0]
        if prediction == 1:
            st.success("✅ This message is **Not Spam** (Ham)")
        else:
            st.error("🚫 This message is **Spam**")
