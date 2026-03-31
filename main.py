import streamlit as st 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="Spam Detector", page_icon="🚫")

# --- 1. LOAD & TRAIN DATA ---

@st.cache_resource # This keeps the model in memory so it doesn't re-train on every click
def train_model():
    data = {
        "label": ["spam", "ham", "spam", "ham", "spam", "ham"],
        "text": [
            "Win money now!!!", "Hey how are you?", 
            "Free iPhone!!!", "Let's meet tomorrow",
            "Click here for cash", "Dinner at 7?"
        ]
    }
    df = pd.DataFrame(data)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])
    model = MultinomialNB()
    model.fit(X, df["label"])
    
    return vectorizer, model

vectorizer, model = train_model()

# --- 2. STREAMLIT UI ---

st.title("🚫 Spam or Ham?")
st.write("Enter a message below to see if our model thinks it's spam.")

# USER INPUT
user_input = st.text_area("Message to analyze:", placeholder="e.g., Free money now!!!")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Transform and Predict
        test_vec = vectorizer.transform([user_input])
        prediction = model.predict(test_vec)[0]
        
        # Calculate Probability (Optional but cool)
        proba = model.predict_proba(test_vec)
        
        # Show Result
        if prediction == "spam":
            st.error(f"🚨 This looks like **SPAM**!")
        else:
            st.success(f"✅ This looks like a legitimate message (**HAM**).")
            
        # UI Polish: Show the confidence level
        st.info(f"Confidence: {max(proba[0]) * 100:.2f}%")

# Sidebar for project info
st.sidebar.header("About the Project")
st.sidebar.info("This app uses a Multinomial Naive Bayes classifier to detect spam patterns in text.")

