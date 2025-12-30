# app.py

# ------------------------------
# Imports
# ------------------------------
import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import InputLayer  # Needed for legacy model loading

# ------------------------------
# Load saved files
# ------------------------------
@st.cache_resource
def load_resources():
    """
    Load the LSTM model, tokenizer, and max_len from disk.
    """
    model = load_model(
        "lstm_model.h5",
        custom_objects={"InputLayer": InputLayer}  # Fix for older H5 model
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len

# Load resources
model, tokenizer, max_len = load_resources()

# ------------------------------
# Prediction function
# ------------------------------
def predict_next_word(text):
    """
    Given an input sentence, predict the next word using the LSTM model.
    """
    if not text.strip():
        return ""

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])[0]

    # Pad sequence
    sequence = pad_sequences([sequence], maxlen=max_len-1, padding='pre')

    # Make prediction
    preds = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(preds)

    # Map predicted index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return ""  # fallback

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Next Word Prediction", layout="centered")

st.title("🧠 Next Word Prediction (LSTM)")
st.write("Enter a sentence and the model will predict the **next word**.")

user_input = st.text_input("✍️ Enter text:", placeholder="Type a sentence here...")

if st.button("Predict Next Word"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        next_word = predict_next_word(user_input)
        if next_word:
            st.success(f"**Predicted Next Word:** {next_word}")
        else:
            st.error("Could not predict a next word for the given input.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Next Word Prediction using Streamlit")
