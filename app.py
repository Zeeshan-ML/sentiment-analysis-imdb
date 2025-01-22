import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb # type: ignore
from tensorflow.keras.preprocessing import sequence # type: ignore
from tensorflow.keras.models import load_model # type: ignore

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model=load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i -3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+ 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## Prediction Function

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

import streamlit as st # type: ignore
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    prediction = model.predict(preprocess_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Predcition Score:{prediction[0][0]}')
else:
    st.write('Please enter a movie review.')
