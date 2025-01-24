import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
ps = PorterStemmer()
st.title('Spam Ham Classifier')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum() and i not in stopwords.words('english') and i not in string.punctuation and ps.stem(i) not in y:
            y.append(ps.stem(i))
    return " ".join(y)


input_sms = st.text_area('Enter the message')
if st.button('Predict'):
    print(">>>>>", input_sms)
    # 1. Preprocess the input message
    transformed_text = transform_text(input_sms)
    print(">>>>>", transformed_text)
    # 2. Vectorize the input message
    vector_input = tfidf.transform([transformed_text])
    print(">>>>>", vector_input)
    # 3. Predict the input message
    result = model.predict(vector_input)[0]
    print(">>>>>", result)
    # 4. Display the output
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
