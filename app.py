

import streamlit as st
import pickle
import string
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')   # Re-downloads the correct tokenizer


from nltk.tokenize import word_tokenize

text = "Your sample text."
tokens = word_tokenize(text) 

ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)

  y = []
  for i in text:
      if i.isalnum():
          y.append(i)

  text = y.copy()

  y.clear()
  for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
          y.append(i)

  text = y.copy()

  y.clear()
  ps = PorterStemmer()
  for i in text:

      y.append(ps.stem(i))

  text = y.copy()
  return ' '.join(text)


tfid = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input('Enter the message')

if st.button('Predict'):

    # 1.Preprocess
    transformed_sms = transform_text(input_sms)
    
    # 2. Vectorize
    
    vector_input = tfid.transform([transformed_sms])
    
    # 3. Predict
    
    result = model.predict(vector_input)[0]
    
    # 4. Display 
    
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')    
