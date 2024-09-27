# Spam Detection Project

## Problem Statement

In this project, we aim to determine whether the SMS and emails we receive are important messages or spam. With the rise of fraudulent messages, many people face harassment from spammers and scammers. To address this, I developed a machine learning-based solution that helps classify messages as spam or not spam.

## Tools and Technologies Used

This project is implemented in **Python**, using **Jupyter Notebook** for model development and testing. We experimented with several machine learning models, and the **Extra Trees (ET)** classifier provided the best results with an **accuracy of 0.975775** and a **precision of 1.0**.

### Libraries and Tools:
- **Python**: Primary language for building the solution.
- **Jupyter Notebook**: For model development and testing.
- **NLTK (Natural Language Toolkit)**: For natural language processing, including tokenization, stopword removal, and stemming.
- **scikit-learn (sklearn)**: For building and evaluating machine learning models.
- **TfidfVectorizer**: To convert text into numerical features for machine learning models.
- **Streamlit**: Used to deploy the project as a web application.
- **String**: Used to handle punctuation during text preprocessing.

## Preprocessing Steps

To make the data suitable for training machine learning models, the following preprocessing steps were applied:

1. **Text Normalization**: Convert all text to lowercase to ensure consistency.
2. **Tokenization**: Break down sentences into individual words using NLTK's `word_tokenize`.
3. **Stopword Removal**: Remove common English stopwords (like "the", "is", "in") that do not contribute to the classification task.
4. **Punctuation Removal**: Eliminate punctuation marks from the text.
5. **Stemming**: Reduce words to their base forms using `PorterStemmer`.
6. **TF-IDF Vectorization**: Use the `TfidfVectorizer` to convert the preprocessed text into numerical vectors suitable for machine learning models.

## Machine Learning Models Performance

Various machine learning models were tested to determine the best one for spam detection. Below is the performance comparison table:

| **Algorithm**                  | **Accuracy** | **Precision** |
|---------------------------------|--------------|---------------|
| **ET (Extra Trees Classifier)** | 0.975775     | 1.000000      |
| **RF (Random Forest Classifier)** | 0.971899   | 1.000000      |
| **NB (Naive Bayes Classifier)**  | 0.964147   | 1.000000      |
| **KNN (K-Nearest Neighbors)**    | 0.915698   | 1.000000      |
| **SVC (Support Vector Classifier)** | 0.968023 | 0.988095      |
| **GB (Gradient Boosting)**       | 0.957364   | 0.986301      |
| **LR (Logistic Regression)**     | 0.949612   | 0.984615      |
| **XGB (XGBoost Classifier)**     | 0.967054   | 0.955056      |
| **AdaBoost**                     | 0.955426   | 0.925926      |
| **DT (Decision Tree Classifier)** | 0.963178  | 0.873786      |
| **Bagging**                      | 0.957364   | 0.873684      |

The **Extra Trees (ET)** classifier performed the best, with a high accuracy and perfect precision, making it the final choice for deployment.

## Project Workflow

1. **Data Preprocessing**: Perform text cleaning, stopword removal, and stemming on the input messages.
2. **TF-IDF Vectorization**: Convert preprocessed text into numerical form.
3. **Model Training**: Train machine learning models on the preprocessed data.
4. **Model Evaluation**: Evaluate the performance of the models and choose the best one (ET).
5. **Spam Detection**: Classify new SMS or email messages as spam or not spam using the trained model.

## How to Use This Project

1. **Clone the Repository**: 
   ```bash
   
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
    
