# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 23:36:43 2024

@author: David
"""


# TODO 
# load text file of questions from user
# finetune on texts
# generate question based on input words

import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# load dataset and parse all questions
with open('train.json',encoding="utf8") as f:
    data = json.load(f)
    df = pd.DataFrame(data['data'])
    
df_normalized = pd.DataFrame.from_dict(pd.json_normalize(df['paragraphs']), orient='columns')
one_series = pd.concat([df_normalized[0],df_normalized[1]],ignore_index=True).dropna()

all_questions = []
for item in one_series:
    if 'qas' in item:
        questions = [qa['question'] for qa in item['qas']]
        all_questions.extend(questions)
    

# tfidf cosine baseline
def find_best_match(user_input, dataset, top_n=1):
    
    all_text = [user_input] + dataset
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()
    top_n = similarities.argsort()[-top_n:][::-1]

    return [dataset[i] for i in top_n]



def main():
    
    st.title("Find n similiar question")
    
    st.sidebar.write("")
    st.sidebar.write("")
    user_input = st.sidebar.text_input("enter question", "כמה זה עולה")
    n = st.sidebar.number_input("how many questions to show", 1, 10, 3)
    
    if st.sidebar.button(f"find {n} closest questions"):
        st.subheader("Result")
        
        # algo
        best_match = find_best_match(user_input, all_questions,n)
        #\algo
        
        st.write(f"original question: {user_input}")
        st.write(f"{n} closest questions :")
        for i, q in enumerate(best_match, start=1):
            st.write(f"{i}. {q}")
            

        rating = st.slider("Rate the result", 0, 5, 0)
        st.write(f"You selected: {rating}")
           
if __name__ == "__main__":
    main()