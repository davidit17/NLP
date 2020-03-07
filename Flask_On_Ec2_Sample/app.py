# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 19:42:41 2019

@author: python
"""
deftext = """
The Mageean Cup is awarded annually to the winners of the Ulster 
Colleges' Senior Hurling Championship. This iconic trophy was 
presented to Ulster Colleges in 1963 by the staff and students
 of Dromantine College, Newry in memory of Most Reverend Daniel Mageean
 , Bishop of Down and Connor (1929 – 1962).[1] St Mary's CBGS Belfast
 are the cup specialists with 28 titles. The cup has spent most of its
 time in the Belfast school with only St Patricks Maghera challenging 
 their dominance

"""

from flask import Flask,render_template, request, jsonify
from gensim.summarization import summarize
import spacy

nlp = spacy.load('en')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/',methods=['POST'])
def result():
    text = request.form['message']
    if not text:
        text = deftext
    docx = nlp(text)
    entities = [(entity.text,entity.label_) for entity in docx.ents]
    entities = ('||| '.join('%s - %s ' % pair for pair in entities))
    try:
        summary = summarize(text)
    except:
        summary='Text is too short for summarization'
    
    return render_template('return.html',entities=entities,summary=summary,text=text)

    
if __name__ == "__main__":
    # Local host:
    app.run(host='127.0.0.1',port=5000)
    # Aws:
    #app.run(host='0.0.0.0',port=80)