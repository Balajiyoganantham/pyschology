
from flask import Flask, render_template, request
import os
import json
import requests
import joblib
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from textblob import TextBlob
from textblob.sentiments import PatternAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import string
import re
from nltk.stem.wordnet import WordNetLemmatizer

app = Flask(__name__)
app.static_folder = 'static'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sent_bertphrase_embeddings = joblib.load(r"C:\mental-health-chatbot-main\model\questionembedding.dump")
sent_bertphrase_ans_embeddings = joblib.load(r"C:\mental-health-chatbot-main\model\ansembedding.dump")

stop_w = stopwords.words('english')
df = pd.read_csv(r"C:\mental-health-chatbot-main\model\20200325_counsel_chat.csv", encoding="utf-8")
lmtzr = WordNetLemmatizer()

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings
    return embeddings.detach().numpy()

def clean(column, df, stopwords=False):
    df[column] = df[column].apply(str)
    df[column] = df[column].str.lower().str.split()
    if stopwords:
        df[column] = df[column].apply(lambda x: [item for item in x if item not in stop_w])
    df[column] = df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
    df[column] = df[column].apply(lambda x: " ".join(x))

def retrieveAndPrintFAQAnswer(question_embedding, sentence_embeddings, FAQdf):
    max_sim = -1
    index_sim = -1
    valid_ans = []
    for index, faq_embedding in enumerate(sentence_embeddings):
        sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
        if sim >= max_sim:
            max_sim = sim
            index_sim = index
            valid_ans.append(index_sim)

    max_a_sim = -1
    answer = ""
    for ans in valid_ans:
        answer_text = FAQdf.iloc[ans, 8]
        answer_em = sent_bertphrase_ans_embeddings[ans]
        similarity = cosine_similarity(answer_em, question_embedding)[0][0]
        if similarity > max_a_sim:
            max_a_sim = similarity
            answer = answer_text

    if max_a_sim < 0.40:
        return "Could you please elaborate your situation more? I don't really understand."
    return answer

def clean_text(greetings):
    greetings = greetings.lower()
    greetings = ' '.join(word.strip(string.punctuation) for word in greetings.split())
    re.sub(r'\W+', '', greetings)
    greetings = lmtzr.lemmatize(greetings)
    return greetings

def predictor(userText):
    data = [userText]
    x_try = pd.DataFrame(data, columns=['text'])
    clean('text', x_try, stopwords=True)

    for index, row in x_try.iterrows():
        question = row['text']
        question_embedding = get_embeddings([question])
        return retrieveAndPrintFAQAnswer(question_embedding, sent_bertphrase_embeddings, df)

greetings = ['hi', 'hey', 'hello', 'heyy', 'hi', 'hey', 'good evening', 'good morning', 'good afternoon', 'good', 'fine', 'okay', 'great', 'could be better', 'not so great', 'very well thanks', 'fine and you', "i'm doing well", 'pleasure to meet you', 'hi whatsup']
happy_emotions = ['i feel good', 'life is good', 'life is great', "i've had a wonderful day", "i'm doing good"]
goodbyes = ['thank you', 'thank you', 'yes bye', 'bye', 'thanks and bye', 'ok thanks bye', 'goodbye', 'see ya later', 'alright thanks bye', "that's all bye", 'nice talking with you', 'i’ve gotta go', 'i’m off', 'good night', 'see ya', 'see ya later', 'catch ya later', 'adios', 'talk to you later', 'bye bye', 'all right then', 'thanks', 'thank you', 'thx', 'thx bye', 'thnks', 'thank u for ur help', 'many thanks', 'you saved my day', 'thanks a bunch', "i can't thank you enough", "`you're great", 'thanks a ton', 'grateful for your help', 'i owe you one', 'thanks a million', 'really appreciate your help', 'no', 'no goodbye']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    cleanText = clean_text(str(userText))

    #check sentiment 
    blob = TextBlob(userText, analyzer=PatternAnalyzer())
    polarity = blob.sentiment.polarity

    if cleanText in greetings:
        return "Hello! How may I help you today?"
    elif polarity > 0.7:
        return "That's great! Do you still have any questions for me?"
    elif cleanText in happy_emotions:
        return "That's great! Do you still have any questions for me?"
    elif cleanText in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"
    topic = predictor(userText)
    return topic

if __name__ == "__main__":
    app.run()
