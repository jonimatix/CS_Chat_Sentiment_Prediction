import csv
import urllib.request
from scipy.special import softmax
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TFXLMRobertaModel, TextClassificationPipeline
import torch
import torch.nn.functional as F

import ast
import pandas as pd
import numpy as np
import os
import gc
import time
import io
import sys
import re
from tqdm import tqdm

import pypyodbc
import configparser
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sqlalchemy import create_engine
from configparser import ConfigParser
import pyodbc

import importlib
from helperFunctions import *
import helperFunctions
from utils import *
import utils
importlib.reload(helperFunctions)
importlib.reload(utils)

warnings.filterwarnings('ignore')

df = load_obj(output_path, 'json_expanded')

'''
cardiffnlp/twitter-xlm-roberta-base-sentiment # This is a XLM-roBERTa-base model trained on ~198M tweets and finetuned for sentiment analysis. The sentiment fine-tuning was done on 8 languages (Ar, En, Fr, De, Hi, It, Sp, Pt) but it can be used for more languages (see paper for details).
nlptown/bert-base-multilingual-uncased-sentiment # This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).
cardiffnlp/twitter-roberta-base-sentiment # This is a roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis with the TweetEval benchmark.
distilbert-base-multilingual-cased
distilbert-base-uncased for english text
roberta-large-mnli
jplu/tf-xlm-roberta-large
jplu/tf-xlm-roberta-base
'''

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained('./Models/model')

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

text = ["Porca miseria!",
        "I would love to kill you one day",
        "Jävla tyskar!",
        "You are the best :)",
        "Jävla horkärring, gå fan och knulla dig där bak i arslet så jävla hårt. Vilken liten jävla kukfitta du är."]

# tokenizer.tokenize(text)
result = classifier(text)

for r in result:
    print(f"Sentiment: {r['label']}, Score:{np.round(r['score'], 2)}")

###
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.",
        "I HATE YOU", "Jävla horkärring, gå fan och knulla dig där bak i arslet så jävla hårt. Vilken liten jävla kukfitta du är."],
    padding=True, truncation=True,
    max_length=512,
    return_tensors="pt")

pt_outputs = model(**pt_batch)
pt_predictions = F.softmax(pt_outputs[0], dim=-1)
print(pt_predictions)


def get_model(model):
    """Loads model from Hugginface model hub"""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model, use_cdn=True)
        model.save_pretrained('./Models/model')
    except Exception as e:
        raise(e)


def get_tokenizer(tokenizer):
    """Loads tokenizer from Hugginface model hub"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenizer.save_pretrained('./Models/model')
    except Exception as e:
        raise(e)


def get_emotion_by_label(self, label: str):
    if label == "LABEL_0":
        return "negative"
    elif label == "LABEL_1":
        return "neutral"
    elif label == "LABEL_2":
        return "positive"
    else:
        print("SOMETHING IS WRONG")
        return ""


def get_emotion(self, phrase):
    results = self.classifier(phrase)
    res = dict()
    for result in results:
        for emotion in result:
            res.update({self.get_emotion_by_label(
                emotion['label']): emotion['score']})
    return res


get_model('deepset/xlm-roberta-large-squad2')
get_tokenizer('deepset/xlm-roberta-large-squad2')

df = (
    df
    .assign(sentiment=lambda x: x['Q72'].apply(lambda s: classifier(s)))
    .assign(
        label=lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
        score=lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
    )
)

##################################################################
# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment


class SentimentDetection(object):
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        self.model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True,
                                                     task="sentiment-analysis")

    def get_emotion_by_label(self, label: str):
        if label == "LABEL_0":
            return "negative"
        elif label == "LABEL_1":
            return "neutral"
        elif label == "LABEL_2":
            return "positive"
        else:
            print("SOMETHING IS WRONG")
            return ""

    def get_emotion(self, phrase):
        results = self.classifier(phrase)
        res = dict()
        for result in results:
            for emotion in result:
                res.update({self.get_emotion_by_label(
                    emotion['label']): emotion['score']})
        return res


em = SentimentDetection(model_name="")
em.get_emotion("son of a bitch")

###########################################################
###########################################################

''' 
# Preprocess text
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)

text = "Good night 😊"
text = preprocess(text)
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores

# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# text = "Good night 😊"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)

# Print labels and scores
ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = config.id2label[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
'''

###########################################################
###########################################################

'''
task = 'emotion'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]
labels

# PT
#model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#text = "Good night 😊"
#text = preprocess(text)
#encoded_input = tokenizer(text, return_tensors='pt')
#output = model(**encoded_input)
#scores = output[0][0].detach().numpy()
#scores = softmax(scores)

# # TF
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

text = "Good night 😊"
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
scores = output[0][0].numpy()
scores = softmax(scores)

ranking = np.argsort(scores)
ranking = ranking[::-1]
for i in range(scores.shape[0]):
    l = labels[ranking[i]]
    s = scores[ranking[i]]
    print(f"{i+1}) {l} {np.round(float(s), 4)}")
'''
###########################################################
###########################################################

'''
# look at for db inserts https://github.com/huggingface/transformers/issues/9629

class GenerateDbThread(object):
    def __init__(self, text: str, created_at: datetime.datetime, get_emotion_function, cursor, table_name):
        self.table_name = table_name

        self.text = text
        self.created_at = created_at
        emotions = get_emotion_function(self.text)

        self.pos = emotions['positive']
        self.neg = emotions['negative']
        self.neu = emotions['neutral']

        self.cursor = cursor

    def execute(self):
        query = f"INSERT INTO {self.table_name}(date, positive, negative, neutral, tweet) " \
                f"VALUES (datetime('{str(self.created_at)}'),{self.pos},{self.neg},{self.neu}, '{self.text}')"
        self.cursor.execute(query)
        self.cursor.commit()


def get_all_data_files_path(data_dir: str):
    return [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

def init_db(db_name, table_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        uid INTEGER PRIMARY KEY AUTOINCREMENT,
        date DATETIME NOT NULL,
        positive REAL NOT NULL,
        negative REAL NOT NULL,
        neutral REAL NOT NULL,
        text TEXT NOT NULL
    )""")
    cursor.execute(
        f"CREATE INDEX IF NOT EXISTS ix_tweets_index ON {table_name}(uid)")
    cursor.close()
'''
