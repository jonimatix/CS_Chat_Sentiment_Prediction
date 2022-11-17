from scipy.special import softmax
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
from transformers import pipeline, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TFXLMRobertaModel, TextClassificationPipeline
import torch
import torch.nn.functional as F
from cleantext import replace_phone_numbers, replace_emails, replace_urls

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

# import pypyodbc
# import configparser
# import pyodbc

from sqlalchemy import create_engine
from configparser import ConfigParser

import importlib
from helperFunctions import *
import helperFunctions
from utils import *
import utils
importlib.reload(helperFunctions)
importlib.reload(utils)

import warnings
warnings.filterwarnings('ignore')

# path = 'C:/Users/Jonathanm/Desktop/LiveChat Chats EDA/'

# print(os.getcwd())
# os.chdir(path)

input_path = '../Input/'
output_path = '../Output/'
model_path = '../Models/'
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
model_full_path = model_path + model_name

# Get data set
df = load_obj(output_path, 'json_expanded')
# df.shape
# df.head(5)

'''
imp: pip install sentencepiece

cardiffnlp/twitter-xlm-roberta-base-sentiment # This is a XLM-roBERTa-base model trained on ~198M tweets and finetuned for sentiment analysis. The sentiment fine-tuning was done on 8 languages (Ar, En, Fr, De, Hi, It, Sp, Pt) but it can be used for more languages (see paper for details).
nlptown/bert-base-multilingual-uncased-sentiment # This a bert-base-multilingual-uncased model finetuned for sentiment analysis on product reviews in six languages: English, Dutch, German, French, Spanish and Italian. It predicts the sentiment of the review as a number of stars (between 1 and 5).
cardiffnlp/twitter-roberta-base-sentiment # This is a roBERTa-base model trained on ~58M tweets and finetuned for sentiment analysis with the TweetEval benchmark.
distilbert-base-multilingual-cased
distilbert-base-uncased for english text
roberta-large-mnli
jplu/tf-xlm-roberta-large
jplu/tf-xlm-roberta-base
'''


def get_model(model_name):
    """Loads model from Hugginface model hub or from disk if the directory already exists"""

    try:
        # Check if directory exists first
        if os.path.exists(model_full_path):
            print('Path already exists')

            model = AutoModelForSequenceClassification.from_pretrained(model_full_path)

            tokenizer = AutoTokenizer.from_pretrained(model_full_path)

            print('Model read from disk')
            logging.info('Model read from disk')

        else:
            print('Path does not exists, so download model from Hub')

            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            tokenizer = AutoTokenizer.from_pretrained(model)
            config = AutoConfig.from_pretrained(model)

            tokenizer.save_pretrained(model_full_path)
            config.save_pretrained(model_full_path)
            model.save_pretrained(model_full_path)

            # Reload tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_full_path)
            print('Model downloaded from Hub')

            logging.info('Model downloaded from Hub')

    except Exception as e:
        raise(e)

    return model, tokenizer


# Get model
model, tokenizer = get_model(model_name)

def prepare_tokenizer_predict(texts):
    ''' Function to tokenise text (parameter) and get prediction probabilities '''
    pt_batch = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    pt_outputs = model(**pt_batch)
    pt_predictions = F.softmax(pt_outputs[0], dim=-1).detach().numpy()
    # print(pt_predictions)
    return pt_predictions


def get_sentiment_by_label(score: int):
    if score == 0:
        return "Negative"
    elif score == 1:
        return "Neutral"
    elif score == 2:
        return "Positive"
    else:
        print("SOMETHING IS WRONG")
        return ""


ISTEST = False

if ISTEST:
    # Initiate classifier
    classifier = pipeline(task='sentiment-analysis', model=model,
                          tokenizer=tokenizer, return_all_scores=False)

    text = ["Porca miseria!",
            "Kurwa",
            "I would love to kill you one day",
            "Jävla tyskar!",
            "You are the best :)",
            "Jävla horkärring, gå fan och knulla dig där bak i arslet så jävla hårt. Vilken liten jävla kukfitta du är."]

    # tokenizer.tokenize(text)
    result = classifier(text)

    for i, r in enumerate(result):
        print(f"Phrase: {text[i]}: Sentiment: {r['label']}, Score:{np.round(r['score'], 2)} \n")
        
else:
    logging.info(f'Calling prepare_tokenizer_predict')

    start = time.time()

    res = []
    chunks = np.ceil(len(df) / 50)  # batches of size 50

    logging.info(f'Processing {chunks} chunks')
    print(f'Processing {chunks} chunks \n')

    for i, chunk in tqdm(enumerate(np.array_split(df, chunks)), total=chunks):
        res.append(prepare_tokenizer_predict(chunk['text'].to_list()))
        gc.collect()

    print('\nTime taken:', (time.time() - start) / 60, "mins")

    logging.info(f'Finished prepare_tokenizer_predict, duration {(time.time() - start) / 60} mins')

    save_obj(output_path, res, 'raw_sentiment_preds')

    res_extracted = []
    for r in res:
        for b in r:
            res_extracted.append([b[0], b[1], b[2], b.argmax()])
            # print(b[0], b[1], b[2], " Final:", b.argmax())

    logging.info(f'Extracted sentiments from array')

    df_sentiments = pd.DataFrame(res_extracted,
                                 columns=['Negative', 'Neutral', 'Positive', 'Score'])

    print(df_sentiments['Score'].value_counts())

    # df_sample = df.head(df_sentiments.shape[0])
    df_sample_sents = pd.concat([df, df_sentiments], axis=1)

    df_sample_sents['Sentiment'] = df_sample_sents['Score'].apply(
        lambda x: get_sentiment_by_label(x))

    # Remove Score
    df_sample_sents.drop(['Score'], axis=1, inplace=True)

    currentdate_yyyymmdd = get_current_datetime_num()

    logging.info(f'Saving chats_with_sentiment_{currentdate_yyyymmdd}')

    save_obj(output_path, df_sample_sents,
             'chats_with_sentiment_' + currentdate_yyyymmdd)
    # df_sample_sents = load_obj(output_path, 'chats_with_sentiment_' + currentdate_yyyymmdd)

    # Impute NANs with a value, otherwise Insert to DW fails
    df_sample_sents.fillna({'VIPLevel': 0, 'Brand': "", 'CountryGroup1': ""}, inplace=True)
    assert df_sample_sents.isnull().sum().sum() == 0

    logging.info(f'Imputed NAs')

    # Insert results in DW in batches
    chunks = np.ceil(len(df_sample_sents) / 1000)  # batches of size 1000

    # Clean and remove sensitive info
    df_sample_sents['html_text'] = df_sample_sents['html_text'].apply(lambda x: replace_emails(x, replace_with="<EMAIL>"))
    df_sample_sents['html_text'] = df_sample_sents['html_text'].apply(lambda x: replace_phone_numbers(x, replace_with="<PHONE NUMBER>"))
    df_sample_sents['html_text'] = df_sample_sents['html_text'].apply(lambda x: replace_urls(x, replace_with="<URL>"))

    df_sample_sents['text'] = df_sample_sents['text'].apply(lambda x: replace_emails(x, replace_with="<EMAIL>"))
    df_sample_sents['text'] = df_sample_sents['text'].apply(lambda x: replace_phone_numbers(x, replace_with="<PHONE NUMBER>"))
    df_sample_sents['text'] = df_sample_sents['text'].apply(lambda x: replace_urls(x, replace_with="<URL>"))

    logging.info(f'Inserting in DW')

    for i, chunk in tqdm(enumerate(np.array_split(df_sample_sents, chunks))):
        connect_insert_chatsentiment_sqlserver(
            df=chunk, 
            table="BI_Malta.dbo.fact_chat_sentiment")

    df_sample_sents.to_feather(output_path+'chat_sentiment.feather')
    gc.collect()

########

# df_sample_sents.iloc[10750]
# df_sample_sents.columns


'''
inputs = tokenizer("We are very happy to show you the 🤗 Transformers library.")
pt_batch = tokenizer(
    ["Porca miseria!",
     "I would love to kill you one day",
     "Jävla tyskar!",
     "You are the best :)",
     "Jävla horkärring, gå fan och knulla dig där bak i arslet så jävla hårt. Vilken liten jävla kukfitta du är."],
    padding=True, truncation=True,
    max_length=512,
    return_tensors="pt")

pt_outputs = model(**pt_batch)
pt_predictions = F.softmax(pt_outputs[0], dim=-1)
print(pt_predictions)
'''


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
task = 'sentiment'
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
    def __init__(self, text: str, created_at: datetime.datetime, get_sentiment_function, cursor, table_name):
        self.table_name = table_name

        self.text = text
        self.created_at = created_at
        sentiments = get_sentiment_function(self.text)

        self.pos = sentiments['positive']
        self.neg = sentiments['negative']
        self.neu = sentiments['neutral']

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
#data = df.head(100)

'''
model_path = './Models/' + model_name

directory = os.listdir(model_path)

if len(directory) == 0:
    print("Empty directory, so download model")

    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)

    tokenizer.save_pretrained(model_path)
    config.save_pretrained(model_path)
    model.save_pretrained(model_path)

    # Reload tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Reload tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
'''


'''
df2 = df.head(50)

start = time.time()
# results = classifier(df2['text'].to_list())
results = classifier(df['text'].to_list())
print('Time taken:', time.time() - start)

for i, r in enumerate(result):
    print(
        f"Phrase: {text[i]}: Sentiment: {r['label']}, Score:{np.round(r['score'], 2)} \n")

#  vocab_size

df2 = df.head(50)

start = time.time()

res = []
for index, row in tqdm(df.iterrows()):
    clf_result = classifier(row['text'])[0]
    res.append(clf_result)

    #df2['sentiment'] = clf_result['label']
    # df2['score'] = np.round(clf_result['score'], 2)

print('Time taken:', time.time() - start)


'''

'''


class SentimentDetection(object):
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        self.model_name = model_name

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.classifier = TextClassificationPipeline(model=model,
                                                     tokenizer=tokenizer,
                                                     return_all_scores=True,
                                                     task="sentiment-analysis")

    def get_sentiment_by_label(self, label: str):
        if label == "LABEL_0":
            return "negative"
        elif label == "LABEL_1":
            return "neutral"
        elif label == "LABEL_2":
            return "positive"
        else:
            print("SOMETHING IS WRONG")
            return ""

    def get_sentiment(self, phrase):
        results = self.classifier(phrase)
        res = dict()
        for result in results:
            for sentiment in result:
                res.update({self.get_sentiment_by_label(
                    sentiment['label']): sentiment['score']})
        return res


em = SentimentDetection(
    model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment")
em.get_sentiment("son of a bitch")
'''


'''
tqdm.pandas()
df['sentiment'] = df.progress_apply(lambda x: classifier(x['text']))

for index, row in tqdm(df.iterrows()):
    classifier(row['text'])

df['text']
df = (
    df
    .assign(sentiment=lambda x: x['text'].apply(lambda s: classifier(s)))
    .assign(
        label=lambda x: x['sentiment'].apply(lambda s: (s[0]['label'])),
        score=lambda x: x['sentiment'].apply(lambda s: (s[0]['score']))
    )
)

df.head(3)
'''


'''
https://towardsdatascience.com/how-to-apply-transformers-to-any-length-of-text-a5601410af7f
'''
