import streamlit as st
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import warnings

import os
import gc
import time
import io
import sys
import re
from os import path

print(os.getcwd())
# os.chdir('../')
warnings.filterwarnings('ignore')

# Debug the application by running this in command line:
# cd /d C:\
# cd C:\Users\Jonathanm\Desktop\LiveChat Chats EDA\App\Multi Page\
# streamlit run App.py --server.port 8502 --server.address "10.13.50.37"

INPUT_PATH = './Input/'
OUTPUT_PATH = './Output/'

KEEPALIVEHOURS = 1  # Keep cache alive for 1 hours (value in seconds)

st.set_page_config(layout="wide", page_title='Chat Details',  # page_icon=favicon,
                   initial_sidebar_state='auto')


@st.cache(ttl=60*60*KEEPALIVEHOURS)
def get_dataset_main():

    df = pd.read_feather(OUTPUT_PATH+'chat_sentiment.feather')
    df.rename(columns = {'ID':'ChatID'}, inplace=True)
    # df.columns
    return df


df = get_dataset_main()

# st.sidebar.text('You selected {}'.format(COUNTRIES))

header = st.container()
body2 = st.container()

with header:
    st.title(':chart_with_upwards_trend: Chat Details')
    # st.subheader(f'Please input a Chat ID')
    # st.text("")

st.sidebar.title(":wrench: Options")
# st.sidebar.markdown("Select your options")

# Get a random ChatID
random_chatid = df.query('ChatType == "Support Chat"')['ChatID'].sample(1).values[0]

# SELECTIONS
TEXTBOX_CHATID = st.sidebar.text_input("Enter Chat ID")


def apply_df_filters(df, ChatID):

    # Chat ID
    if not ChatID or ChatID is None or ChatID == "":
        # List is empty
        # df = df
        random_chatid = df.query('ChatType == "Support Chat"')['ChatID'].sample(1).values[0]
        df = df.query('ChatID == @random_chatid')
    else:
        df = df.query('ChatID == @ChatID')

    return df


df_orig = apply_df_filters(df, TEXTBOX_CHATID)


with body2:
    col1, col2 = st.columns(2)

    # st.write(df['html_text'][10], unsafe_allow_html=True)
    # st.markdown(df['html_text'][10], unsafe_allow_html=True,)
    with col1:
        st.subheader("Chat Transcript for Chat ID {}".format(TEXTBOX_CHATID))
        st.markdown(df['html_text'].values[0], unsafe_allow_html=True,)

    with col2:
        df_t = df_orig.copy()

        df_t.drop(['html_text', 'text'], axis=1, inplace=True)

        df_t.columns = ['User ID', 'Date', 'VIP Level', 'Rating', 'Duration (Mins)',
                    'Brand', 'Country Group', 'Chat Type', 'Chat ID', 'Agent',
                    'Negative Score', 'Neutral Score', 'Positive Score', 'Sentiment']                        


        df_t = df_t.T
        df_t.columns = ['Value']
        st.dataframe(df_t, )

        sent = df_orig.Sentiment.values[0]

        if sent == "Positive":
            st.success(f'Sentiment is {sent}')
        elif sent == "Neutral":
            st.info(f'Sentiment is {sent}')
        else:
            st.warning(f'Sentiment is {sent}')

        st.image(f'./Images/{sent}.jpg')

    try:
        if df_orig.Sentiment.values[0] == "Positive" and df_orig.Positive_Score.values[0] >= 0.8:
            st.balloons()
    except: pass

