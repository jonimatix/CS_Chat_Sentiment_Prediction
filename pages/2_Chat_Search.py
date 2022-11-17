import streamlit as st
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from configparser import ConfigParser
import pyodbc

from datetime import datetime, timedelta
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

st.set_page_config(layout="wide", page_title='Chat Search',  # page_icon=favicon,
                   initial_sidebar_state='auto')
                   
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1', float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'



@st.cache(ttl=60*60*KEEPALIVEHOURS)
def get_dataset_main():

    df = pd.read_feather(OUTPUT_PATH+'chat_sentiment.feather')
    df.rename(columns = {'ID':'ChatID'}, inplace=True)
    # df.columns
    return df


df = get_dataset_main()

date_min = df.Date.min()
date_max = df.Date.max() + timedelta(days=1)

viplevel_min = int(df.VIPLevel.min())
viplevel_max = int(df.VIPLevel.max())

durationinmin_min = int(df.DurationInMins.min())
durationinmin_max = int(df.DurationInMins.max())

BRANDS = sorted(df.Brand.unique())
BRANDS = ["All Brands"] + BRANDS

COUNTRIES = sorted(df.CountryGroup1.unique())
COUNTRIES = ["All Countries"] + COUNTRIES

AGENTS = sorted(df.AgentDisplayName.unique())
AGENTS = ["All Agents"] + AGENTS

# st.sidebar.text('You selected {}'.format(COUNTRIES))

header = st.container()
body1 = st.container()

with header:
    st.title(':chart_with_upwards_trend: Chat Search')

st.sidebar.title(":wrench: Options")
# st.sidebar.markdown("Select your options")

# SELECTIONS
SLIDER_DATE = st.sidebar.date_input("Select Date", [date_min, date_max])

COUNTRIES_SELECTED = st.sidebar.multiselect(
    'Select Countries', COUNTRIES, default="All Countries")

BRANDS_SELECTED = st.sidebar.multiselect(
    'Select Brands', BRANDS, default="All Brands")

SLIDER_VIP = st.sidebar.slider(
    "Select VIP Level", min_value=viplevel_min, max_value=viplevel_max, value=(viplevel_min, viplevel_max), step=1)

SLIDER_DURATION = st.sidebar.slider(
    "Select Duration (min)", min_value=durationinmin_min, max_value=durationinmin_max, value=(durationinmin_min, durationinmin_max), step=1)

MULTISELECT_CHATTYPE = st.sidebar.multiselect(
    'Select Chat Type', options=list(df.ChatType.unique()), default=list(df.ChatType.unique()))

MULTISELECT_AGENTDISPLAYNAME = st.sidebar.multiselect(
    'Select Agent', AGENTS, default="All Agents")

MULTISELECT_SENTIMENT = st.sidebar.multiselect('Select Sentiment',
                                                ['Negative', 'Neutral', 'Positive'], default=['Negative', 'Neutral', 'Positive'])

TEXTBOX_CHATTEXT = st.sidebar.text_input(
    "Search Chat Transcript (Use | for multiple keywords)")

BUTTON_DOWNLOAD = st.sidebar.button("Download first 100 records")




def apply_df_filters(df, Date, Country, Brand, VIP, ChatType, Sentiment, Duration, Agent, ChatText):

    # Date Filter
    df = df[(df['Date'] >= pd.to_datetime(Date[0])) &
            (df['Date'] <= pd.to_datetime(Date[1]))]

    # Country Filter
    if not Country:
        # List is empty
        df = df
    elif "All Countries" in Country:
        df = df
    else:
        df = df.query('CountryGroup1 == @Country')

    # Brand Filter
    if not Brand:
        # List is empty
        df = df
    elif "All Brands" in Brand:
        df = df
    else:
        df = df.query('Brand == @Brand')

    # VIP Filter
    df = df.query(
        'VIPLevel >= {} and VIPLevel <= {}'.format(VIP[0], VIP[1]))

    # Chat Type
    if not ChatType:
        # List is empty
        df = df
    else:
        df = df.query('ChatType == @ChatType')

    # Sentiment Filter
    if not Sentiment:
        # List is empty
        df = df
    else:
        df = df.query('Sentiment == @Sentiment')

    # Duration Filter
    df = df.query('DurationInMins >= {} and DurationInMins <= {}'.format(
        Duration[0], Duration[1]))

    # Agent
    if not Agent:
        # List is empty
        df = df
    elif "All Agents" in Agent:
        df = df
    else:
        df = df.query('AgentDisplayName == @Agent')

    # Chat Text Filter
    if len(ChatText) > 0:
        df = df[df['html_text'].str.contains(
            ChatText)]  # Ex. "Hello|Britain"

    return df


df_orig = apply_df_filters(df, SLIDER_DATE, COUNTRIES_SELECTED, BRANDS_SELECTED, SLIDER_VIP, MULTISELECT_CHATTYPE,
                        MULTISELECT_SENTIMENT, SLIDER_DURATION, MULTISELECT_AGENTDISPLAYNAME, TEXTBOX_CHATTEXT)

if BUTTON_DOWNLOAD:
    tmp_download_link = download_link(
        df_orig.head(100).drop(["html_text"], axis = 1), 
        'Data.csv', 
        'Click to download data')
        
    st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)


with body1:
    # start = time.time()
    # df_tbl = get_dataset_table(COUNTRIES_SELECTED, USERID_INPUT)
    # Same as st.write(df)

    show_n_rows = 500
    df_t = df_orig.head(show_n_rows).copy()
    
    if 'html_text' in df_t.columns: df_t.drop(['html_text'], axis=1, inplace=True)
    
    #df_t.columns = ['Chat ID', 'User ID', 'Date', 'VIP Level', 'Rating', 'Duration (Mins)',
        #               'Brand', 'Country Group', 'Chat Type', 'Negative Score', 'Neutral Score',
        #              'Positive Score', 'Sentiment', 'Text', 'Agent']

    df_t.columns = ['Text', 'User ID', 'Date', 'VIP Level', 'Rating', 'Duration (Mins)',
                    'Brand', 'Country Group', 'Chat Type', 'Chat ID', 'Agent',
                    'Negative Score', 'Neutral Score', 'Positive Score', 'Sentiment']                        

    df_t.drop(['Rating', 'Negative Score', 'Neutral Score',
                'Positive Score'], axis=1, inplace=True)

    st.subheader(f'Showing first {show_n_rows} chats. Search returned {df.shape[0]} chats')

    df_t['Duration (Mins)'] = np.round(df_t['Duration (Mins)'], 0)
    df_t['Text'] = df_t['Text'].apply(lambda x: x[:100]) + "..."

    def color_sentiment(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
        if val == "Negative":
            color = 'orange'
        elif val == "Neutral":
            color = "blue"
        else:
            color = 'green'

        return 'color: %s' % color

    st.table(df_t.style.applymap(color_sentiment, subset=['Sentiment']))



