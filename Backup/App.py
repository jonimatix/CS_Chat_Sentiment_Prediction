import streamlit as st
from streamlit_multipage import MultiPage # pip install streamlit-multipage
# from streamlit.hashing import _CodeHasher
# from streamlit.report_thread import get_report_ctx
# from streamlit.server.server import Server

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

import importlib
from helperFunctions import *
import helperFunctions
from utils import *
import utils
importlib.reload(helperFunctions)
importlib.reload(utils)

print(os.getcwd())
warnings.filterwarnings('ignore')

# Debug the application by running this in command line:
# cd /d C:\
# cd C:\Users\Jonathanm\Desktop\LiveChat Chats EDA\App\Multi Page\
# streamlit run App.py --server.port 8502 --server.address "10.13.50.37"

input_path = '../../Input/'
output_path = '../../Output/'

KEEPALIVEHOURS = 1  # Keep cache alive for 1 hours (value in seconds)



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

    df = pd.read_feather(output_path+'chat_sentiment.feather')
    df.rename(columns = {'ID':'ChatID'}, inplace=True)
    # df.columns
    return df


@st.cache(ttl=60*60*KEEPALIVEHOURS)
def page_dashboard(st, **state):

    df = get_dataset_main()
    df_orig = df.copy()

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
    body3 = st.container()
    body1 = st.container()
    body2 = st.container()
    body4 = st.container()

    with header:
        st.title(':chart_with_upwards_trend: Chat Sentiment Dashboard')
        st.subheader(f'This application shows Customer Support Chats and the predicted sentiment')

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

    def apply_df_filters(df, Date, Country, Brand, VIP, ChatType, Sentiment, Duration, Agent):

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

        return df

    df = apply_df_filters(df_orig, SLIDER_DATE, COUNTRIES_SELECTED, BRANDS_SELECTED, SLIDER_VIP, MULTISELECT_CHATTYPE,
                          MULTISELECT_SENTIMENT, SLIDER_DURATION, MULTISELECT_AGENTDISPLAYNAME)

    with body3:
        col1, col2 = st.columns(2)

        with col1:
            x = df.copy()
            x['Date'] = pd.to_datetime(x['Date']).dt.round('H')
            x = x.groupby(['Date'],
                          as_index=False).size().reset_index(drop=True) # .to_frame()
            x.columns = ['Date', 'No of Chats']

            # fig = px.area(df, facet_col="company", facet_col_wrap=2)
            # fig = go.Figure([go.Scatter(x=df['Date'], y=df['AAPL.High'])])

            fig = px.line(x, x='Date', y="No of Chats",
                          title="No of Chats by Date Time")
    #        fig.update_yaxes(showgrid=False, gridwidth=1, gridcolor='LightPink')
            fig.update_layout(yaxis={'visible': True, 'showticklabels': False,
                                     'showgrid': False}, xaxis={'visible': True, 'title': ""})
            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7d", step="day",
                             stepmode="backward"),
                        dict(count=1, label="1m", step="month",
                             stepmode="backward"),
                        dict(count=6, label="6m", step="month",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig)

        with col2:
            x = df.copy()
            x['Date'] = pd.to_datetime(x['Date']).dt.round('H')
            x = x.groupby(['Date', 'Sentiment'],
                          as_index=False).size().reset_index(drop=True) # .to_frame()

            x.columns = ['Date', 'Sentiment', 'No of Chats']

            fig = px.line(x, x='Date', y="No of Chats",
                          color="Sentiment", title="Sentiment by Date Time")

            fig.update_yaxes(matches=None, showticklabels=True,
                             visible=True, title=None)
            fig.update_xaxes(matches=None, showticklabels=True,
                             visible=True, title=None)
            fig.update_annotations(font=dict(size=12))
            fig.update_layout(showlegend=True)
            fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)'})

            fig.update_xaxes(
                rangeslider_visible=True,
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="7d", step="day",
                             stepmode="backward"),
                        dict(count=1, label="1m", step="month",
                             stepmode="backward"),
                        dict(count=6, label="6m", step="month",
                             stepmode="backward"),
                        dict(count=1, label="YTD",
                             step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig)

    with body2:
        col1, col2 = st.columns(2)

        # st.write(df['html_text'][10], unsafe_allow_html=True)
        # st.markdown(df['html_text'][10], unsafe_allow_html=True,)

        with col1:

            x = df.groupby(['VIPLevel', 'Sentiment'],
                           as_index=False).size().reset_index(drop=True) # .to_frame()
            x.columns = ['VIPLevel', 'Sentiment', 'No of Chats']

            x_negative = x.query('Sentiment == "Negative"')
            x_neutral = x.query('Sentiment == "Neutral"')
            x_positive = x.query('Sentiment == "Positive"')

            fig = go.Figure(data=[
                go.Bar(name='Negative', x=x_negative.VIPLevel,
                       y=x_negative['No of Chats']),
                go.Bar(name='Neutral', x=x_neutral.VIPLevel,
                       y=x_neutral['No of Chats']),
                go.Bar(name='Positive', x=x_positive.VIPLevel,
                       y=x_positive['No of Chats'])
            ])

            # Add title
            fig.update_layout(
                title_text='No of Chats by VIP Level and Sentiment')

            st.plotly_chart(fig)

        with col2:
            x = df[['DurationInMins', 'Sentiment']]

            x_negative = x.query('Sentiment == "Negative"')['DurationInMins']
            x_neutral = x.query('Sentiment == "Neutral"')['DurationInMins']
            x_positive = x.query('Sentiment == "Positive"')['DurationInMins']

            hist_data, group_labels = [], []

            if len(x_negative) > 0:
                hist_data.append(x_negative)
                group_labels.append('Negative')
            if len(x_neutral) > 0:
                hist_data.append(x_neutral)
                group_labels.append('Neutral')
            if len(x_positive) > 0:
                hist_data.append(x_positive)
                group_labels.append('Positive')

            # hist_data = [x_negative, x_neutral, x_positive]
            # group_labels = ['Negative', 'Neutral', 'Positive']
            # colors = ['#A56CC1', '#A6ACEC', '#63F5EF']

            # Create distplot with curve_type set to 'normal'
            fig = ff.create_distplot(hist_data, group_labels, show_hist=False,
                                     bin_size=.4, show_rug=False)

            # Add title
            fig.update_layout(title_text='Density plot showing Chat Duration (mins) by Sentiment')
            # fig.show()

            st.plotly_chart(fig)

    with body4:

        x = df.copy()
        x['Date'] = pd.to_datetime(x['Date']).dt.round('H')
        x = x.groupby(['Date', 'CountryGroup1'],
                      as_index=False).size().reset_index(drop=True) # .to_frame()
        x.columns = ['Date', 'Country Group', 'No of Chats']

        fig = px.area(x, x='Date', y="No of Chats",
                      facet_col="Country Group", facet_col_wrap=4, color="Country Group")

        # fig.for_each_annotation(lambda a: a.update(text=""))
        fig.update_yaxes(matches=None, showticklabels=True,
                         visible=True, title=None)
        fig.update_xaxes(matches=None, showticklabels=True, visible=False)
        fig.update_annotations(font=dict(size=12))
        fig.update_layout(showlegend=False)
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1]))

        st.plotly_chart(fig)


def page_chatsearch(st, **state):

  
    df = get_dataset_main()
    df_orig = df.copy()

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

    df = apply_df_filters(df_orig, SLIDER_DATE, COUNTRIES_SELECTED, BRANDS_SELECTED, SLIDER_VIP, MULTISELECT_CHATTYPE,
                          MULTISELECT_SENTIMENT, SLIDER_DURATION, MULTISELECT_AGENTDISPLAYNAME, TEXTBOX_CHATTEXT)

    if BUTTON_DOWNLOAD:
        tmp_download_link = download_link(
            df.head(100).drop(["html_text"], axis = 1), 
            'Data.csv', 
            'Click to download data')
            
        st.sidebar.markdown(tmp_download_link, unsafe_allow_html=True)


    with body1:
        # start = time.time()
        # df_tbl = get_dataset_table(COUNTRIES_SELECTED, USERID_INPUT)
        # Same as st.write(df)

        show_n_rows = 1000
        df_t = df.head(show_n_rows).copy()
        
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


def page_chatdetails(st, **state):

    df = get_dataset_main()
    df_orig = df.copy()

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
    random_chatid = df_orig.query('ChatType == "Support Chat"')['ChatID'].sample(1).values[0]

    
    # if state.textinput_chatid is None:
    #     state.textinput_chatid = random_chatid

    # SELECTIONS
    TEXTBOX_CHATID = st.sidebar.text_input("Enter Chat ID", value=random_chatid)

    # Initialize the session state for ChatID
    MultiPage.save({"ChatID": TEXTBOX_CHATID}, namespaces=["Chats"])
    
    def apply_df_filters(df, ChatID):

        # Chat ID
        if not ChatID:
            # List is empty
            df = df
        else:
            df = df.query('ChatID == @ChatID')

        return df

    df = apply_df_filters(df_orig, TEXTBOX_CHATID)


    with body2:
        col1, col2 = st.columns(2)

        # st.write(df['html_text'][10], unsafe_allow_html=True)
        # st.markdown(df['html_text'][10], unsafe_allow_html=True,)
        with col1:
            st.subheader("Chat Transcript for Chat ID {}".format(TEXTBOX_CHATID))
            st.markdown(df['html_text'].values[0], unsafe_allow_html=True,)

        with col2:
            df_t = df.copy()

            df_t.drop(['html_text', 'text'], axis=1, inplace=True)

            df_t.columns = ['User ID', 'Date', 'VIP Level', 'Rating', 'Duration (Mins)',
                        'Brand', 'Country Group', 'Chat Type', 'Chat ID', 'Agent',
                        'Negative Score', 'Neutral Score', 'Positive Score', 'Sentiment']                        


            df_t = df_t.T
            df_t.columns = ['Value']
            st.dataframe(df_t, )

            sent = df.Sentiment.values[0]

            if sent == "Positive":
                st.success(f'Sentiment is {sent}')
            elif sent == "Neutral":
                st.info(f'Sentiment is {sent}')
            else:
                st.warning(f'Sentiment is {sent}')

            st.image(f'../../Images/{sent}.jpg')

        if df.Sentiment.values[0] == "Positive" and df.Positive_Score.values[0] >= 0.8:
            st.balloons()


st.set_page_config(layout="wide", page_title='Chat Sentiment App',  # page_icon=favicon,
                       initial_sidebar_state='auto')

app = MultiPage()
app.st = st

app.navbar_style = "HorizontalButton"
# app.navbar_name = "Pages:"
# app.next_page_button = "Next Page"
# app.previous_page_button = "Previous Page"
# app.reset_button = "Delete Cache"
#app.header = "HEY"
#app.footer = "YOU"
# app.navbar_extra = sidebar

app.add_app("Dashboard", page_dashboard)
app.add_app("Chat Search", page_chatsearch)
app.add_app("Chat Details", page_chatdetails)

app.run()
