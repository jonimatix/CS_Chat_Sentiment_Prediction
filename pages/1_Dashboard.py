import streamlit as st
import base64
from io import BytesIO

import pandas as pd
import numpy as np
import warnings
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

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

st.set_page_config(layout="wide", page_title='Chat Sentiment App',  # page_icon=favicon,
                       initial_sidebar_state='auto')


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


# SIDE BAR STUFF
BRANDS = sorted(df.Brand.unique())
BRANDS = ["All Brands"] + BRANDS

COUNTRIES = sorted(df.CountryGroup1.unique())
COUNTRIES = ["All Countries"] + COUNTRIES

AGENTS = sorted(df.AgentDisplayName.unique())
AGENTS = ["All Agents"] + AGENTS

# st.sidebar.text('You selected {}'.format(COUNTRIES))

header = st.container()
body1 = st.container()
body2 = st.container()
body3 = st.container()
body4 = st.container()

with header:
    st.title(':chart_with_upwards_trend: Chat Sentiment Dashboard')
    st.subheader(f'Customer Support activity dashboard and predicted sentiment')

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


df_orig = apply_df_filters(df, SLIDER_DATE, COUNTRIES_SELECTED, BRANDS_SELECTED, SLIDER_VIP, MULTISELECT_CHATTYPE,
                        MULTISELECT_SENTIMENT, SLIDER_DURATION, MULTISELECT_AGENTDISPLAYNAME)

with body3:
    col1, col2 = st.columns(2)

    with col1:
        x = df_orig.copy()
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
        x = df_orig.copy()
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

        x = df_orig.groupby(['VIPLevel', 'Sentiment'],
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
        x = df_orig[['DurationInMins', 'Sentiment']]

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

    x = df_orig.copy()
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




