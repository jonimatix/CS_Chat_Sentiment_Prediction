import streamlit as st
import importlib, sys

sys.path.insert(0, './Scripts')
from helperFunctions import *
import helperFunctions
from utils import *
import utils
# importlib.reload(Scripts.helperFunctions)
# importlib.reload(Scripts.utils)

# print(os.getcwd())

INPUT_PATH = './Input/'
OUTPUT_PATH = './Output/'

KEEPALIVEHOURS = 1  # Keep cache alive for 1 hours (value in seconds)



def run():

    st.set_page_config(layout="wide", page_title='Customer Support Chats',  # page_icon=favicon,
                   initial_sidebar_state='auto')

    st.title('Customer Support Analytics')

    st.sidebar.success("Select page from the above")

    st.markdown(
        """
        The application uses sampled data to analyse the performance of Customer Support department.
        
        Chat data (from LiveChat) was heavily cleaned, personal information was removed as much as possible, and formatted to HTML as a preliminary step.
        
        Chat sentiment was then extracted using Transformer model from HuggingFace 🤗 named cardiffnlp/twitter-xlm-roberta-base-sentiment (https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
        
        👈 **Select a page from the sidebar** 
    """
    )


if __name__ == "__main__":
    run()
    # Run: 
    # cd C:\Users\Jonathan\Desktop\GameRecommendationApp
    # streamlit run Main_Page.py
    # http://192.168.1.101:8501