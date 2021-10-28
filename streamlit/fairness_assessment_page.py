import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(extract_data):
    df = extract_data('csv')
    st.sidebar.title('Options')
    dataset = st.sidebar.selectbox('Fairness Metrics', 
                               options=['Class Imbalance (CI)', 'Jensen-Shannon Divergence (JS)'],
                               index=0) 

    # Extract data
    try:
        df = pd.read_csv("data/final.csv")
    except:
        st.error('Example dataset file not found, please upload dataset.')
        st.sidebar.error('Example dataset file not found, please upload dataset.')
        df = pd.read_csv(file)
    
    # Main
    st.subheader("Faireness Assessment on Training Data")