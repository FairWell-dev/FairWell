import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render(sidebar_handler):
    # Sidebar
    df_list, df = sidebar_handler('Training Dataset(s) for Fairness Exploration', ['csv'], ['data/final.csv', 'data/final_copy.csv'])
    st.sidebar.title('Options')
    dataset = st.sidebar.selectbox('Fairness Metrics', 
                               options=['Class Imbalance (CI)', 'Jensen-Shannon Divergence (JS)'],
                               index=0)
    
    # Main
    st.subheader("Faireness Assessment on Training Data")
    st.subheader('Data Sample (5 rows)')
    st.dataframe(df.sample(5))