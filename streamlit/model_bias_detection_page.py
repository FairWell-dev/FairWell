import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def render(extract_data):
    # Sidebar
    st.sidebar.title('Upload')
    file = st.sidebar.file_uploader('Upload Model (PTH)', type=['pth']) 
    
    # Main
    st.subheader("Bias Detection on Trained Model")
    st.subheader("Recommended Bias Mitigation Techniques")
