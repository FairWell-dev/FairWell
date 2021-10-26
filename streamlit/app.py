import pandas as pd
import streamlit as st
import seaborn as sns

import features_page
import fairness_assessment_page
import model_bias_detection_page


def extract_data(type):
    # Sidebar
    st.sidebar.title('Upload')
    file = st.sidebar.file_uploader('Upload Dataset (%s)' % type.upper(), type=[type])

    # Extract data
    try:
        df = pd.read_csv("data/final.csv")
    except:
        st.error('Example dataset file not found, please upload dataset.')
        st.sidebar.error('Example dataset file not found, please upload dataset.')
        df = pd.read_csv(file)
    return df

# Config 
st.set_page_config(page_title='FairWell', 
                   layout='wide', 
                   initial_sidebar_state='expanded')

# Sidebar
st.sidebar.title('FairWell')
page = st.sidebar.radio('Navigate', 
                        options=['About',
                        'Feature Exploration', 
                        'Data Fairness Assessment', 
                        'Model Bias Detection'], # geospatial
                        index=0)

st.sidebar.title('Example')  
dataset = st.sidebar.selectbox('Dataset', 
                               options=['NYC Subway'],
                               index=0)

# Title
st.title('FairWell')

# Pages
if page.lower() == 'about':
    about = open('streamlit/about.md', 'r')
    st.markdown(about.read())
elif page.lower() == 'feature exploration':
    features_page.render(extract_data)
elif page.lower() == 'data fairness assessment':
    fairness_assessment_page.render(extract_data)
elif page.lower() == 'model bias detection':
    model_bias_detection_page.render(extract_data)
else:
    st.text('Page ' + page + ' is not implemented.')