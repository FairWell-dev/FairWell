import pandas as pd
import streamlit as st
import seaborn as sns

import features_page

# Config 
st.set_page_config(page_title='Responsible AI Toolkit', 
                   layout='wide', 
                   initial_sidebar_state='expanded')

# Sidebar
page = st.sidebar.radio('Page', 
                        options=['Feature Explorer'], 
                        index=0)
st.sidebar.markdown('**Select or Upload a Dataset**')

def update_selectbox(file):
    selectbox_options = [file.name]
    selectbox_idx = len(selectbox_options) - 1

dataset = st.sidebar.selectbox('Dataset', 
                               options=['NYC Subway'],
                               index=0)

file = st.sidebar.file_uploader('Upload Dataset (CSV)', type=['csv'])

dataset = 'diamonds'  # To remove after we preprocess our subway dataset

# Title
st.title('Responsible AI Toolkit')

# Extract data
if file is None:
    df = sns.load_dataset(dataset)
else:
    df = pd.read_csv(file)

if page.lower() == 'feature explorer':
    features_page.render(df)
else:
    st.text('Page ' + page + ' is not implemented.')