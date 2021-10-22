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
dataset = st.sidebar.text_input('Dataset', 
                                value='NYC Subway.csv')
dataset = 'diamonds'  # To remove after we preprocess our subway dataset

# Title
st.title('Responsible AI Toolkit')

# Extract data
df = sns.load_dataset(dataset)

if page.lower() == 'feature explorer':
      features_page.render(df)
else:
      st.text('Page ' + page + ' is not implemented.')