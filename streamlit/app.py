import pandas as pd
import streamlit as st
import seaborn as sns
# import torch

import features_page
import data_fairness_page
import model_bias_detection_page

def read_csv_list(file_list, selected=None):
    if not selected:
        selected = st.sidebar.selected("Select One (CSV)",
            options=[file.name for file in file_list],
            index=0)
    df_list = []
    select_df = pd.DataFrame()
    for file_path in file_list:
        df = pd.read_csv(file_path)
        df_list.appent(df)
        if not isinstance(file_path, str):
            file_path = file_path.name
        if file_path == selected:
            select_df = df
    return df_list, select_df

def read_pth_list(file_list, selected=None): #TODO
    if not selected:
        selected = st.sidebar.selected("Select One (PTH)",
            options=[file.name for file in file_list],
            index=0)
    model = []
    select_model = 'placeholder'
    for file_path in file_list:
        # model = torch.load(file_path)
        model_list.appent(df)
        if not isinstance(file_path, str):
            file_path = file_path.name
        if file_path == selected:
            select_model = model
    return model, select_model

def sidebar_handler(label, type_list, sample_file_paths):
    # Example Use Case
    st.sidebar.title('Example')
    select_sample = st.sidebar.selectbox(label,
                                        options=sample_file_paths,
                                        index=0) 
    
    # User Upload
    st.sidebar.title('Upload')
    file_list = st.sidebar.file_uploader('%s, (%s)' % (label, ', '.join([type.upper() for type in type_list])),
                                        type = type_list,
                                        accept_multiple_files = True)
    
    if file_list:
        csv_files = [file for file in file_list if file.type == 'text/csv']
        pth_files = [file for file in file_list if file.type == 'pth'] #TODO
        all_uploads, select_upload = read_csv_list(csv_files)
        if len(type_list) > 1:
            all_models, select_model = read_pth_list(pth_files)
            all_uploads = (all_preds, all_models)
            select_upload = (select_pred, select_model)
        return all_uploads, select_upload
    else:
        sample_csv_files = [file for file in sample_file_paths if file.endswith('.csv')]
        return read_csv_list(sample_csv_files, select_sample)

# Config 
st.set_page_config(page_title='FairWell', 
                   layout='wide', 
                   initial_sidebar_state='expanded')

# Sidebar
st.sidebar.title('FairWell')
page = st.sidebar.radio('Navigate', 
                        options=['Guide',
                        'Feature Exploration', 
                        'Data Fairness Assessment', 
                        'Model Bias Detection'], # geospatial
                        index=0)

# Title
st.title('FairWell')

# Pages
if page.lower() == 'guide':
    about = open('streamlit/about.md', 'r')
    st.markdown(about.read())
elif page.lower() == 'feature exploration':
    features_page.render(extract_data)
elif page.lower() == 'data fairness assessment':
    data_fairness_page.render(extract_data)
elif page.lower() == 'model bias detection':
    model_bias_detection_page.render(extract_data)
else:
    st.text('Page ' + page + ' is not implemented.')