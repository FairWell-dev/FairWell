import os
import pandas as pd
import streamlit as st
import seaborn as sns
import torch

import features_page
import data_fairness_page
import model_bias_detection_page

def read_csv_list(file_list, selected=None):
    # Check for user uploads
    if not selected:
        selected = st.sidebar.selectbox("Select One (CSV)",
            options=[file.name for file in file_list],
            index=0)
    df_dict = {}
    select_key = None
    for file_path in file_list:
        df = pd.read_csv(file_path)
        if not isinstance(file_path, str):
            file_path = file_path.name
        key = os.path.basename(file_path)[:-4]
        df_dict[key] = df
        if file_path == selected:
            select_key = key
    return df_dict, select_key

# def run_inference(file_list, df_dict): #TODO
    # Loads list of models, runs inference and returns predictions
    # pred_dict = {}
    # select_key = None
    # for file_path in file_list:
    #     TODO: Add model loading and inference code
    #     pred_dict[file_path] = pred_df
    # return pred_dict, select_key

def sidebar_handler(label, type_list, eg_dict):
    eg_labels = eg_dict.keys()
    # Example Use Case
    st.sidebar.title('Example: NYC Subway Traffic')
    selected_eg = st.sidebar.selectbox('',
                                        options=eg_labels,
                                        index=0) 
    
    # User Upload
    st.sidebar.title('Upload')
    file_list = st.sidebar.file_uploader('%s, (%s)' % (label, ', '.join([type.upper() for type in type_list])),
                                        type = type_list,
                                        accept_multiple_files = True)
    
    # Load Files
    if file_list:
        csv_files = [file for file in file_list if file.type in ['text/csv', 'application/vnd.ms-excel']]
        pth_files = [file for file in file_list if file.type in ['application/octet-stream']] #TODO
        upload_dict = dict(zip(list(range(len(file_list))),file_list))
        df_dict, select_key = read_csv_list(csv_files)
        # TODO: Model Upload feature
        # if len(type_list) > 1:
            # Run Inference
            # return run_inference(models, df_dict)
        return df_dict, select_key
    else:
        return read_csv_list(eg_dict.values(), eg_dict[selected_eg])

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
    about = open('README.md', 'r')
    st.markdown(about.read())
elif page.lower() == 'feature exploration':
    features_page.render(sidebar_handler)
elif page.lower() == 'data fairness assessment':
    data_fairness_page.render(sidebar_handler)
elif page.lower() == 'model bias detection':
    model_bias_detection_page.render(sidebar_handler)
else:
    st.text('Page ' + page + ' is not implemented.')