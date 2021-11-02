import json
import numpy as np
import os
import pandas as pd
import streamlit as st
import seaborn as sns
import torch

import features_page
import data_fairness_page
import model_bias_detection_page

def convert_to_tensor(df, columns, type='Float'):
    arr = np.array(df[[*columns]]).astype(int) 
    tensor = torch.Tensor(arr)

    if type == 'Long':
        return tensor.type(torch.LongTensor)
    
    return tensor

def predict(model, inputs, threshold=0.5):
    """
    :param model: Torchscript model
    :param inputs: Torch tensor (or tuple of Torch tensors) to be fed to model
    :param threshold: Classification threshold value, default 0.5
    :return pred_proba: Numpy array, probability predictions for class 1
    :return y_pred: Numpy array, predicted labels (0/1) based on threshold
    """
    with torch.no_grad():
      model.eval()
      pred_proba = model(*inputs)

    # convert from tensor to np array
    pred_proba = pred_proba.detach().cpu().numpy()   
    
    y_pred = [1 if i >= threshold else 0 for i in pred_proba]

    return pred_proba, y_pred

def read_csv_list(file_list, selected=None):
    # Check for user uploads
    if not selected:
        selected = st.sidebar.selectbox("Select one dataset for selection of features.",
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

def run_inference(file_list, df_dict, json_files): #TODO
    # Loads list of models, runs inference and returns predictions
    pred_dict = {}
    for file_path in file_list:
        # Define key from file_path
        key = os.path.basename(file_path.name)[:-3]

        # Get corresponding feature_dict
        json_file = [file for file in json_files 
                if file.name[:-5] == key][0]
        feature_dict = json.load(json_file)
        
        # Get corresponding test data
        test_df = df_dict[key]
        x1_ts = convert_to_tensor(test_df, feature_dict.get('x1'), type='Long')
        x2_ts = convert_to_tensor(test_df, feature_dict.get('x2'))
        
        # Load model and get predictions
        model = torch.jit.load(file_path)
        pred_proba, y_pred = predict(model, (x1_ts, x2_ts))
        test_df[feature_dict['y'][0]+'_prediction'] = y_pred
        test_df[feature_dict['y'][0]+'_probability'] = pred_proba
        pred_dict[key] = test_df
    return pred_dict

def sidebar_handler(label, type_list, eg_dict):
    
    # Example Use Case
    eg_labels = eg_dict.keys()
    selected_eg = list(eg_labels)[0]
    st.sidebar.title('Example: NYC Subway Traffic')
    example = ''
    for dataset in eg_labels:
        example += '- **%s**\n' % dataset
    st.sidebar.markdown(example)

    # User Upload
    st.sidebar.title('Upload')
    file_list = st.sidebar.file_uploader('%s, (%s)' % (label, ', '.join([type.upper() for type in type_list])),
                                        type = type_list,
                                        accept_multiple_files = True)
    
    # Load Files
    if file_list:
        csv_files = [file for file in file_list if file.type in ['text/csv', 'application/vnd.ms-excel']]
        pt_files = [file for file in file_list if file.type in ['application/octet-stream']]
        json_files =  [file for file in file_list if file.type in ['application/json']]
        df_dict, select_key = read_csv_list(csv_files)
        if len(type_list) > 1:
            try:
            # Run Inference
                pred_dict = run_inference(pt_files, df_dict, json_files)
                selected = pred_dict[select_key]
                return pred_dict, select_key
            except:
                st.warning("Please ensure you have uploaded the corresponding model, test dataset and features json files with the same name for each model")
                return read_csv_list(eg_dict.values(), eg_dict[selected_eg])
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
                        'Feature Explorer', 
                        'Data Fairness Assessment', 
                        'Model Bias Detection & Mitigation'],
                        index=0)

# Title
st.title('FairWell')

# Pages
if page.lower() == 'guide':
    about = open('README.md', 'r')
    st.markdown(about.read())
elif page.lower() == 'feature explorer':
    features_page.render(sidebar_handler)
elif page.lower() == 'data fairness assessment':
    data_fairness_page.render(sidebar_handler)
elif page.lower() == 'model bias detection & mitigation':
    model_bias_detection_page.render(sidebar_handler)
else:
    st.text('Page ' + page + ' is not implemented.')
