import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

@st.cache
def infer_dtypes(df):
    dtype_dict = dict()

    for col_name in df.columns:
        series = df[col_name]
        dtype_dict[col_name] = 'categorical'

        if not pd.api.types.is_bool_dtype(series) and \
            pd.api.types.is_numeric_dtype(series): # Boolean type should be categorical instead of numeric
            dtype_dict[col_name] = 'numerical'

    return dtype_dict

@st.cache
def numerical_cols_summary_stats(df, numerical_col_names):
    if len(numerical_col_names) == 0:
        return pd.DataFrame()
    
    summary_df = df.loc[:, numerical_col_names] \
        .describe(percentiles=[0.5]) \
        .transpose() \
        .rename({'50%': 'median'}, axis='columns')

    num_rows = df.shape[0]

    # Calculate % missing
    missing = [df[col_name].isna().sum() / num_rows * 100 for col_name in numerical_col_names]
    summary_df['missing'] = missing

    # Count zeros
    zeros = [num_rows - np.count_nonzero(df[col_name]) for col_name in numerical_col_names]
    summary_df['zeros'] = zeros

    return summary_df

@st.cache
def categorical_cols_summary_stats(df, categorical_col_names):
    if len(categorical_col_names) == 0:
        return pd.DataFrame()

    return df.loc[:, categorical_col_names] \
        .describe(include='all') \
        .transpose()

@st.cache
def get_spec(numerical_summary_df, 
             categorical_summary_df, 
             numerical_cols_width_prop=0.6, 
             chart_col_width_scale=3):
    num_numerical_cols = numerical_summary_df.shape[1]
    num_categorical_cols = categorical_summary_df.shape[1]

    numerical_cols_spec = np.ones(num_numerical_cols)
    numerical_cols_spec = np.append(numerical_cols_spec, [chart_col_width_scale])  # Numerical features visualisations
    spec = numerical_cols_width_prop / numerical_cols_spec.sum() * numerical_cols_spec # Standardise prop to numerical width

    categorical_cols_spec = np.ones(num_categorical_cols)
    categorical_cols_spec = np.append(categorical_cols_spec, [chart_col_width_scale])  # Categorical features visualisations
    spec = np.append(spec, (1 - numerical_cols_width_prop) / categorical_cols_spec.sum() * categorical_cols_spec) # Standardise prop to numerical width

    return spec

def extract_row_data(row_idx, numerical_summary_df, categorical_summary_df, data_df):
    feature_names = list()
    data = list()

    if row_idx < numerical_summary_df.shape[0]:
        feature_names += [numerical_summary_df.iloc[row_idx].name]
        data += list(numerical_summary_df.iloc[row_idx].values)
        data += [plot_histogram(data_df[feature_names[-1]])]
    else:
        # Pad (there are more categorical columns)
        data += [''] * (numerical_summary_df.shape[1] + 1)  # Add 1 for chart col

    if row_idx < categorical_summary_df.shape[0]:
        feature_names += [categorical_summary_df.iloc[row_idx].name]
        data += list(categorical_summary_df.iloc[row_idx].values)
        data += [plot_histogram(data_df[feature_names[-1]])]

    return (feature_names, data)

@st.cache
def plot_histogram(series):
    fig = go.Figure(data=go.Histogram(x=series))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=100)

    return fig

def render(extract_data):
    df = extract_data('csv')
    st.sidebar.title('Options')
    # Main
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader('Data Sample (100 rows)')
        st.dataframe(df.sample(100))

    with col2:
        st.subheader('Inferred Data Types')
        dtype_dict = infer_dtypes(df)
        st.json(dtype_dict)

    numerical_cols_width_prop = 0.5
    numerical_col, categorical_col = st.columns([
          numerical_cols_width_prop, 
          1 - numerical_cols_width_prop
    ])
    
    with numerical_col:
        numerical_col_names = [col for col, dtype in dtype_dict.items() if dtype == 'numerical']
        num_num = len(numerical_col_names)
        subset_numerical_col = numerical_col_names[0:num_num+1]
        if num_num > 4:
            subset_numerical_col = numerical_col_names[0:5]
        select_numerical_col = st.sidebar.multiselect('Numerical Features (' + str(len(numerical_col_names)) + ')', 
                               options=numerical_col_names,
                               default=subset_numerical_col) 
        st.subheader('Numerical features (' + str(len(select_numerical_col)) + ')')
    with categorical_col:
        categorical_col_names = [col for col, dtype in dtype_dict.items() if dtype == 'categorical']
        num_cat = len(categorical_col_names)
        subset_categorical_cols = categorical_col_names[0:num_cat+1]
        if num_cat > 4:
            subset_categorical_cols = categorical_col_names[0:5]
        select_categorical_col = st.sidebar.multiselect('Categorical Features (' + str(len(categorical_col_names)) + ')', 
                                options=categorical_col_names,
                                default=subset_categorical_cols) 
        st.subheader('Categorical features (' + str(len(select_categorical_col)) + ')')

    # Calculate summaries
    numerical_summary_df = numerical_cols_summary_stats(df, select_numerical_col)
    categorical_summary_df = categorical_cols_summary_stats(df, select_categorical_col)

    # Get column specifications for rendering
    spec = get_spec(numerical_summary_df, 
                    categorical_summary_df, 
                    numerical_cols_width_prop=numerical_cols_width_prop)
    
    # Render column names
    cols = st.columns(spec=spec)
    col_names = list(numerical_summary_df.columns) + \
        ['chart'] + \
        list(categorical_summary_df.columns) + \
        ['chart']

    for col, col_name in zip(cols, col_names):
        with col:
            st.markdown('**' + col_name + '**')
    
    # Render values
    num_rows = max(numerical_summary_df.shape[0], categorical_summary_df.shape[0])

    for idx in range(num_rows):
        feature_names, row_data = extract_row_data(idx, 
                                                   numerical_summary_df, 
                                                   categorical_summary_df, 
                                                   df)

        # Render feature names as separate row
        cols = st.columns(spec=[numerical_cols_width_prop, (1 - numerical_cols_width_prop)])
        for col, feature_name in zip(cols, feature_names):
            with col:
                  col.text(feature_name)

        # Render feature summary and chart
        cols = st.columns(spec=spec)
        for col, (col_idx, col_name), data in zip(cols, enumerate(col_names), row_data):
            with col:
                if col_name.lower() == 'chart':
                    st.plotly_chart(data,
                                    use_container_width=True,
                                    config={'responsive': False, 'displayModeBar': False})
                elif isinstance(data, str):
                    st.text(data)
                elif col_name.lower() == 'count':
                    st.text(round(data))
                elif col_name.lower() in ['missing', 'zeros']:
                    st.text(str(round(data, 2)) + '%')
                else:
                    st.text(round(data, 2))