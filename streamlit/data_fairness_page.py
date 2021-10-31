import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from itertools import combinations
from numpy import log

@st.cache
def infer_dtypes(df):
    dtype_dict = dict()

    for col_name in df.columns:
        series = df[col_name]
        dtype_dict[col_name] = 'categorical'
        nunique = df[col_name].nunique()

        if nunique > 10 and \
            not pd.api.types.is_bool_dtype(series) and \
            pd.api.types.is_numeric_dtype(series):
            dtype_dict[col_name] = 'numerical'
    
    return dtype_dict

@st.cache
def get_cols(df):
    cols = list(df.columns)

    return cols

@st.cache
def get_js(class_p_cnt, class_cnt, pair, col):
    try:
        p0 = class_p_cnt[pair[0]]/class_cnt[pair[0]]
        p1 = class_p_cnt[pair[1]]/class_cnt[pair[1]]
    except KeyError as e:
        print('Could not find probability of {} class in {} column'.format(e,col))
        p0 = p1 = js = np.nan
    if p0!=np.nan and p1!=np.nan:
        p = 0.5*(p0+p1)
        js = 0.5*((p0*log(p0/p)+(1-p0)*log((1-p0)/(1-p))) + (p1*log(p1/p)+(1-p1)*log((1-p1)/(1-p))))
    return js

@st.cache
def get_ci(class_cnt, pair):
    ci = (class_cnt[pair[0]]-class_cnt[pair[1]])/(class_cnt[pair[0]]+class_cnt[pair[1]])
    return ci

@st.cache
def get_metric(df, col):
    col_score_list = list()
    class_cnt = df[col].value_counts()
    class_comb = combinations(class_cnt.index,2)
    class_p_cnt = df[df['binary_target'].isin([1])][col].value_counts()
    for pair in class_comb:
        tmp = {'pair':str(pair)}

        # calculate ci
        ci = get_ci(class_cnt, pair)
        tmp['Class Imbalance'] = ci
        
        # calculate js
        js = get_js(class_p_cnt, class_cnt, pair, col)
        tmp['Jensen-Shannon Divergence'] = js

        col_score_list.append(tmp)

    return col_score_list

def render(sidebar_handler):
    # Sidebar
    dataset_dict, df = sidebar_handler('Training Dataset(s) for Fairness Exploration', ['csv'], ['data/final.csv'])

    # Main
    st.subheader("Fairness Assessment on Training Data")

    df_name_list = ['Dataset 1', 'Dataset 2']

    # Config
    col1, col2 = st.columns(2)

    with col1: 
        target = st.selectbox('Target Feature',
                            options=get_cols(df))
        
        col_to_select=[col for col in get_cols(df) 
                        if infer_dtypes(df)[col]=='categorical'
                        and df[col].nunique()<=10
                        and col!=target]
        container = st.container()
        all = st.checkbox('Select all')
        if all:
            eval_cols = container.multiselect('Features to evaluate (select one or more)',
                                            col_to_select,
                                            col_to_select)
        else:
            eval_cols = container.multiselect('Features to evaluate (select one or more)',
                                            col_to_select)

        metric_to_show = st.radio('Metric', 
                options=['Class Imbalance', 'Jensen-Shannon Divergence'])

        run_comparison = st.button('Run Fairness Assessment')

    with col2:
        with st.expander('Show subgroups (unique values) of selected features to evaluate'):
            unique_vals_dict = dict()
            for col in eval_cols:
                unique_vals_dict[col] = str(df[col].unique().tolist())
            st.json(unique_vals_dict)

    if df[target].nunique() > 2:
        run_comparison = False
        st.write('Not implemented: Target variable ' + target + ' has more than 2 unique values, fairness metrics cannot be calculated.')

    if run_comparison:
        st.subheader('Fairness Assessment')

        for col in eval_cols:
            st.markdown('##### ' + col.replace('_', ' '))

            if df[col].nunique() > 10:
                st.text(col + ' has too many subgroups (' + str(nunique) + '). Metrics and comparisons will not be run.')
                continue

            col1, col2 = st.columns([0.6, 0.4])

            with col1:
                metric_df = pd.DataFrame(get_metric(df,col))
                max_metric_df = pd.DataFrame({
                    'Max Class Imbalance': metric_df['Class Imbalance'].max(),
                    'Max Jensen-Shannon Divergence': metric_df['Jensen-Shannon Divergence'].max()
                    }, index=['Dataset 1'])
                st.table(max_metric_df)

                overall_df=pd.DataFrame(columns=['pair', 'Class Imbalance', 'Jensen-Shannon Divergence','dataset'])
                for df_name in dataset_dict.keys():
                    with st.expander('Details: ' + df_name, expanded=False):
                        dataset_metric_df = pd.DataFrame(get_metric(dataset_dict[df_name],col))
                        dataset_metric_df['dataset']=df_name
                        overall_df = overall_df.append(dataset_metric_df)
                        dataset_metric_df.drop(columns=['dataset'], inplace=True)
                        st.table(dataset_metric_df.set_index('pair'))

            with col2:
                x = overall_df['pair']
                y = overall_df[metric_to_show]
                fig = px.scatter(x=x,
                                y=y,
                                color=overall_df['dataset'],
                                labels={'x':'Subgroups', 'y':metric_to_show})
                fig.update_traces(textposition='top right')
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig,
                    use_container_width=True,
                    config={'responsive': False, 'displayModeBar': False})

























