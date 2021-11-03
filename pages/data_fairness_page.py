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
def get_class_summary(target, df, col):
    class_cnt = df[col].value_counts()
    class_p_cnt = df[df[target].isin([df[target].unique()[1]])][col].value_counts()

    return class_cnt, class_p_cnt

@st.cache
def get_metric(target, df, col):
    col_score_list = list()
    class_cnt = df[col].value_counts()
    class_comb = combinations(class_cnt.index,2)
    class_p_cnt = df[df[target].isin([df[target].unique()[1]])][col].value_counts()

    for pair in class_comb:
        tmp = {'Subgroup Pair':str(pair)}

        # calculate ci
        ci = get_ci(class_cnt, pair)
        tmp['Class Imbalance'] = ci
        
        # calculate js
        js = get_js(class_p_cnt, class_cnt, pair, col)
        tmp['Jensen-Shannon Divergence'] = js

        col_score_list.append(tmp)

    return col_score_list

def render(sidebar_handler):
    eg_dict = {
        'Baseline Dataset': 'data/baseline_binned.csv',
        'Bias Mitigated - Undersampling Dataset': 'data/undersampling_binned.csv'
    }

    # Sidebar
    dataset_dict, select_key = sidebar_handler('Dataset(s) for Fairness Assessment', ['csv'], eg_dict)
    df = dataset_dict[select_key]

    # Main
    st.subheader("Fairness Assessment on Dataset")

    # Config
    col1, col2 = st.columns(2)

    with col1: 
        target = st.selectbox('Target Feature',
                            options=get_cols(df))
        
        col_to_select=[col for col in get_cols(df) 
                        if infer_dtypes(df)[col]=='categorical'
                        and df[col].nunique()<=10
                        and df[col].nunique()>1
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
        
        metric_to_show = st.radio('Metric to plot (y axis)', 
                options=['Class Imbalance', 'Jensen-Shannon Divergence'])
        num_rows = st.number_input('Select number of results to show in summary', 
                                min_value=1,
                                max_value=100,
                                value=10,
                                step=1)
        run_comparison = st.button('Run Fairness Assessment')


    with col2:
        with st.expander('Show subgroups (unique values) of selected features to evaluate', expanded=True):
            unique_vals_dict = dict()
            for col in eval_cols:
                unique_vals_dict[col] = str(np.sort(df[col].unique()).tolist())
            st.json(unique_vals_dict)

    if df[target].nunique() > 2:
        run_comparison = False
        st.write('Not implemented: Target variable ' + target + ' has more than 2 unique values, fairness metrics cannot be calculated.')

    if run_comparison:
        st.subheader('Fairness Assessment')

        summary_ctn = st.container()
        summary_df = pd.DataFrame(columns=['Subgroup Pair', 'Class Imbalance', 'Jensen-Shannon Divergence','dataset','column'])

        for col in eval_cols:
            st.markdown('##### ' + col.replace('_', ' '))

            if df[col].nunique() > 10:
                st.text(col + ' has too many subgroups (' + str(nunique) + '). Metrics and comparisons will not be run.')
                continue

            col1, col2 = st.columns([0.5, 0.5])

            with col1:
                container = st.container()
                max_metric_df = pd.DataFrame(columns=['Max Class Imbalance', 'Max Jensen-Shannon Divergence'])
                overall_df=pd.DataFrame(columns=['Subgroup Pair', 'Class Imbalance', 'Jensen-Shannon Divergence','dataset'])
                for df_name in dataset_dict.keys():
                    df_tmp = dataset_dict[df_name]
                    dataset_metric_df = pd.DataFrame(get_metric(target, df_tmp, col))
                    max_metric_df = max_metric_df.append(pd.DataFrame({
                        'Max Class Imbalance': dataset_metric_df['Class Imbalance'].max(),
                        'Max Jensen-Shannon Divergence': dataset_metric_df['Jensen-Shannon Divergence'].max()
                        }, index=[df_name]))

                    with st.expander('Details: ' + df_name, expanded=False):
                        st.caption('Subgroup details')
                        class_details = pd.concat(get_class_summary(target, df_tmp, col), axis=1)
                        class_details.columns = ['N', 'N_label_{}'.format(df_tmp[target].unique()[1])]
                        class_details['Label_probability'] = class_details['N_label_{}'.format(df_tmp[target].unique()[1])]/class_details['N']
                        st.table(class_details)

                        st.caption('Metrics details')
                        dataset_metric_df['dataset']=df_name
                        overall_df = overall_df.append(dataset_metric_df)
                        dataset_metric_df.drop(columns=['dataset'], inplace=True)
                        st.table(dataset_metric_df.set_index('Subgroup Pair'))

                container.table(max_metric_df)

            with col2:
                x = overall_df['Subgroup Pair']
                y = overall_df[metric_to_show]
                fig = px.scatter(x=x,
                                y=y,
                                color=overall_df['dataset'],
                                labels={'x':'Subgroup Pairs', 'y':metric_to_show})
                fig.update_traces(textposition='top right')
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                st.plotly_chart(fig,
                    use_container_width=True,
                    config={'responsive': False, 'displayModeBar': False})

            overall_df['column'] = col
            summary_df = summary_df.append(overall_df[overall_df['dataset']==select_key])

        summary_df.sort_values(by=[metric_to_show], ascending=False, inplace=True)
        summary_ctn.markdown(f'##### {metric_to_show} score summary for {select_key} dataset')
        summary_ctn.table(summary_df[['column','Subgroup Pair',metric_to_show]].iloc[:num_rows].set_index(['column']))


























