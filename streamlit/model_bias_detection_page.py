import numpy as np
import pandas as pd
import plotly.express as px
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
def get_losses(metric):
    # TODO: Implement actual loss calculations
    losses = [0.84, 0.834, 0.827, 0.793, 0.783, 0.772]
    return losses if metric == 'accuracy' else losses[::-1]

@st.cache
def get_disparities(metric):
    # TODO: Implement actual diaparities calculations
    return [0.16, 0.119, 0.066, 0.022, 0.003, 0.01]

def render(sidebar_handler):
    # Sidebar
    all_data_models, selected = sidebar_handler('Training Dataset(s) and Model(s) for Bias Detection', 
                                                ['csv', 'pth'], 
                                                ['data/final.csv', 'data/final_copy.csv']) # sample files will be csv with predicted labels
    
    # Unpack values
    if isinstance(selected, tuple):
        df, model = selected
        df_list, model_list = all_data_models
    else:
        df = selected
        df_list = all_data_models

    # Main
    st.subheader("Bias Detection on Trained Model")

    # Config
    col1, col2 = st.columns([0.4, 0.6])

    with col1:
        target = st.selectbox('Target feature', options=df.columns)
        dtype_dict = infer_dtypes(df)
        evaluate = st.selectbox('Features to evaluate for fairness', 
                                options=[col for col in df.columns 
                                         if (target != col) and (dtype_dict[col] == 'categorical')])
        run_comparison = st.button('Run Model Comparison')

    with col2:
        st.subheader('Subgroups (unique values)')
        st.json(df[evaluate].unique().tolist())

    if (dtype_dict[target] == 'categorical') and df[target].nunique() > 2:
        run_comparison = False
        st.write('Target variable ' + target + ' has more than 2 unique values and model comparison cannot be run.')

    if run_comparison:
        metric = 'accuracy' if dtype_dict[target] == 'categorical' else 'loss'
        col1, col2 = st.columns([0.7, 0.3])

        with col1:
            st.subheader('Model Comparison: Disparity in Predictions')

            xs = df.copy().drop(target, axis=1)
            y = df[target]

            # TODO: Run models to get predictions, and calculate disparity and loss
            # xs_list = [xs]
            # y_list = [y]
            # models = []
            # losses = calculate_losses(xs_list, y_list, models, metric)
            # disparities = calculate_disparities(xs_list, y_list, models, metric)

            losses = get_losses(metric)
            disparities = get_disparities(metric)

            fig = px.scatter(x=losses, y=disparities, labels={'x': metric, 'y': 'disparity'})
            fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig,
                           use_container_width=True,
                           config={'responsive': False, 'displayModeBar': False})

        with col2:
            st.subheader('How to read this chart')
            objective = 'high' if metric == 'accuracy' else 'low'

            read_chart_description = """
                This chart represents each model as a point.\n
                The x-axis represents {} (the {}er the better).\n
                The y_axis represents disparity (the lower the fairer).
            """
            st.write(read_chart_description.format(metric, objective))

            st.subheader('Insights')
            insights_description = """
                {} ranges from {}% to {}%, while disparity ranges from {}% to {}%.

                The model with the {} {} achieves an {} of {}% with a disparity of {}%.

                The least-disparity model achieves an {} of {}% with a disparity of {}%.
            """

            metric_argmax = np.argmax(losses)
            disparity_argmax = np.argmax(disparities)
            st.write(insights_description.format(metric.title(), min(losses), max(losses), min(disparities), max(disparities),
                                                 objective + 'est', metric, metric, losses[metric_argmax], disparities[disparity_argmax],
                                                 metric, losses[metric_argmax], disparities[disparity_argmax]))


    st.subheader("Recommended Bias Mitigation Techniques")
