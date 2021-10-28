import fairlearn.metrics as flm
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn.metrics as skm
import streamlit as st

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
def get_losses(metric):
    # TODO: Implement actual loss calculations
    losses = [0.84, 0.834, 0.827, 0.793, 0.783, 0.772]
    return losses if metric == 'accuracy' else losses[::-1]

@st.cache
def get_disparities(metric):
    # TODO: Implement actual diaparities calculations
    return [0.16, 0.119, 0.066, 0.022, 0.003, 0.01]

def predictive_parity(y_true, y_pred):
    if y_pred.sum() == 0:
        return np.nan
    return y_true.sum() / y_pred.sum()

def predictive_parity_difference(y_true, y_pred, sensitive_feature):
    pps = list()
    
    for val in sensitive_feature.unique():
        idxs = sensitive_feature.values == val
        y_true_subgroup = y_true.values[idxs]
        y_pred_subgroup = y_pred.values[idxs]
        
        pp = predictive_parity(y_true_subgroup, y_pred_subgroup)
        pps.append(pp)
        
    return max(pps) - min(pps)

def render(sidebar_handler):
    # Sidebar
    all_data_models, selected = sidebar_handler('Training Dataset(s) and Model(s) for Bias Detection', 
                                                ['csv', 'pth'], 
                                                ['../data/final_predictions.csv']) # sample files will be csv with predicted labels
    
    # Unpack values
    if isinstance(selected, tuple):
        df, model = selected
        df_list, model_list = all_data_models
    else:
        df = selected
        df_list = all_data_models

    dtype_dict = infer_dtypes(df)

    # Main
    st.subheader("Bias Detection on Trained Model")

    # Config
    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox('Target feature (categorical)', options=[col for col in df.columns
                                                         if (dtype_dict[col] == 'categorical')])

        prediction = st.selectbox('Predictions', options=[col for col in df.columns
                                                         if (target != col) and (dtype_dict[col] == 'categorical')])

    with col2:
        cols_to_eval = st.multiselect('Categorical features to evaluate for fairness', 
                                  options=[col for col in df.columns 
                                  if (col not in [target, prediction]) and (dtype_dict[col] == 'categorical')])

        y_axis_metric = st.radio('Metric to plot (y-axis)', 
                                 options=['Demographic Parity', 'Equalized Odds', 'Predictive Parity'])

    run_comparison = st.button('Run Fairness Assessment and Model Comparison')
    
    with st.expander('Show subgroups (unique values) of features to evaluate'):
        unique_vals_dict = dict()
        for col in cols_to_eval:
            unique_vals_dict[col] = str(np.sort(df[col].unique()).tolist())
        st.json(unique_vals_dict)

    if df[target].nunique() > 2:
            run_comparison = False
            st.write('Not implemented: Target variable ' + target + ' has more than 2 unique values, fairness metrics cannot be calculated.')

    if run_comparison:
        st.subheader('Fairness Assessment and Model Comparison')

        for col_name in cols_to_eval:
            st.markdown('##### ' + col_name)

            nunique = df[col_name].nunique()
            if nunique > 10:
                st.text(col_name + ' has too many subgroups (' + str(nunique) + '). Metrics and comparisons will not be run.')
                continue

            col1, col2 = st.columns([0.6, 0.4])

            with col1:
                # Calculate overall scores
                dpd = flm.demographic_parity_difference(y_true=df[target],
                                                        y_pred=df[prediction], 
                                                        sensitive_features=df[col_name],
                                                        method='between_groups')
                eod = flm.equalized_odds_difference(y_true=df[target],
                                                    y_pred=df[prediction], 
                                                    sensitive_features=df[col_name],
                                                    method='between_groups')
                ppd = predictive_parity_difference(y_true=df[target],
                                                   y_pred=df[prediction], 
                                                   sensitive_feature=df[col_name])

                overall_df = pd.DataFrame({
                    'Demographic Parity Difference': [dpd],
                    'Equalised Odds Difference': [eod],
                    'Predictive Parrity Difference': [ppd]
                }, index=['Overall'])
                st.table(overall_df)

                grouped_metric = flm.MetricFrame(metrics={'N': flm.count,
                                            'Accuracy': skm.accuracy_score,
                                            'Selection Rate': flm.selection_rate,
                                            'True Positive Rate': flm.true_positive_rate,
                                            'False Positive Rate': flm.false_positive_rate,
                                            'Predictive Parity': predictive_parity
                                            },
                                    y_true=df[target],
                                    y_pred=df[prediction],
                                    sensitive_features=df[col_name])

                results = grouped_metric.by_group
                results.index.name = 'Subgroup'
                st.table(results)

            with col2:
                # TODO: Plot chart with actual values
                losses = get_losses('accuracy')
                disparities = get_disparities('accuracy')

                fig = px.scatter(x=losses, y=disparities, labels={'x': 'accuracy', 'y': 'disparity'})
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig,
                            use_container_width=True,
                            config={'responsive': False, 'displayModeBar': False})

        # metric = 'accuracy' if dtype_dict[target] == 'categorical' else 'loss'
        # col1, col2 = st.columns([0.7, 0.3])

        # with col1:
        #     st.subheader('Model Comparison: Disparity in Predictions')

        #     xs = df.copy().drop(target, axis=1)
        #     y = df[target]

        #     # TODO: Run models to get predictions, and calculate disparity and loss
        #     # xs_list = [xs]
        #     # y_list = [y]
        #     # models = []
        #     # losses = calculate_losses(xs_list, y_list, models, metric)
        #     # disparities = calculate_disparities(xs_list, y_list, models, metric)

        #     losses = get_losses(metric)
        #     disparities = get_disparities(metric)

        #     fig = px.scatter(x=losses, y=disparities, labels={'x': metric, 'y': 'disparity'})
        #     fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        #     st.plotly_chart(fig,
        #                    use_container_width=True,
        #                    config={'responsive': False, 'displayModeBar': False})

        # with col2:
        #     st.subheader('How to read this chart')
        #     objective = 'high' if metric == 'accuracy' else 'low'

        #     read_chart_description = """
        #         This chart represents each model as a point.\n
        #         The x-axis represents {} (the {}er the better).\n
        #         The y_axis represents disparity (the lower the fairer).
        #     """
        #     st.write(read_chart_description.format(metric, objective))

        #     st.subheader('Insights')
        #     insights_description = """
        #         {} ranges from {}% to {}%, while disparity ranges from {}% to {}%.

        #         The model with the {} {} achieves an {} of {}% with a disparity of {}%.

        #         The least-disparity model achieves an {} of {}% with a disparity of {}%.
        #     """

        #     metric_argmax = np.argmax(losses)
        #     disparity_argmax = np.argmax(disparities)
        #     st.write(insights_description.format(metric.title(), min(losses), max(losses), min(disparities), max(disparities),
        #                                          objective + 'est', metric, metric, losses[metric_argmax], disparities[disparity_argmax],
        #                                          metric, losses[metric_argmax], disparities[disparity_argmax]))


    st.subheader("Recommended Bias Mitigation Techniques")
