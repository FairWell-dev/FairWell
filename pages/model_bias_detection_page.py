import fairlearn.metrics as flm
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
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


def infer_target_col(cols):
    for idx, col in reversed(list(enumerate(cols))):
        if ('predict' not in col) and ('target' in col) or ('label' in col):
            return idx
    return len(cols) - 1 # Assume right most column is target


def render(sidebar_handler):
    # Sidebar
    eg_dict = {
        'Baseline Model': 'data/pred_basemodel_1103.csv',
        'Bias Mitigated - Undersampling Model': 'data/pred_undersampling_1103.csv',
        'Bias Mitigated - Reweighing Model' : 'data/pred_reweighing_1103.csv',
        'Bias Mitigated - FairTorch Model': 'data/pred_fairtorch_1103.csv'
    }

    pred_df_dict, selected = sidebar_handler('Test Dataset(s) and Model(s) for Bias Detection', 
                                                ['csv', 'pt', 'json'], 
                                                eg_dict)
    df = pred_df_dict[selected]
    model_name_list = pred_df_dict.keys()
        
    dtype_dict = infer_dtypes(df)

    # Main
    st.subheader("Bias Detection and Comparison on Trained Model")

    # Config
    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox('Target feature (categorical)', 
                              options=[col for col in df.columns
                                       if (dtype_dict[col] == 'categorical')],
                              index=infer_target_col(
                                        [col for col in df.columns
                                         if (dtype_dict[col] == 'categorical')]))

        prediction = target + '_prediction'

        all_cols_to_eval = [col for col in df.columns \
                                               if (col not in [target, prediction]) and \
                                                   (dtype_dict[col] == 'categorical') and \
                                                   (df[col].nunique() <= 10)]
        container = st.container()
        all = st.checkbox('Select all')
        if all:
            cols_to_eval = container.multiselect('Categorical features to evaluate for fairness \
                                        (features with >10 subgroups will not be shown)', 
                                        options=all_cols_to_eval,
                                        default=all_cols_to_eval)
        elif selected.lower() == 'baseline model':
            cols_to_eval = container.multiselect('Features to evaluate (select one or more)',
                                            all_cols_to_eval,
                                            [col for col in all_cols_to_eval if col.startswith('Privileged')])
        else:
            cols_to_eval = container.multiselect('Categorical features to evaluate for fairness \
                                        (features with >10 subgroups will not be shown)', 
                                        options=all_cols_to_eval)
        x_axis_metric = 'Accuracy'
        y_axis_metric = st.radio('Metric to plot (y-axis)', 
                                 options=['Demographic Parity', 'Equalized Odds', 'Predictive Parity'])

        run_comparison = st.button('Run Fairness Assessment and Model Comparison')

    with col2:
        with st.expander('Show subgroups (unique values) of features to evaluate'):
            unique_vals_dict = dict()
            for col in cols_to_eval:
                unique_vals_dict[col] = str(np.sort(df[col].unique()).tolist())
            st.json(unique_vals_dict)
        with st.expander('How to interpret the scatter plot', expanded=True):
            read_chart_description = """
                The scatter plot represents each model as a point. 
                The x-axis represents accuracy (the higher the better). 
                The y_axis represents disparity (the lower the fairer).
            """
            st.markdown(read_chart_description)

    if df[target].nunique() > 2:
            run_comparison = False
            st.write('Not implemented: Target variable ' + target + ' has more than 2 unique values, fairness metrics cannot be calculated.')

    if run_comparison:
        st.subheader('Overall Model Performance')
        for model_name in model_name_list:
            df = pred_df_dict[model_name]
            with st.expander('Model Performance: ' + model_name):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown('##### Classification Report')
                    report = skm.classification_report(df[target], df[prediction], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.table(report_df)

                with col2:
                    st.markdown('##### Confusion Matrix')
                    confusion_matrix = skm.confusion_matrix(df[target], df[prediction], labels=[0, 1])
                    confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=['0 (Predicted)', '1 (Predicted)'], index=['0 (Actual)', '1 (Actual)'])
                    colour_map = sns.light_palette('lightblue', as_cmap=True)
                    st.dataframe(confusion_matrix_df.style.background_gradient(cmap=colour_map), width=500)

        st.subheader('Fairness Assessment and Model Comparison (per Feature)')

        combined_df = pd.DataFrame()
        for col_name in cols_to_eval:
            st.markdown('##### Feature: ' + col_name.replace('_', ' '))
            col1, col2 = st.columns([0.6, 0.4])

            with col1:
                acc_l, dpd_l, eod_l, ppd_l = [], [], [], []
                for model_name in model_name_list:
                    df = pred_df_dict[model_name]
                    
                    # Calculate overall scores
                    acc = skm.accuracy_score(df[target], df[prediction])
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
                    acc_l.append(acc)
                    dpd_l.append(dpd)
                    eod_l.append(eod)
                    ppd_l.append(ppd)
                overall_df = pd.DataFrame({
                    'Accuracy': acc_l,
                    'Demographic Parity': dpd_l,
                    'Equalized Odds': eod_l,
                    'Predictive Parity': ppd_l
                }, index=model_name_list)
                st.table(overall_df)

                # Add to combined df
                overall_df['Feature'] = col_name.replace('_', ' ')
                combined_df = pd.concat([combined_df, overall_df])

                for model_name in model_name_list:
                    df = pred_df_dict[model_name]
                    with st.expander('Details: ' + model_name, expanded=True):
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
                x = overall_df[x_axis_metric]
                y = overall_df[y_axis_metric]
                fig = px.scatter(x=x, 
                                 y=y, 
                                 text=overall_df.index,
                                 labels={'x': 'Accuracy', 'y': y_axis_metric})
                fig.update_traces(textposition='top right')
                fig.update_xaxes(range=[max([0, x.min() - 0.1]), 
                                        min([1, x.max() + 0.1])])
                fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig,
                            use_container_width=True,
                            config={'responsive': False, 'displayModeBar': False})

            with st.expander('Insights'):
                insights_description = """
                    Accuracy ranges from {}% to {}%, while disparity ranges from {}% to {}%. \n
                    The model with the highest accuracy ({}%) has a disparity of {}%. \n
                    The least disparate model achieves an accuracy of {}% with a disparity of {}%.
                """

                losses = overall_df[x_axis_metric]
                min_loss = '{:.2f}'.format(min(losses))
                max_loss = '{:.2f}'.format(max(losses))

                disparities = overall_df[y_axis_metric]
                min_disparity = '{:.2f}'.format(min(disparities))
                max_disparity = '{:.2f}'.format(max(disparities))

                metric_argmax = np.argmax(losses)
                highest_acc = '{:.2f}'.format(losses[metric_argmax])
                highest_acc_disparity = '{:.2f}'.format(disparities[metric_argmax])

                disparity_argmin = np.argmin(disparities)
                lowest_disparity = '{:.2f}'.format(disparities[disparity_argmin])
                lowest_disparity_acc = '{:.2f}'.format(losses[disparity_argmin])
                
                st.markdown(insights_description.format(min_loss, max_loss, min_disparity, max_disparity,
                                                     highest_acc, highest_acc_disparity,
                                                     lowest_disparity_acc, lowest_disparity))

        
        st.subheader(y_axis_metric.title() + ' across Models')

        with st.expander('Explanation'):
            st.markdown("""
                    The table shows the diaparity scores across features, sorted in descending order. 
                    Features at the top of the list should be prioritised for mitigation - recommendations for mitigation techniques \
                    can be viewed after this section.
                """.format())

        combined_df = combined_df.reset_index()
        combined_df = combined_df.drop('Accuracy', axis=1)
        combined_df = combined_df.rename({'index': 'Model'}, axis=1)

        for model_name in model_name_list:
            st.markdown('##### Model: ' + model_name)
            combined_df_filtered = combined_df.loc[combined_df['Model'] == model_name]
            combined_df_filtered = combined_df_filtered.drop('Model', axis=1)
            combined_df_filtered = combined_df_filtered.sort_values(y_axis_metric, ascending=False)

            col1, col2 = st.columns([0.6, 0.4])
            with col1:
                # colour_map = sns.light_palette('lightblue', as_cmap=True)
                # st.table(combined_df_filtered.set_index('Feature').style.background_gradient(cmap=colour_map, 
                #                                                                              subset=y_axis_metric))
                st.table(combined_df_filtered.set_index('Feature').style.bar(color='#4267B2', 
                                                                                subset=y_axis_metric))

            with col2:
                combined_df_filtered = combined_df_filtered.iloc[:10]
                fig, ax = plt.subplots()
                sns.barplot(x='Feature', y=y_axis_metric, data=combined_df_filtered, color='#4267B2')
                plt.xticks(rotation=90)
                plt.title('Top 10 Feature Disparity: ' + model_name)
                st.pyplot(fig)

        st.subheader("Mitigation Recommendations: Bias Mitigation Techniques")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('##### Pre-Processing')
            st.markdown("""
                - Suppression: Reduce discrimination between target and sensitive feature _S_ by removing _S_ \
                and features highly correlated to _S_
                    - Remove sensitive features and their correlated features with \
                    [Fairlearn](https://fairlearn.org/v0.7.0/api_reference/fairlearn.preprocessing.html)
                - Re-weighing: Add a weights column to the dataset for each (feature group, label) combination \
                to ensure dataset is fair before training - observations with higher weights should be prioritised \
                ("seen more") by the model during training
                    - [AI Fairness 360](https://aif360.readthedocs.io) \
                    provides an [API for generating weights](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.Reweighing.html#aif360.algorithms.preprocessing.Reweighing)
                - Sampling: Calculate sample sizes for each (sensitive feature, label) combination to reduce \
                discrimination - oversample the minority groups and undersample the majority groups using Uniform \
                Sampling and/or Preferential Sampling techniques  
                    - Generate synthetic data with Generative Adversarial Networks (GANs) \
                    [TorchGAN](https://torchgan.readthedocs.io/en/latest/index.html), \
                    [Mimicry] (https://github.com/kwotsin/mimicry), \
                    [TF-GAN] (https://github.com/tensorflow/gan)
                    - Generate synthetic data with \
                    [Variational Autoencoder (VAE) with PyTorch](https://visualstudiomagazine.com/Articles/2021/05/06/variational-autoencoder.aspx)
                    - SMOTE (Synthetic Minority Over-sampling Technique) using [imbalanced-learn](https://imbalanced-learn.org/stable/index.html)
            """.format())

        with col2:
            st.markdown('##### In-Processing')
            st.markdown("""
                - Fairness loss metrics: Incorporate custom loss functions when training the model, making them a constraint
                    - [FairTorch](https://github.com/wbawakate/fairtorch) applies group fairness \
                    (demographic parity and equalized odds) as losses for model training in PyTorch
                - Discrimination-aware Regularization: Adding discrimination-aware regularization term to objective function
                    - [AI Fairness 360](https://aif360.readthedocs.io) \
                    provides an [API for adding regularization the model](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.PrejudiceRemover.html)
            """.format())

        with col3:
            st.markdown('##### Post-Processing')
            st.markdown("""
                - Calibrated predictions: Adjust model predictions to achieve equalized odds objective
                    - [AI Fairness 360](https://aif360.readthedocs.io) \
                    provides an [API for calibrating scores](https://aif360.readthedocs.io/en/v0.2.3/modules/postprocessing.html#aif360.algorithms.postprocessing.CalibratedEqOddsPostprocessing)
                - Per-subgroup threshold: Optimise thresholds for each subgroup to achieve equalized odds
                    - [AI Fairness 360](https://aif360.readthedocs.io) \
                    provides an [API for finding these probabilities](https://aif360.readthedocs.io/en/v0.2.3/modules/postprocessing.html#equality-of-odds)
            """.format())
