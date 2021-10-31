# FairWell

## Introduction

FairWell is a Responsible AI tool developed using Streamlit. In this report, research on Responsible AI is presented, followed by elaborating on the FairWell tool. Lastly, a use case on how Responsible AI can be used to evaluate a model is shown, where the New York City (NYC) Subway dataset is used to model a binary classification problem on high/low traffic.

## Responsible AI Research

The problem of defining and addressing fairness has been a topic of increasing importance in the recent years. Especially with work surrounding machine learning. Multiple definitions of fairness have been raised, each having their own pros and cons. This work does not aim to propose a single notion of fairness but rather we aim to (i) provide tools on how to measure and assess fairness and (ii) how to mitigate bias in models where necessary. Our work can be summarised as below:

1. Fairness assesment<sup>1, 12, 13</sup>
   1. On data
   2. On model predictions
2. Bias mitigation<sup>12</sup>
   1. Pre-processing by transforming the data 
   2. In-processing by imposing constraints during training
   3. Post-processing where the predictions of models are modified

### Fairness Metrics
The first step to mitigating bias is measuring. In this work, we look at identifying the presence of bias in 2 locations: (i) bias in the data and (ii) bias in the predictions.

#### Fairness Metrics (Data)

The quality of data used will heavily influence the performance and fairness of the model. Therefore, identifying bias in the training data is an important step in weeding out potential biases. However, this assumes that historical bias and discrimination are not the root cause of bias in the data. Nonetheless, several model-independent metrics can be used to inform the user about the presence of bias and efforts can be made to reduce skewed data distributions. Examples include but not limited to class imbalance, Jensen-Shannon divergence, etc. <sup>1</sup>

#### Fairness Metrics (Predictions)

More commonly, we identify bias based on decisions (i.e. predicitions) made by the model after training (and bias mitigation). Generally, fairness decision definitions can be categorised into 4 areas, (i) individual, (ii) group, (iii) per group performance and (iv) causality based criteria.<sup>13</sup>

- **Individual Fairness.** Fairness definitions at the individual level focuses on the relative similarity between individuals. Therefore, similar individuals should be treated similarly with similar decisions.<sup>12, 13</sup>

- **Group Fairness.** Group fairness as the term suggests, focuses on reducing bias for a group of individuals. It is believed that different groups of people are being unfairly treated and thus aims to attain fairness for each respective group.<sup>13</sup> Some popular definitions of group fairness include but not limited to demographic parity, Equal oportunity, predictive parity, etc.<sup>14</sup>

- **Per Group Performance Fairness.** Another school of thought for fairness is the idea of per group performance. This school of thought attempts to maximise the utility of an individual group to attain fairness.<sup>13</sup> Examples include Pareto-fairness<sup>15</sup> and Rawlsian Max-Min fairness<sup>16</sup>.

- **Causality Based Citeria.** These notions of fairness is distinct from the previous 3 definitions that are based on observational results. Causality based citeria, attempts to create connections amongst related variables to derive causal relationship of the problem. The relationship obtain can then be used to answer counter factual questions such as “what would have been the decision if that individual had a different gender?”.<sup>13</sup>

### Responsible AI in Businesses

Businesses have recognised the need to develop AI models that are responsible and fair towards their data inputs. 

Google, who adopted an AI-first strategy, is working on fair Machine Learning systems. They created a Fairness Indicators package to calculate metrics, and integrated them into their What-If Tool for data scientists to assess their datasets, and visualise and present their model performance. These tools are also integrated with TensorFlow Extended that helps with TensorFlow model evaluation.

Amazon Web Services has a feature, Clarify, in AWS SageMaker that aims to not only help calculate fairness metrics, it raises an alert when there is feature attribution drift. A feature attribution drift occurs when the distribution of a feature in the model during training is vastly different from the distribution of a feature in the model when it is deployed to production. 

Facebook, too, understands the importance of responsible AI as it can help mitigate concerns surrounding privacy, fairness, accountability and transparency in its algorithms. They set up a cross-disciplinary Responsible AI (RAI) team to ensure Machine Learning systems are designed and used responsibly. The team has since released Fairness Flow, an internal tool to assess AI fairness.

> "Fairness Flow lists four definitions (of fairness) that engineers can use according to which suits their purpose best, such as whether a speech-recognition model recognizes all accents with equal accuracy or with a minimum threshold of accuracy."

As Facebook progresses in building Responsible AI, they are guided by these key pillars:
- Privacy & Security
- Fairness & Inclusion
- Robustness & Safety
- Transparency & Control
- Accountability & Governance

## FairWell

### Tools and Technologies Used

- <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="20"/> [Streamlit](https://streamlit.io/)
- [Microsoft Fairlearn](https://fairlearn.org/)
- [AI Fairness 360](https://aif360.mybluemix.net/)
- Data Science packages (NumPy, Pandas, Plotly)

### Features Explorer

The features explorer page allows users to quickly view the distributions of their dataset, as well as the summary statistics for each feature. An algorithm is created to quickly infer the data types of each feature, categorising them into either numerical or categorical. Based on the data type, a histogram or bar chart will be shown for numerical or categorical data type respectively.

### Fairness Assessment (Data)

### Fairness Assessment (Model Predictions) and Mitigation Recommendations

With a model, users can easily assess the fairness of their model's predictions with regard to the input dataset. The aim is to allow for identifying bias from the model after it is trained. Coupled with the previous page on fairness assessment on the data, users can identify if the source of bias comes from the data or model, or both.

Users will first select the target feature from the dataset, along with the features they would like to evaluate for fairness. The inference process will then kick off for every model, returning predictions as outputs. With both the ground truth (target) and the predictions, the following fairness metrics will be calculated:

- Demographic Parity (DP): Measures the same positive prediction ratio across groups identified by the sensitive features.
- Equalized Odds (EO): Measures the equality in terms of error rate. Same false positive rate and false negative rate across sensitive groups identified.
- Predictive Parity (PP): Measures the error rates in terms of fraction of errors over the ground truth. The model should have the same precision across sensitive groups.

A scatter plot that plots the relationship between the selected fairness metric and each model's performance will be shown. This is coupled with an expandable insights section, allowing users to evaluate the potential trade-offs from their models and fairness.

Lastly, the page will compare the aforementioned fairness metrics of each feature selected for fairness assessment, providing users with useful mitigation approaches they can take towards fairer model development. After applying fairness mitigation, users can revisit this page with a new dataset and model for comparison against their previous iterations.

## Dataset: New York City (NYC) Subway Traffic

The dataset we selected consists of subway traffic in NYC, along with neighborhood census data of the city. It is hosted on Kaggle by Edden, who has performed preprocessing steps to convert the raw data provided by The Metropolitan Transportation Authority (MTA), North America's largest transportation network. The census data is from NYU Furman Center's New York City Neighborhood Data Profiles and the neighborhood data is from University of Berkeley GeoData Library.

### Problem Statement

Public transport has become a necessity in our modern landscape. Agencies are interested in capitilising on data on public transport usage such as subway traffic to inform their location based business decisions.

Government entities involved in urban planning might utilise subway traffic conditions to determine neighbourhoods that could benefit from neighbourhood rejuvenation or to inform other land usage planning decisions.<sup>8</sup> Governments can also benefit from having a gauge of how investing into a neighbourhood will affect traffic volume via subway in different areas through census data.

Alternatively, businesses such as the real estate and media industries can benefit from integrating subway traffic conditions as their decision making factors. Subway traffic can greatly affect real estate prices, thus this can inform real estate developers in their development strategies<sup>9</sup>. With 1.7 billion turnstile swipe in 2019 alone, the subway is New York City's most popular mode of transit<sup>10</sup>, priming it to an effective mode of advertisement - a reason why subway advertising has become a regular part of every commuter's life. Subway traffic can also be used to inform media agencies of their audience, allowing them to identify prime locations - and thereby the corresponding bullet services - for their advertisement campaigns in order to maximise their effectiveness<sup>11</sup>.

### Approach

The baseline model is built for a time series binary classification problem to predict whether subway traffic is high or low.

### Pipeline Overview

![](./images/Pipeline%20Overview.png)

### Data preprocessing

#### Neighborhood Census Data

The neighborhood census data of the city consisted of a total of 87 columns, out of which we selected 15 columns to be included in the final dataset. Of these 15 columns, none contained missing data points.

* Neighborhood
* Car-free commute (% of commuters)
* Disabled population
* Foreign-born population
* Median household income (2018$)
* Median rent, all (2018$)
* Moderately rent-burdened households
* Percent Asian
* Percent Hispanic
* Percent black
* Percent white
* Population
* Poverty rate
* Public housing (% of rental units)
* Unemployment rate

#### Subway Data

As the neighborhood census data was taken on June 2020, we selected 3 months worth of data from the subway data between April to June 2020, with the assumption that the neighborhood census remain consistent for these 3 months.

Missing values: The Subway Dataset has the following columns with missing data. Upon further evaluation of these columns, we decided to drop these two columns.

| Column                  | Missing (%)     |
| ----------------------- | --------------- |
| North Direction Label	  | 3.19            |
| South Direction Label	  | 2.58            |


One Hot Encoding: One Hot Encoding was done for the following categorical columns: - "Division", "Structure", "Borough", "Connecting Lines", "Daytime Route". Upon performing One Hot Encoding on these, columns, we conducted a correlation analysis to check for high correlated columns that may result in the Curse of Dimensionality. We discovered that there was a high correlation between the "Connecting Lines" column and the "Daytime Route" column. As such, we dropped the "Daytime Route" column. 

9 columns were then selected to be in the final dataset.
* Datetime
* Stop Name
* Connecting Lines
* Division
* Structure
* Borough
* neighborhood
* EntriesExit (Binary Target)

Target Variable: The current subway dataset has no clear target output. Hence, we want to aggregate the current "Entry" and "Exit" columns such that we can get a target binary column (EntriesExit) to represent the total amount of traffic for every station. We combine the current "Entry" and "Exit" columns by summing them up to aggregate the total number of people passing through the station in each 4 hr interval. We then find the median number of people passing through all the neighborhoods in each Datetime period. If EntriesExit value is greater than the median, we classify that it is crowded (1) and vice versa (0) to derive a binary target. We chose to split the column by median to ensure that our target would have a uniform (balanced) distribution.

### Feature Engineering

Feature Engineering was done on two columns: "Number of Stations" and "Neighborhood Area Size". "Number of Stations" is derived by grouping the subway data by neighborhood to find the number of unique stations in each neighborhood. "neighborhood Area Size" is derived from the original neighborhood census data columns: "Population" and "Population Density (1000 persons per square mile)".

Lastly, we merged the two datasets together. Our final dataset contains the following features:

| Features from Neighborhood Census Data  | Features from Subway Data\* |
| --------------------------------------- | -------------------------- |
| Neighborhood<br>Car-free commute (% of commuters)\*\*\*<br>Disabled population<br>Foreign-born population<br>Median household income (2018$)<br>Median rent, all (2018$)<br>Percent Asian\*\*\*<br>Percent Hispanic\*\*\*<br>Percent Black\*\*\*<br>Percent white\*\*\*<br>Population\*\*\*\*<br>Poverty rate<br>Public housing (% of rental units)\*\*\*<br>Unemployment rate<br>Residential units within 12 mile of a subway station<br>Population density (1,000 persons per square mile)\*\*\*\*<br>Serious crime rate (per 1,000 residents)<br>Severely rent-burdened households<br>Rental vacancy rate<br>Mean travel time to work (minutes)| Datetime<br>Stop Name<br>Connecting Lines\*\*<br>Division\*\*<br>Structure\*\*<br>Borough\*\*<br>Neighborhood<br>Entries\*\*\*\*\*<br>Exits\*\*\*\*\* |

\* Used to derive "Number of Stations" feature \
\*\* One hot encoded features \
\*\*\* Binned features \
\*\*\*\* Used to derive ""Neighborhood Area Size"" feature \
\*\*\*\*\* Used to derive "EntriesExits" target feature

### Fairness Assessment on Dataset


### Modelling with PyTorch


### Fairness Assessment on Model Predictions


### Fairness Mitigation


### Effect of Mitigation Approach


## Conclusion and Future Work

## References

1. [Amazon AI Fairness and Explainability Whitepaper](https://pages.awscloud.com/rs/112-TZM-766/images/Amazon.AI.Fairness.and.Explainability.Whitepaper.pdf)
2. [Facebook’s five pillars of Responsible AI](https://ai.facebook.com/blog/facebooks-five-pillars-of-responsible-ai/)
3. [Fairness Indicators](https://github.com/tensorflow/fairness-indicators)
4. [Machine Learning Glossary](https://developers.google.com/machine-learning/glossary)
5. [How Facebook got addicted to spreading misinformation](https://www.technologyreview.com/2021/03/11/1020600/facebook-responsible-ai-misinformation/)
6. [Normal and New Normal NYC Subway Traffic 2017-21](https://www.kaggle.com/eddeng/nyc-subway-traffic-data-20172021)
7. [What-If Tool](https://pair-code.github.io/what-if-tool/)
8. [Integration of public transport and urban planning](https://www.researchgate.net/publication/-228716874_Integration_of_public_transport_and_urban_planning)
9. [The Impact of Subway Lines on Residential Property Values in Tianjin: An Empirical Study Based on Hedonic Pricing Model](https://www.hindawi.com/journals/ddns/2016/1478413/)
10. [How Coronavirus has changed New York City Transit, in one chart](https://www.nytimes.com/interactive/2021/03/08/climate/nyc-transit-covid.html) 
11. [Optimization of Subway Advertising Based on Neural Networks](https://www.hindawi.com/journals/mpe/2020/1871423/)
12. [The zoo of Fairness metrics in Machine Learning](https://arxiv.org/pdf/2106.00467.pdf)
13. [Fairness without Demographics through Adversarially Reweighted Learning](https://arxiv.org/pdf/2006.13114.pdf)
14. [Fairness and Machine Learning](http://www.fairmlbook.org)
15. [Pareto-Efficient Fairness for Skewed Subgroup Data](https://aiforsocialgood.github.io/icml2019/accepted/track1/pdfs/24_aisg_icml2019.pdf)
16. [The Price of Fairness](https://core.ac.uk/download/pdf/4429576.pdf)
## Acknowledgements

Thanks to our friends at Mastercard (Apurva, Bharathi, Hui Chiang, Idaly and Louis) for their advice and guidance on AI fairness.
