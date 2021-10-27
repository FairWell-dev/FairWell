# uwu

## Research on Responsible AI

### Reponsible AI Reseach in Businesses

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

### Fairness Metrics
The problem of defining and addressing fairness has been a topic of increasing importance in the recent years, especially with work surrounding machine learning. Multiple definitions of fairness have been raised, each having their own pros and cons. This work does not aim to propose a single notion of fairness but rather we aim to (i) provide tools on how to measure and assess fairness and (ii) how to mitigate bias in models where necessary. Furthermore with respect to (ii), bias removing techniques can be broadly catergorised into pre-processing on data, in-processing by imposing constraints during training and post-processing where the results of the models are modified. (https://arxiv.org/pdf/2106.00467.pdf) This work takes the approach of the in-processing technique. 

Generally, fairness definitions can be categorised into 4 areas, (i) individual, (ii) group, (iii) per group performance (https://arxiv.org/pdf/2006.13114.pdf).

**Individual Fairness.** Fairness definitions at the individual level focuses on the relative similarity between individuals. Therefore, similar individuals should be treated similarly with similar decisions. (https://arxiv.org/pdf/2006.13114.pdf)

**Group Fairness.** Group fairness as the term suggests, focuses on reducing bias for a group of individuals. It is believed that different groups of people are being unfairly treated and thus aims to attain fairness for each respective group. Some popular definitions of group fairness include but not limited to Demographic Parity, Equal Oportunity, etc. (https://arxiv.org/pdf/2006.13114.pdf)

**Per Group Performance Fairness.** Another school of thought for fairness is the idea of per group performance. This school of thought attempts to maximise the utility of an individual group to attain fairness. Examples include Pareto-fairness and Rawlsian Max-Min fairness. (https://arxiv.org/pdf/2006.13114.pdf)

## Dataset: New York City (NYC) Subway Traffic

The dataset we selected consists of subway traffic in NYC, along with neighbourhood census data of the city. It is hosted on Kaggle by Edden, who has performed preprocessing steps to convert the raw data provided by The Metropolitan Transportation Authority (MTA), North America's largest transportation network. The census data is from NYU Furman Center's New York City Neighborhood Data Profiles and the neighborhood data is from University of Berkeley GeoData Library.

### Problem Statement

Public transport has become a necessity in our modern landscape. Agencies are interested in capitilising on data on public transport usage such as subway traffic to inform their location based business decisions.

Government entities involved in urban planning might utilise subway traffic conditions to determine neighbourhoods that could benefit from neighbourhood rejuvenation or to inform other land usage planning decisions.[^1] Governments can also benefit from having a gauge of how investing into a neighbourhood will affect traffic volume via subway in different areas through census data.

Alternatively, businesses such as the real estate and media industries can benefit from integrating subway traffic conditions as their decision making factors. Subway traffic can greatly affect real estate prices, thus this can inform real estate developers in their development strategies[^2]. With 1.7 billion turnstile swipe in 2019 alone, the subway is New York City's most popular mode of transit[^3], priming it to an effective mode of advertisement - a reason why subway advertising has become a regular part of every commuter's life. Subway traffic can also be used to inform media agencies of their audience, allowing them to identify prime locations - and thereby the corresponding bullet services - for their advertisement campaigns in order to maximise their effectiveness[^4].

### Approach

The baseline model is built for a time series binary classification problem to predict whether subway traffic is high or low.

### Data preprocessing

#### Neighbourhood Census Data

The neighbourhood census data of the city consisted of a total of 87 columns, out of which we selected 15 columns to be included in the final dataset. Of these 15 columns, none contained missing data points.

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

As the neighbourhood census data was taken on June 2020, we selected 3 months worth of data from the subway data between Apr-Jun 2020, with the assumption that the neighbourhood census data stays constant for the 3 months. 

**Missing Values**

The Subway Dataset has the following columns with missing data. Upon further evaluation of these columns, we decided to drop these two columns.
| Column Header  | Percentage Missing |
| ------------- | ------------- |
| North Direction Label	  | 3.198842  |
| South Direction Label	  | 2.585992  |


**One Hot Encoding**

One Hot Encoding was done for the following categorical columns: - "Division", "Structure", "Borough", "Connecting Lines", "Daytime Route". Upon performing One Hot Encoding on these, columns, we conducted a correlation analysis to check for high correlated columns that may result in the Curse of Dimensionality. We discovered that there was a high correlation between the "Connecting Lines" column and the "Daytime Route" column. As such, we dropped the "Daytime Route" column. 

**Binary Target Variable**

The current subway dataset has no clear target output. Hence, we want to aggregate the current 'Entry' and 'Exit' columns such that we can get a target binary column (EntriesExit) for our model. We combine the current 'Entry' and 'Exit' columns by summing them up to aggregate the total number of people passing through the station in each 4 hr interval. We then find the median number of people passing through all the neighbourhoods in each Datetime period. If EntriesExit value is greater than the median, we classify that it is crowded (1) and vice versa to get a target binary column. We chose to split the column by median to ensure that our target binary column would have a more uniform distribution and prevent imbalanced data.

9 columns were then selected to be in the final dataset.
* Datetime
* Stop Name
* Connecting Lines
* Division
* Structure
* Borough
* Neighbourhood
* EntriesExit (Target Binary)

### Feature Engineering

Feature Engineering was done on 2 columns: num_of_stations and Neighbourhood Area size. Num_of_stations was derived by grouping the subway data by neighbourhood to find the number of unique stations in each neighbourhood. Neighbourhood Area Size was derived from the original neighbourhood census data columns: Population and Population density (1000 persons per square mile).

Lastly, we merged the two datasets together. Our final dataset contains the following features:

| Features  | Features  |
| ------------- | ------------- |
| Neighbourhood	  | Percent Asian	  |
| Datetime	  | Percent Hispanic	  | 
| **EntriesExit**	  | Percent black	  | 
| Division	  | Percent white	  | 
| Structure	  | Population	  | 
| Borough	  | Poverty rate	  | 
| Car-free commute (% of commuters)	  | Public housing (% of rental units)	  | 
| Disabled population		  | Unemployment rate	  | 
| Foreign-born population	  | Residential units within 12 mile of a subway station		  | 
| Median household income (2018$)	  | Population density (1,000 persons per square mile)		  | 
| Median rent, all (2018$)	  | Serious crime rate (per 1,000 residents)		  | 
| Severely rent-burdened households		  | Rental vacancy rate		  | 
| Mean travel time to work (minutes)		  | Area (in sq miles)		  | 



## References

- [Amazon AI Fairness and Explainability Whitepaper](https://pages.awscloud.com/rs/112-TZM-766/images/Amazon.AI.Fairness.and.Explainability.Whitepaper.pdf)
- [Facebook’s five pillars of Responsible AI](https://ai.facebook.com/blog/facebooks-five-pillars-of-responsible-ai/)
- [Fairness Indicators](https://github.com/tensorflow/fairness-indicators)
- [How Facebook got addicted to spreading misinformation](https://www.technologyreview.com/2021/03/11/1020600/facebook-responsible-ai-misinformation/)
- [Normal and New Normal NYC Subway Traffic 2017-21](https://www.kaggle.com/eddeng/nyc-subway-traffic-data-20172021)
- [What-If Tool](https://pair-code.github.io/what-if-tool/)
- [https://www.researchgate.net/publication/-228716874_Integration_of_public_transport_and_urban_planning](https://www.researchgate.net/publication/-228716874_Integration_of_public_transport_and_urban_planning)[^1]: 
- [https://www.hindawi.com/journals/ddns/2016/1478413/](https://www.hindawi.com/journals/ddns/2016/1478413/)[^2]: 
- [https://www.nytimes.com/interactive/2021/03/08/climate/nyc-transit-covid.html](https://www.nytimes.com/interactive/2021/03/08/climate/nyc-transit-covid.html)[^3]: 
- [https://www.hindawi.com/journals/mpe/2020/1871423/](https://www.hindawi.com/journals/mpe/2020/1871423/)[^4]: 



## Acknowledgements

Thanks to our friends at Mastercard (Apurva, Bharathi, Hui Chiang, Idaly and Louis) for their advice and guidance on AI fairness.
