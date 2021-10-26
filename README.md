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


## Dataset: New York City (NYC) Subway Traffic

The dataset we selected consists of subway traffic in NYC, along with neighbourhood census data of the city. It is hosted on Kaggle by Edden, who has performed preprocessing steps to convert the raw data provided by The Metropolitan Transportation Authority (MTA), North America's largest transportation network. The census data is from NYU Furman Center's New York City Neighborhood Data Profiles and the neighborhood data is from University of Berkeley GeoData Library.

### Problem Statement

Public transport has become a necessity in our modern landscape. Agencies are interested in capitilising on data on public transport usage such as subway traffic to inform their location based business decisions.

Government entities involved in urban planning might utilise subway traffic conditions to determine neighbourhoods that could benefit from neighbourhood rejuvenation or to inform other land usage planning decisions.[^1] Governments can also benefit from having a gauge of how investing into a neighbourhood will affect traffic volume via subway in different areas through census data.

Alternatively, businesses such as the real estate and media industries can benefit from integrating subway traffic conditions as their decision making factors. Subway traffic can greatly affect real estate prices, thus this can inform real estate developers in their development strategies[^2]. With 1.7 billion turnstile swipe in 2019 alone, the subway is New York City's most popular mode of transit[^3], priming it to an effective mode of advertisement - a reason why subway advertising has become a regular part of every commuter's life. Subway traffic can also be used to inform media agencies of their audience, allowing them to identify prime locations - and thereby the corresponding bullet services - for their advertisement campaigns in order to maximise their effectiveness[^4].

### Approach

The baseline model is built for a time series binary classification problem to predict whether subway traffic is high or low.

### Data preprocessing

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

As the neighbourhood census data was taken on June 2020, we selected 3 months worth of data from the subway data between Apr-Jun 2020, with the assumption that the neighbourhood census data stays constant for the 3 months. 10 columns were then selected to be in the final dataset.
* Datetime
* Stop Name
* Connecting Lines
* Daytime Routes
* Division
* Structure
* Borough
* Neighbourhood
* Entries
* Exits

The entries and exits were summed to get the total traffic flow in each subway station for each period. One-hot encoding is then done for - "Division", "Structure", "Borough", "Connecting Lines" and "Daytime Routes" - before merging the subway data and neighbourhood census data into the final dataset.


## References

Amazon AI Fairness and Explainability Whitepaper: https://pages.awscloud.com/rs/112-TZM-766/images/Amazon.AI.Fairness.and.Explainability.Whitepaper.pdf 
Facebook’s five pillars of Responsible AI: https://ai.facebook.com/blog/facebooks-five-pillars-of-responsible-ai/ 
Fairness Indicators: https://github.com/tensorflow/fairness-indicators
How Facebook got addicted to spreading misinformation: https://www.technologyreview.com/2021/03/11/1020600/facebook-responsible-ai-misinformation/
Normal and New Normal: NYC Subway Traffic 2017-21: https://www.kaggle.com/eddeng/nyc-subway-traffic-data-20172021
What-If Tool: https://pair-code.github.io/what-if-tool/
[^1]: https://www.researchgate.net/publication/228716874_Integration_of_public_transport_and_urban_planning
[^2]: https://www.hindawi.com/journals/ddns/2016/1478413/
[^3]: https://www.nytimes.com/interactive/2021/03/08/climate/nyc-transit-covid.html
[^4]: https://www.hindawi.com/journals/mpe/2020/1871423/



## Acknowledgements

Thanks to our friends at Mastercard (Apurva, Bharathi, Hui Chiang, Idaly and Louis) for their advice and guidance on AI fairness.
