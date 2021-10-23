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


## References

Amazon AI Fairness and Explainability Whitepaper: https://pages.awscloud.com/rs/112-TZM-766/images/Amazon.AI.Fairness.and.Explainability.Whitepaper.pdf 
Facebook’s five pillars of Responsible AI: https://ai.facebook.com/blog/facebooks-five-pillars-of-responsible-ai/ 
Fairness Indicators: https://github.com/tensorflow/fairness-indicators
How Facebook got addicted to spreading misinformation: https://www.technologyreview.com/2021/03/11/1020600/facebook-responsible-ai-misinformation/
Normal and New Normal: NYC Subway Traffic 2017-21: https://www.kaggle.com/eddeng/nyc-subway-traffic-data-20172021
What-If Tool: https://pair-code.github.io/what-if-tool/


## Acknowledgements

Thanks to our friends at Mastercard (Apurva, Bharathi, Hui Chiang, Idaly and Louis) for their advice and guidance on AI fairness.