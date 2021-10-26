FairWell is an AI fairness framework/ tool that ...


* * *

### The FairWell Process

| Step | Description | Approach |
| --- | ----------- | --- |
| 1 | Feature Selection and Engineering | FairWell Feature Exploration |
| 2 | Fairness Assessment on Training Data | FairWell Data Fairness Assessment |
| 3 | Model Training | PyTorch... |
| 4 | Fairness Evaluation on Trained Model | FairWell Model Bias Detection |
| 5 | Fairness Mitigation | ... |
| 6 | Iterate | Repeat steps 2 - 5 until Model Evaluation yields satisfactory results |


[GitHub Repository] (https://github.com/hozongsien/uwu/)

* * *

## Example

Utilising the FairWell framework, our team has demonstrated how the FairWell framework can incorporate AI Fairness into..

### Dataset: New York City (NYC) Subway Traffic

The dataset we selected consists of subway traffic in NYC, along with neighbourhood census data of the city. It is hosted on Kaggle by Edden, who has performed preprocessing steps to convert the raw data provided by The Metropolitan Transportation Authority (MTA), North America's largest transportation network. The census data is from NYU Furman Center's New York City Neighborhood Data Profiles and the neighborhood data is from University of Berkeley GeoData Library.

#### Problem Statement

Public transport has become a necessity in our modern landscape. Agencies are interested in capitilising on data on public transport usage such as subway traffic to inform their location based business decisions.

Government entities involved in urban planning might utilise subway traffic conditions to determine neighbourhoods that could benefit from neighbourhood rejuvenation or to inform other land usage planning decisions.[^1] Governments can also benefit from having a gauge of how investing into a neighbourhood will affect traffic volume via subway in different areas through census data.

Alternatively, businesses such as the real estate and media industries can benefit from integrating subway traffic conditions as their decision making factors. Subway traffic can greatly affect real estate prices, thus this can inform real estate developers in their development strategies[^2]. With 1.7 billion turnstile swipe in 2019 alone, the subway is New York City's most popular mode of transit[^3], priming it to an effective mode of advertisement - a reason why subway advertising has become a regular part of every commuter's life. Subway traffic can also be used to inform media agencies of their audience, allowing them to identify prime locations - and thereby the corresponding bullet services - for their advertisement campaigns in order to maximise their effectiveness[^4].

#### Approach

The baseline model is built for a time series binary classification problem to predict whether subway traffic is high or low.

### Feature Selection and Engineering

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

### Fairness Assessment on Training Data 

### Model Training

### Fairness Evaluation on Trained Model

### Fairness Mitigation

| Technique | Description |
| --- | ----------- |
| Undersampling | Text |
| Synthetic Data Generation | Text |

Mitigation is specific to the different priority of the data scientist and your specific project.
