# Short-term-energy-demand-forecasting
This project is the final project repo for the [AkademyAi](akademy.ai) machine learning course. Investigates various models to predict short-term (24 hour advance) energy demands in the Spanish energy market.

Watch the final project presentation for this repo.

#### -- Project Status: [Completed]

## Problem Definition and Motivation
This project is inspired by the paper [Tackling Climate Change with Machine Learning](https://arxiv.org/abs/1906.05433) where forecasting is identified as one of the highest impact research areas to contributing to more renewable energy in the grid. 

The objective was to learn how to implement classic and state of the art prediction models, and walk forward cross validation.

The specific problem addressed is to use past energy consumption data, day of the week, holidays, and weather data to once daily predict the next 24 hours of energy demand. This is a highly relevant problem carried out everyday by electrical grid Transmission Service Operators (TSOs) across the world. In order to appropriately meet energy demands TSOs issue energy demand forecasts once a day for the coming 24 hour period. The expected maxium energy demand is forecasted on an hourly basis and consists of 24 hourly slices. These forecasts are used in the planning of supply dispatch, for day-ahead bidding processes, and combined with ultrashort term (6 hours or less) forecasts that maintain balance in the grid. 

Timeseries forecasting models implemented in this project are:
1. SARIMA - Seasonal Autoregressive Integrated Moving Average
2. Prophet General Additive Model by Facebook
3. Long-Short Term Memory Nerual Network


### Methods Used
* Data Wrangling
* Machine Learning
* Regression
* Neural Networks
* Predictive Modelling
* Walk forward cross validation
* Hypothesis Testing

### Technologies
* Python
* Keras, Tensorflow
* Pandas, Numpy, Jupyter
* Statsmodels
* Prophet
* Joblib, holidays libraries
* Google Cloud Platform

## Project Description







(Provide more detailed overview of the project.  Talk a bit about your data sources and what questions and hypothesis you are exploring. What specific data analysis/visualization and modelling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)




## Project needs and core tasks

- data processing/cleaning
    - cleaning of energy data, weather data, and generation of holiday data
    - process data to generate autoregressive features
    - processing data to frame the problem for SARIMA, Prophet, LSTM
- data exploration
    - visualize energy consumption patterns at different temporal scales
    - visualize weather correlations and 
- statistical modeling
    - (auto)correlation analysis of model features and feature selection
    - PCA transformation of colinear (weather) features
    - parameter selection for SARIMA model: determining differncing, seasonality, and trend components
    - parameter selection for Prophet model: configure base mode, additional regressors
- machine learning
    - configuration, hyperparmeter tuning, training, and testing of LSTM neural network
- data pipeline
    - helper functions to prepare input, calcualte erros, run walk forward cross validation, and produce visualizations for each model
- reporting/presentation
    - documentation of helper functions
    - presentation of work at live event 

## Run these models yourself

Each model pipeline may be run independently. To replicate results, or build on this project, you can get started by

1. Clone this repo
2. Raw data used for this project Data is found in CSV format within this repo [here](Repo folder containing raw data) within this repo.
    a. Updated energy data can be downloaded from the [ENTSOE Transparency Platform](https://transparency.entsoe.eu/)
    b. Weather data was obtained from the [OpenWeatherApi](https://openweathermap.org/api) and additional data may be purchased.
3. Follow the requirements.yml file to install dependencys




4. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
5. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)


## Contributing DSWG Members

**Team Leads (Contacts) : [Full Name](https://github.com/[github handle])(@slackHandle)**

#### Other Members:

|Name     |  Slack Handle   | 
|---------|-----------------|
|[Full Name](https://github.com/[github handle])| @johnDoe        |
|[Full Name](https://github.com/[github handle]) |     @janeDoe    |

## Contact
* If you haven't joined the SF Brigade Slack, [you can do that here](http://c4sf.me/slack).  
* Our slack channel is `#datasci-projectname`
* Feel free to contact team leads with any questions or if you are interested in contributing!









Errors in prediction can be costly. Considering a levelized cost of energy of EUR50/MWh and an average difference in forecast to observed consumption of ~200MW. The daily value of the error is EUR 240,000. An improvement in forecasting that reduces error by 2% represents a potential annual cost reduction of EUR 1.5M. Similar examples can also be made for predicting load forecasts from solar, wind, battery storage, and other intermittent energy sources. 


## Task Completed

### 0. Problem Definition
- [x] Forecast target
- [x] Data Structure inputs and outputs

### 1. Data Cleaning/Preprocessing
 
- [X] Retreve data from public sources
  - [x] Energy load data: entose Transparency Platform
  - [x] (optional) Calendar data: Pandas library
  - [x] (optional) Weather Data: OpenWeatherApi
- [x] Clean data
  - [x] Parse dates into datetime
  - [x] Evaluate quantitiy of nans and impute missing if possible
  - [x] Elimate duplicated values
- [x] Process energy load  data into feature vectors
  - [x] 'Univariate' Day In, and predict Day + 1
  - [x] 'Multivariate' In: Day, Day-1, Day-2, Day-7, and predict Day + 1
- [ ] (optional) Process Dates data into dummy variables
- [ ] (optional) Clean weather data and preprocess into feature vectors with load and date data.

### 2. Dataset Analysis
- [x] Statistical analysis of processed data in Task set #1
  - [x] General descrption and investigation of the data's structure.
  - [x] Is data stationary?
  - [x] What are distributions of day head forecasts, actual loads, dates
- [ ] (Auto)Correlation analysis of energy data with target vector.
  - [ ] Identify autocorrelated time step features for the multivariate case.
- [ ] Correlation analysis of date and weather data with the target vector.
  - [ ] Identify correlated date and weather features with high correlation.

### 3. Data Transforms
- [x] Utility to join csvs into single df, clean, interpolate nans
- [x] Utility to transform Day>Hour format to Day>Day format (columns hours)
- [x] Utility to add labels to transformed columns and autoregressive windows

### 4. Multistp Model Building
- [ ] Baseline Forecast Models
  - [x] Previous Day By Day
  - [ ] Year ago day-by-day
- [ ] Classic Forecast models
  - [ ] ARIMA Hour-by-hour
    - [ ] Box Jeknens Method for parameter discovery
    - [ ] Grid search around parameters
- [ ] Neural Network (paper model implementation)
  - [ ] CNN / LSTM / MLP
- [ ] Error Evaluation
  - [ ] Error plots

### 5. Model Improvement (Optional Ideas)
- [ ] Hyperparmeter Tuning
  - [ ] Learning rate optimization (tensorflow callback method)
- [ ] Multivariate mutli step models
  - [ ] Add weather/dates features
    - [ ] ARIMA (w/ exod vars)
    - [ ] CNN / LSTM / MLP
  - [ ] Multi headed model
- [ ] Feature engineering
  - [ ] Autoregressive analysis of features.
  - [ ] Clustering of weather features
  
### 6. Possible Pivots
- [ ] Compare two NN models instead of ARIMA
- [ ] Compare ARIMA vs SARIMA, ARIMAX, SARIMAX
- [ ] Skip ARIMA, build multivariate multi step directly (ie. using day and weather data)
- [ ] Swap energy demand data for Solar data: forecast solar generation
  - [ ] Option 1 Model comparisons
  - [ ] Option 2 Use mutliple input variables, one model
