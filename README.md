# Short-term-energy-demand-forecasting
This project is the final project repo for the [AkademyAi](akademy.ai) machine learning course. Investigates various models to predict short-term (24 hour advance) energy demands in the Spanish energy market.

[Watch the final project presentation](https://youtu.be/KaWCwBD_UBA) for this repo.

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

intro

main questions




### Data sources

### Modelling Short-Term Energy Demand


Features input
- autocorrelated energy consumption
- hourly weather data
- day of the week
- holidays

Output
- Hourly peak demand for the next 24 hour window

### Supervised learning problem



#### SARIMA & Prophet

Image here of inputs and output mapping

#### LSTM

Image here of inputs fomulation and output mapping

<img src="img/walk-forward-validation.png" width=600 height=400>


### Cross validation descrption

Cross validation (backtesting) was used to verify the results of forecasts. 


<img src="img/walk-forward-validation.png" width=600 height=400>





Errors in prediction can be costly. Considering a levelized cost of energy of EUR50/MWh and an average difference in forecast to observed consumption of ~200MW. The daily value of the error is EUR 240,000. An improvement in forecasting that reduces error by 2% represents a potential annual cost reduction of EUR 1.5M. Similar examples can also be made for predicting load forecasts from solar, wind, battery storage, and other intermittent energy sources. 




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

Each model pipeline may be run independently. To replicate results, or build on this project, you can get started by:

1. Clone this repo
2. Raw data used for this project Data is found in CSV format within this repo [here](Repo folder containing raw data) within this repo.
    a. Updated energy data can be downloaded from the [ENTSOE Transparency Platform](https://transparency.entsoe.eu/)
    b. Weather data was obtained from the [OpenWeatherApi](https://openweathermap.org/api) and additional data may be purchased.
3. Follow the requirements.yml file to install dependencys
4. Data processing, transformation, and models are found in the main repo.
5. Executing any of the following files will run the designated model:
    a. model_sarima.py
    b. model_prophet.py
    c. model_lstm.py
6. Model output is saved in json for the SARIMA and LSTM. Prophet output is saved in csv.
7. Results folder stores the model outputs under their respective folder names.

## Featured Notebooks & Deliverables

#### Models
* [SARIMA](link)
* [Prophet](link)
* [LSTM](link)

#### Anlysis and Helper Functions
* [Dataset creation](link)
* [Data window and transform functions](link)
* [Feature Analysis Energy and Weather](link)

#### Communications
* [Presentation Deck](https://github.com/nicholasjhana/short-term-energy-demand-forecasting/blob/master/presentation-short-term-load-forecasting.pdf)
* [Presentation Video](https://youtu.be/KaWCwBD_UBA)
* [Blog Post](link)


## Contact

**Project lead: [Nicholas Shaw](https://github.com/nicholasjhana) nicholas at nicholasjhana.com**
