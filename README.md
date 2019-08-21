# day-ahead-load-forecasting
Final project repo for the AkademyAi bootcamp. Investigates various models to predict hourly day ahead forecasts in the Spanish energy market.

## Problem Definition
In order to appropriately meet energy demands on a 24 hour basis Transmission Service Operators make forecasts for the maximum power expected over the coming 24 hour period. These forecasts are used in the planning of supply dispatch and for day-ahead bidding processes. 

Errors in prediction can be costly. Considering a levelized cost of energy of EUR50/MWh and an average difference in forecast to observed consumption of ~200MW. The daily value of the error is EUR 240,000. An improvement in forecasting that reduces error by 2% represents a potential annual cost reduction of EUR 1.5M. Similar examples can also be made for predicting load forecasts from solar, wind, battery storage, and other intermittent energy sources. 


## Task Identification

### 1. Data Cleaning/Preprocessing
 
- [ ] Retreve data from public sources
  - [ ] Energy load data: entose Transparency Platform
  - [ ] Calendar data: Pandas library
  - [ ] Weather Data: ???
- [ ] Clean data
  - [ ] Parse dates into datetime
  - [ ] Evaluate quantitiy of nans and imputer if possible
- [ ] Process energy load  data into feature vectors
- [ ] (optional) Process Dates data into dummy variables
- [ ] (optional) Clean weather data and preprocess into feature vectors with load and date data.

### 2. Data Analysis
- [ ] Statistical analysis of processed data in Task set #1
  - [ ] General descrption and investigation of the data's structure.
  - [ ] Is data stationary?
  - [ ] What are distributions of day head forecasts, actual loads, dates
- [ ] (Auto)Correlation analysis of energy data with target vector.
  - [ ] Identify autocorrelated time step features for the multivariate case.
- [ ] Correlation analysis of date and weather data with the target vector.
  - [ ] Identify correlated weather features with high correlation.

### 3. Model Building
- [ ] Naive forecast model (multi- input, multi-output)
  - [ ] Single layer of 24 perceptrons, one per output hour of the day.
  - [ ] Single input of time t-1 predicting time t.
- [ ] MLP forecast model
  - [ ] Hidden layer of 128 perceptrons, output layer of 24 perceptrons
  - [ ] Two test cases:
    - [ ] Single day's worth of data input (24 hours of data), with 24 data point output.
	- [ ] Mutliple day's worth of data input (i.e. 5 X 24 hours data), with 24 data point output.
- [ ] CNN forecast model
    - [ ] Naive forecast variant t-1 --> t
    - [ ] Multi day input forecast variant t-1, t-2, t-3, t-7, t-30 --> t
- [ ] LSTM forecast model
    - [ ] Naive forecast variant t-1 --> t
    - [ ] Multi day input forecast variant t-1, t-2, t-3, t-7, t-30 --> t

### 4. Model Evaluation
- [ ] Record inital model results - build table for model result tracking. 
- [ ] Optimize learning rate in models (tensorflow callback method)
- [ ] Plot errors distributions
- [ ] Add date and weather features and reevaluate the models. 

## Model Explanations
