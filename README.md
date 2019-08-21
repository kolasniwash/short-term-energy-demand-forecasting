# day-ahead-load-forecasting
Final project repo for the AkademyAi bootcamp. Investigates various models to predict hourly day ahead forecasts in the Spanish energy market.

## Problem Definition
In order to appropriately meet energy demands on a 24 hour basis Transmission Service Operators make forecasts for the maximum power expected over the coming 24 hour period. These forecasts are used in the planning of supply dispatch and for day-ahead bidding processes. 

Errors in prediction can be costly. Considering a levelized cost of energy of EUR50/MWh and an average difference in forecast to observed consumption of ~200MW. The daily value of the error is EUR 240,000. An improvement in forecasting that reduces error by 2% represents a potential annual cost reduction of EUR 1.5M. Similar examples can also be made for predicting load forecasts from solar, wind, battery storage, and other intermittent energy sources. 


## Task Identification

### 1. Data Cleaning/Preprocessing
 
- [X] Retreve data from public sources
  - [x] Energy load data: entose Transparency Platform
  - [ ] (optional) Calendar data: Pandas library
  - [ ] (optional) Weather Data: **Need a source**
- [ ] Clean data
  - [x] Parse dates into datetime
  - [ ] Evaluate quantitiy of nans and impute missing if possible
  - [ ] Elimate duplicated values
- [ ] Process energy load  data into feature vectors
  - [ ] 'Univariate' Day In, and predict Day + 1
  - [ ] 'Multivariate' In: Day, Day-1, Day-2, Day-7, and predict Day + 1
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
  - [ ] Identify correlated date and weather features with high correlation.

### 3. Model Building
- [ ] Naive forecast models AKA test rigs
  - [ ] Linear naive model
  - [ ] Standard ML models: regression, randomforest, etc.
  - [ ] Single layer of 24 perceptrons, one per output hour of the day.
  - [ ] CNN
  - [ ] LSTM
- [ ] Univariate 
  - [ ] Moving averages, ARIMA
  - [ ] Multi layer perceptron
  - [ ] CNN
  - [ ] LSTM
- [ ] Multivariate
  - [ ] Moving averages, SARIMA
  - [ ] Multi layer perceptron
  - [ ] CNN
  - [ ] LSTM

### 4. Model Improvement
- [ ] Learning rate optimization (tensorflow callback method)
- [ ] (optional) Add dates features
- [ ] (optional) Add weather features

### 5. Model Evaluation 
- [ ] Record inital model results - build table for model result tracking. 
- [ ] Plot errors distributions
