# day-ahead-load-forecasting
Final project repo for the AkademyAi bootcamp. Investigates various models to predict hourly day ahead forecasts in the Spanish energy market.

## Problem Definition
In order to appropriately meet energy demands on a 24 hour basis Transmission Service Operators make forecasts for the maximum power expected over the coming 24 hour period. These forecasts are used in the planning of supply dispatch and for day-ahead bidding processes. 

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
