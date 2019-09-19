#!/usr/bin/env python
# coding: utf-8

# ## ARIMA Models
# 
# ARIMA stands for Autoregressive Integrated Moving Average. Compared with the above model it uses a linear combination of past time steps, and moving averages to predict t.
# 
# 
# ## Contents 
# 1. ARIMA parameters analysis
# 2. ARIMA multi-step taining and evaluation
# 
# 
# #### ARIMA Models Evaluated
# 1. Multi-setp output ARIMA
# 
# 
# 
# 
# 
# 
# ARIMA takes only a stationary time series. As explored in the ***DATA ANALYSIS*** notebook the load data can be made stationary by ***ANALYSIS RESULTS***
# 
# 
# 
# We will use the ARIMA model from statsmodels.api which takes the following arguments:
# - p: is the number of lag observations in the model and can be estimated from Autocorrelation plots
# - d: the number of times raw observations are differenced in order to make the series stationary. This is determined with a Dicky-Fuller test.
# - q: the side of the moving average window. The order of moving average.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
sns.set_style('dark')

#import libraries for statistical analysis
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.graphics.gofplots import qqplot


from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


#import libraries for parallel processing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings

#import custom utils including persistence testbench
from model_persistence import get_persistence_dataset, train_test_split, walk_forward_evaluation, calculate_errors, plot_error
from create_day_types import get_days_dummies


#define a model to insert into the training rig created in the persistence workbook
def sarimax_model(history, history_exog, test_exog, config):
    
    #convert dataframe to numpy array and flatten into column vector
    history_flat = history.values.flatten()
    
    p, d, q, P, D, Q, m = config
    
    print(history_exog)
    print(test_exog)
    
    if history_exog is not None:
        #initalize the SARIMA model with exodrogenous variables
        model = SARIMAX(history_flat, order=(p, d, q), seasonal_order=(P, D, Q, m), exog=history_exog, trend='n')
    else:
        #initalize the SARIMA model as univariate sequence
        model = SARIMAX(history_flat, order=(p, d, q), seasonal_order=(P, D, Q, m), trend='n')
       
    
    #fit model
    model_fit = model.fit()
    
    #generate forecast for the next 24 hours
    forecast= model_fit.forecast(24, exog=test_exog)
    
    return forecast


# In[56]:





# ##### GridSearch framework for testing
# 
# Grid searching over the identified hyperparameter space will allow us to see which configuration delivers the best forecasts.
# 
# The grid search code below is adapted from Jason Brownlee's impelmentation in his Book: Deep Learning Time Series Forecasting.

# In[3]:


def score_model(model, data, exog, split_date, cfg, debug=False):
    """
    Function that
    - splits the data into test and train
    - tries to initalize an intence of the model
    - prints and returns the model errors, mean rmse, and predictions
    
    """
    
    result = None
    errors = None
    predictions = None
    
    #get the name of the model
    name = str(model)
    #get the model parameters
    key = str(cfg)

    #initalize the train and test data
    train, test = train_test_split(data, split_date=split_date)

    if exog is not None:
        train_exog, test_exog = train_test_split(exog, split_date=split_date)
    else:
        train_exog, test_exog = None, None
    if debug:
        result, errors, predictions = walk_forward_evaluation(model, train, test, train_exog, test_exog, name, config=cfg)
    
    else:    
        try:
            with catch_warnings():
                filterwarnings('ignore')

                #run walk forward and return rmse, errors, and forecast
                result, errors, predictions = walk_forward_evaluation(model, train, test, train_exog, test_exog, name, config=cfg)

        except:
            print('Bad config' + str(cfg))

    if result is not None:
        print(' Model {}: {}'.format(key, result))
    
    return (key, result, errors, predictions)


# In[4]:


def grid_search(model, data, cfg_list,split_date, debug, exog=None, parallel=True):
    """
    Grid search function
    - launches parallel jobs according to number of cpus available
    - one job per model configuration 
    """

    
    scores = []

    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), verbose=10, backend='multiprocessing')
        #set tasks into the buffer
        tasks = (delayed(score_model)(model, data, exog, split_date, cfg, debug) for cfg in cfg_list)
        #execute tasks in order
        scores = executor(tasks)
    else:
        for cfg in cfg_list:
            score = score_model(model, data, split_date, cfg)

        scores.append(score)
    
    #remove the scores with no values
    scores = [r for r in scores if r[1] != None]
    
    # sort configs by rmse ascending so lowest first
    scores.sort(key=lambda tup: tup[1])
    
    #scores contains the configuration, error_rmse, errors, predictions
    return scores


# In[5]:


# function to create the model configurations to search over.
def arima_configs(t_lags, t_diffs, t_mas, s_lags, s_diffs, s_mas, s_ms):
    configs = list()
    for t_lag in t_lags:
        for t_diff in t_diffs:
            for t_ma in t_mas:
                for s_lag in s_lags:
                    for s_diff in s_diffs:
                        for s_ma in s_mas:
                            for s_m in s_ms:
                                cfg = [t_lag, t_diff, t_ma, s_lag, s_diff, s_ma, s_m]
                                configs.append(cfg)
    return configs


# ##### Collect scores and save to a csv
# 
# Before launching the job in the cloud want to be able to save and revisit the results.
# 
# The scores tuple saves the data in the form (configuration, RMSE, RMSE hourly errors, forecast).    

# In[6]:


def save_all_results(scores):
    
    #break apart the scores tuple
    model_cfg = []
    rmse_list = []
    hourly_errors = []
    forecasts = []

    #assign each element in scores to its own array
    for config, rmse, hs, fs in scores:
        model_cfg.append(config)
        rmse_list.append(rmse)
        hourly_errors.append(hs)
        forecasts.append(fs)
        
        
    #combine model id and rmse results in datframe
    model_names = pd.concat([pd.Series(model_cfg), pd.Series(rmse_list)], axis=1)
    model_names.columns = ['model_cfg', 'total_rmse']

    #generate name and save the summary json file
    name_name = 'sarimax_summary_' + str(datetime.datetime.now()) + '.json'
    model_names.to_json('./results/sarima/' + name_name)
    
    for m_id, h_error, f in zip(model_cfg, hourly_errors, forecasts):
        #set the name of the model
        model_id_errors = 'sarimax_errors'+ str(m_id) + "_"+ str(datetime.datetime.now())+'.json'
        model_id_forecasts = 'sarimax_forecasts'+ str(m_id) + "_"+ str(datetime.datetime.now())+'.json'
       
    
        #save the hourly errors
        h_error.to_json('./results/sarima/hourly_errors/' + model_id_errors)
        
        #save the forecast results
        f.to_json('./results/sarima/forecasts/' + model_id_forecasts)


# ##### Testing the grid search function
# 
# The 

# In[7]:


def test_run_sarimax_gridsearch(cfg_list, debug, start='2018-01-01', stop='2018-03-31', split_date = '2018-03-01'):

    data = get_persistence_dataset(start=start, stop=stop, transformed=True)

    model = sarimax_model
    
    scores = grid_search(model, data, cfg_list, split_date, debug, parallel=True)

    print('Saving...')
    save_all_results(scores)
    print('Results saved in /results/sarima/')
    print('done')



t_lags = [6]
t_diffs = [1]
t_mas = [2]

s_lags = [2]
s_diffs = [1]
s_mas = [1]
s_ms = [24]

start_date = '2017-08-01'
stop_date = '2018-01-31'
split_date = '2017-12-31'


print('Running Single Cross validation of SARIMA')
print('Type: Univariate')
print('Configuration')
print('Trend settings: ({},{},{})'.format(str(t_lags), str(t_diffs), str(t_mas)))
print('Seasonal settings: ({},{},{},{})'.format(str(s_lags), str(s_diffs), str(s_mas), str(s_ms)))
print('Training period start {} stop {}'.format(start_date, split_date))
print('Testing period start {} stop {}'.format(split_date, stop_date))
print('############################################################')
print('############################################################')
print('Note: This is a computationally intensive script. On a 2012 3GHz Macbook with 16GB ram takes 12 hours to converge.')
print('############################################################')
print('############################################################')




cfg_list = arima_configs(t_lags, t_diffs, t_mas, s_lags, s_diffs, s_mas, s_ms)

test_run_sarimax_gridsearch(cfg_list, debug=True, start=start_date, stop=stop_date, split_date=split_date)

