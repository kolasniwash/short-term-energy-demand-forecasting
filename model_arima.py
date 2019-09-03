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

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('dark')

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from model_persistence import get_persistence_dataset, train_test_split, walk_forward_evaluation, calculate_errors, plot_error



# ##### Summary of hyperparmeter values to investigate
# 
# - p (autoregressive): 24, 48, 168
# - d (differncing): 0, 24, 168
# - q (moving average): 0, 2, 3, 12, 24

# ### ARIMA Model 1: Baseline Parameters
# 
# Baseline parametres are chosen as (p, d, q) = (24, 0, 0)

def __getnewargs__(self):
    return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))

ARIMA.__getnewargs__ = __getnewargs__


def arima_model(history, config):
    
    #convert dataframe to numpy array and flatten into column vector
    history_flat = history.values.flatten()

    #set the ARIMA hyperparamters    
    p, d, q = config

    #initalize the ARIMA model
    model = ARIMA(history_flat, order=(p, d, q))
    
    #fit model
    model_fit = model.fit()
    
    #generate forecast for the next 24 hours
    forecast, _, _ = model_fit.forecast(24)
    
    #save the model
    #name = 'model-' + str(datetime.datetime.now()) +'.pkl'
    #model_fit.save(name)
    
    #print('Done. Model Saved.')
    
    return forecast



def arima_forecasts(model_set = {'arima_model': arima_model}, config=(1,0,0), title='Baseline Arima Model Forecast'):

    #get the data for the model
    data = get_persistence_dataset(start='2015', stop='2018')

    # set the train/test split as 0.75 to split first 3 years as train.
    train, test =train_test_split(data,split_date='2017-12-31')

    #check we are splitting in the right spot
    print('Train set start {} and stop {}' .format(train.index.min(), train.index.max()))
    print('Test set start {} and stop {}' .format(test.index.min(), test.index.max()))

    #initate error lists
    errors = []
    error_means = []

    #iterate through the models and test them in the walk forward evaluation
    for name, function in model_set.items():

        #pass the model, train and test and return the model errors and mean error
        errors_model, error_mean, predictions = walk_forward_evaluation(function, train, test, name, config)

        errors.append(errors_model)
        error_means.append(error_mean)

    print(predictions.shape)

    errors = pd.concat([error for error in errors], axis=1)
    model_forecast = pd.concat([pred for pred in predictions], axis=1)


    plot_error(errors, result_set=list(model_set.keys()), title=title)

    return error_means