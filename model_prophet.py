#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark')

from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline


from statsmodels.graphics.gofplots import qqplot

from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_cross_validation_metric


from model_persistence import get_persistence_dataset, train_test_split, calculate_errors, plot_error

from create_day_types import get_days_dummies


# ### Import combined data set

# In[2]:

def get_prophet_data(start_date, stop_date):
    data_comb = pd.read_csv('./data/cleaned_data/energy_weather_2015_2018.csv', parse_dates=True, index_col='time')
    #data_comb.head(3)

    data_comb = data_comb[start_date:stop_date]

    return data_comb

# In[10]:


def standard_index(data):
    data = data.reset_index()
    
    #now that the datetime index is gone be double sure our data is ordered correctly.
    data.sort_values(by=['ds'], inplace=True)
    
    return data


# In[11]:


def prophet_cv_performance(model, train, minmax_pipe):
    
    #fit the model
    model.fit(train)
    
    #run cross validation
    #cv_results = cross_validation(model, initial='365 days', period='24 hours', horizon='24 hours')
    
    cv_results = cross_validation(model, initial='366 days', period='24 hours', horizon='24 hours')
    
    #inverse transform the target and prediction columns
    cv_results[['yhat', 'y']] = minmax_pipe.inverse_transform(cv_results[['yhat','y']])
    
    #get performance results on the cv output
    cv_performance = performance_metrics(cv_results)
    
    #display the results
    print('Model Mean Hourly MAE {0:.2f}' .format(np.mean(cv_performance['mae'])))
    print('Model Mean Hourly MAPE {0:.2f}' .format(np.mean(cv_performance['mape'])))
    
    return cv_results, cv_performance


# In[ ]:





# In[ ]:

def pca_minmax_pipes():

    ## apply PCA pipeline to the weather features.

    #initate a min max scaler
    minmax = MinMaxScaler()

    #initate a PCA object to cpature 90% of dimensionality
    pca = PCA(n_components=2)

    #initate the pipeline for pc
    pca_pipe = make_pipeline(pca, minmax)

    minmax_pipe = make_pipeline(minmax)

    return pca_pipe, minmax_pipe


def make_weather_data(data_comb, pca_pipe):
    weather_cols = ['temp','pressure','wind_speed','rain_1h','rain_3h','snow_3h','heatdd','cooldd']

    #select the weather columns and make sure there are no Nans before transform
    data_weather = data_comb[weather_cols].interpolate(method='linear')

    #transform the weather data
    pca_weather = pca_pipe.fit_transform(data_weather)

    #make dataframe with the transformed weather pca
    pca_weather_df = pd.DataFrame(pca_weather, index=data_weather.index, columns=['pca1', 'pca2'])

    return pca_weather_df


def make_energy_data(data_comb, minmax_pipe):
    #slice out the energy data
    energy_data = data_comb['actual_load']

    #apply minmax scaler. reshape because passing Series into function expects 2d array
    energy_minmax = minmax_pipe.fit_transform(np.reshape(energy_data.values, (-1,1)))

    #convert back to dataframe
    energy_minmax_df = pd.DataFrame(energy_minmax, index=energy_data.index, columns=['y'])

#energy_minmax_df.head(3)
    return energy_minmax_df


def make_train_test(pca_weather_df, energy_minmax_df, split_date):

    #combine the energy and weather again as scaled features
    data_scaled = pd.concat([energy_minmax_df,pca_weather_df], axis=1)

    data_scaled.index.name='ds'

    #split in train and test
    train_e_w, test_e_w = train_test_split(data_scaled, split_date)


    #change and order index for prophet 
    train_e_w = standard_index(train_e_w)

    ##change and order index for prophet

    test_e_w = standard_index(test_e_w)

    print('Length of the train dataset {}' .format(len(train_e_w)))
    print('Length of the test dataset {}' .format(len(test_e_w)))

    return train_e_w, test_e_w


def initalize_model(yearly_seasonality=True, seasonality_mode='additive', weather_prior=0.8):
    #weather base model initalize
    model = Prophet(yearly_seasonality=yearly_seasonality, 
                        seasonality_mode=seasonality_mode)

    #add regressors for the pca weather regressors
    model.add_regressor('pca1', prior_scale=weather_prior, mode='additive')
    model.add_regressor('pca2', prior_scale=weather_prior, mode='additive')

    return model



def run_prophet_weather_model(start_date, stop_date):

    #initalize dataset. contains both energy and weather data
    data_comb = get_prophet_data(start_date, stop_date)

    #setup the pca nad minmax transformers
    pca_pipe, minmax_pipe = pca_minmax_pipes()

    #scale and prep the weather data for prophet format
    pca_weather_df = make_weather_data(data_comb, pca_pipe)

    #scale and prep the energy data for prophet format
    energy_minmax_df = make_energy_data(data_comb, minmax_pipe)

    #make the training and test sets from the energy and weather data
    train_e_w, test_e_w = make_train_test(pca_weather_df, energy_minmax_df, '2017-12-31')

    #setup the prophet model
    weather_model_2017 = initalize_model(yearly_seasonality=True, seasonality_mode='additive', weather_prior=0.8)

    return weather_model_2017, train_e_w, minmax_pipe


def process_and_save_results(weather_model_2017, train_e_w, minmax_pipe):

    #fit model, run cross validation, return results, performance, and mean MAE
    weather_cv_results, weather_cv_perfomance = prophet_cv_performance(weather_model_2017, train_e_w, minmax_pipe)

    #plot the crossvalidated error performance
    plot_cross_validation_metric(weather_cv_results, metric='mae', rolling_window=0);

    path = './results/prophet/'

    weather_cv_results.to_csv(path + 'weather_cv_results.csv')
    weather_cv_perfomance.to_csv(path + 'weather_cv_performance.csv')

    print('Results saved at {}'.format(path))


print('Running Crossvalidation of Prophet Weather Model with the following parameters')
print('Yearly seasonality is TRUE')
print('Seasonality mode is ADDITIVE')
print('Weather prior is 0.8')
print('Inital training period of cross validation is 365 days')
print('Type: Multivariate')


weather_model_2017, train_e_w, minmax_pipe = run_prophet_weather_model('2016-01-01', '2018-03-31')

print('Saving results...')

process_and_save_results(weather_model_2017, train_e_w, minmax_pipe)

print('Done')










