#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing
# 
# This notebook explains how to prepare the data cleaned in section 1.1 for use in the naive model set.
# 
# 
# 1. Problem Framing
# 2. Data Manipulation
# 
# 
# 
# ## Problem Framing
# 
# Data read from the file load_forecast_2016_2018 is in the format:
# 
# |time |day_forecast|actual_load|
# |-----|------------|---------------|
# | 2016-01-01 00:00:00 | 23273.0 | 22431.0 |
# | 2016-01-01 01:00:00	| 22495.0	| 21632.0 |
# | 2016-01-01 02:00:00	| 21272.0	| 20357.0 |
# 
# In this form it is easy to predict the next hour in advance. One wya to predict the next day would be to predict the same hour the next day i.e. jumping 24 values. According to the problem definition the goal is to predict the next 24 hours in advance. An easier way to do this is to create a window of calendar days consisting of 24 hour segments. That way each indivdual hour from day-1, is used to predict each hour of the current day.
# 
# Isolating for only the actual_load, the data is reformatted into the following format
# 
# 
# | date | h00 | h01 | ... | h23 |
# |------|----|----|-----|-----|
# | 2016-01-01 | 22431.0 | 21632.0 | ... | 24000.0 |
# | 2016-01-01 | 22113.0 | 20515.0 | ... | 26029.0 |
# 
# 
# Therefore the naive problem becomes predicting the next day's loads at any given hour using the previous day's hourly loads. We have reduced a multiple input, multiple output problem into 24 univariate naive predictions.
# 
# 
# ### Walk Forward Validation
# 
# Using the above structure we can establish a walk forward method of predicting the next value. The table below shows how for each hour of the day, there is a separate model to predict the next day's predicted maximum load for the given hour. In this case ARIMA is a distinct statistical model for hours h0, h1, ... h23.
# 
# | date | h00 | h01 | ... | h23 |
# |------|----|----|-----|-----|
# | 2016-01-01 | 22431.0 | 21632.0 | ... | 24000.0 |
# | |⌄⌄⌄⌄⌄|⌄⌄⌄⌄⌄|⌄⌄⌄⌄⌄|⌄⌄⌄⌄⌄|
# ||ARIMA-h0|ARIMA-h1|...|ARIMA-h23|
# | |⌄⌄⌄⌄⌄|⌄⌄⌄⌄⌄|⌄⌄⌄⌄⌄|⌄⌄⌄⌄⌄|
# | 2016-01-01 | 22113.0 | 20515.0 | ... | 26029.0 |
# 
# ### Feature engineering (Windowing)
# 
# To prepare the data for the walk forward validation model structure the data is windowed per day. This means hourly data at time t is shifted to become time t-1. The model is then fed data from time t-1, and compared to time t in a supervised learning problem.
# 
# Generalizing this, more features are defined in a similar way. Data is shifted by x steps, and the last x steps of the data set are removed. The result of a similar transform is shown below.
# 
# | date | h00 | h01 | ... | h23 |
# |------|----|----|-----|-----|
# | t | 22431.0 | 21632.0 | ... | 24000.0 |
# 
# 
# #### Days shifted by x steps
# 
# | Date | t | t-1 | t-2 | t-3 |
# |---|---|---|---|---|
# | Day 1:| 0 | Nan | Nan | Nan |
# | Day 2:| 1 | 0 | Nan | Nan |   
# | Day 3:| 2 | 1 | 0 | Nan |
# | Day 4:| 3 | 2 | 1 | 0 |
# | Day 5:| 4 | 3 | 2 | 1 |
# | Day 6:| 5 | 4 | 3 | 2 |
# | Day 7:| 6 | 5 | 4 | 3 |
# 
# 
# 
# #### Days shifted and truncated
# | Date | t | t-1 | t-2 | t-3 |
# |---|---|---|---|---|
# | Day 4:| 3 | 2 | 1 | 0 |
# | Day 5:| 4 | 3 | 2 | 1 |
# | Day 6:| 5 | 4 | 3 | 2 |
# | Day 7:| 6 | 5 | 4 | 3 |
# 
# Where each set of t represents a vector of length (number of days, 24 hours).
# 
# 
# 
# ## Summary of functions
# - transform_to_windows
#     - converts the data from row data into windowed rows where each row is a day with 24 columns representing each hour of the day.
# - plot_hour
#     - helper function to view series data
# - shift_by_days
#     - helper function to make_shifted_features. calls pd.shift on the input dataframe to shift the data x number of rows.
# - make_shifted_features
#     - calls shift_by_days for a list of shift values.
#     - shortenes the resulting dataframe
# - trim_length
#     - helper function to make_shifted features. Shortens the length of the final dataframe of fatures so there are no NaNs.
# - rename_cols
#     - Helper function used in make_shifted_features. Labels the columns of the shifted dataframes with an appropriate label indicating the shift value.
# 

import pandas as pd
import numpy as np


#a function to transform
def transform_to_windows(data, load_type='actual_load'):
    """
    Input
    
    Output
    
    """
    
    #from the original datetime index create new columns with each of the year, month, day, and hour.
    data.loc[:,'year'] = data.index.year
    data.loc[:,'month'] = data.index.month
    data.loc[:,'day'] = data.index.day
    data.loc[:,'hours'] = data.index.hour
    
    #construct datetimes from the split year, month, day columns
    data.loc[:,'date'] = pd.to_datetime(data.loc[:,['year', 'month', 'day']], format='%Y-%m-%d', errors='ignore')
    
    #set the index to dates only
    data = data.set_index(pd.DatetimeIndex(data['date']))
    
    #drop non target columns 
    data = data.loc[:,[load_type, 'hours']]
    
    #pivot the table into the format Date h0, h1, ...h23
    data = data.pivot(columns='hours', values=load_type)
    
    return data



#define a helper function to plot the indivdual hours
def plot_hour(data, hour):
    
    #set figure size
    fig = plt.figure(figsize = (20,7))
    
    #select the hour to display
    series = data.iloc[:,hour]
    
    plt.plot(series.index, series.values)




def shift_by_days(data, num_days):
    """
    Input a timeseries of the form 24 hourly measurements per day
    
    Output returns 
    
    """
    data_shifted = data.shift(num_days)
    
    return data_shifted
    


def make_shifted_features(data, shifts_list):
    
    #set the columns names on the original data set
    data = rename_cols(data, 0)
    
    #initate list of dataframes
    periods = [data]
    
    #cycle through list of shifts i.e. features
    for shift in shifts_list:
        
        if shift == 0:
            pass

        else:
            #shift the data by shift value
            data_shifted = shift_by_days(data, shift)

            #update column identifers
            data_shifted = rename_cols(data_shifted, shift)
        
            periods.append(data_shifted)
        
    #concatenate all shifted datasets into one dataframe.    
    data = pd.concat(periods, axis = 1)
    
    data = trim_length(data, shifts_list)
    
    return data




def rename_cols(data, shift):
    
    cols = data.columns
    
    cols_list = []
    
    for idx, col in enumerate(cols):
        
        new_col = 't-' + str(shift) + ' h_' + str(idx)
        
        cols_list.append(new_col)
        
    data.columns = cols_list
    
    return data



def trim_length(data, shifts_list):
    
    start_point = sorted(shifts_list, reverse=True)[0]
    
    return data.iloc[start_point:,:]


