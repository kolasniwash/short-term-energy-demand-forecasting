#!/usr/bin/env python
# coding: utf-8

# # Energy Load Data Cleaning Explanation
# 
# This notebook descrbes the process used to construct and clean the dataset.
# 
# Data was aquired from entsoe Transparency Platform at the following [link](https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=09.08.2015+00:00|CET|DAY&biddingZone.values=CTY|10YES-REE------0!BZN|10YES-REE------0&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)#) (2015 data). Data is downloadable on an annual basis. this workbook constructs an example dataset using the years 2016-2018. The same functions may be used to construct any number of years available from this source.
# 
# Processes completed in the following functions:
# 1. format_data
#     - renames the columns
#     - shortens the text identifier for times
#     - converts to a Datetime index
# 2. combine_annual_data
#     - joins a dictionary if dataframes
# 3. interpolate_nans
#     - fills the missing values using a linear interpolation method
# 

# In[3]:


import pandas as pd
import numpy as np


def process_energy_data(path='./data/raw_data_ES/', files=['Total Load - Day Ahead _ Actual_2016.csv', 'Total Load - Day Ahead _ Actual_2017.csv', 'Total Load - Day Ahead _ Actual_2018.csv']
):
    """
    Input the path and a list of file nmaes for the annual data as exported from ENTOSE. 

    Output a single dataframe with energy load data combined and cleaned.

    """

    #try to load in list of the datasets
    try: 
        data_sets = [pd.read_csv(path+file) for file in files]
    except:
        print('Error loading files')


    #create list of keys for the dictionary of dataframes. one per year.
    years = range(len(files))

    #create a dictionary of formatted pandas dataframes where key is each year
    format_sets = {year: format_data(data_set) for year,data_set in zip(years, data_sets)}

    #combine into one single dataframe
    data = combine_annual_data(format_sets)

    #linearly interpolate nan values
    data = interpolate_nans(data)

    #test for rows that are duplicated and keep first instance
    data = remove_duplicated_rows(data)

    return data



def format_data(data):
    '''
    Input: A dataframe of Day Ahead Total Load, and Actual Load obtained from csv data obtained from the entsoe Transparency Platform.
    
    Descrption:
    Input is a 3 column dataframe consisting of text time stamps with hourly frequency. 
    - Function formats the string in order to be formatted into a datetime.
    - Appends a datetime index and drops the time strings
    
    Output: A 2 column dataframe with a DatetimeIndex
    
    '''
    
    #set column names to something simple
    data.columns = ['time', 'day_forecast',
       'actual_load']

    #set the time to the first element in the time string. 
    #So 01.01.2018 00:00 - 01.01.2018 01:00 becomes 01.01.2018 00:00
    data['time'] = data['time'].str.split('-').apply(lambda x: x[0]).str.strip()
     
    #set the time strings to datetime obejects and set index as date time
    datetimes = pd.to_datetime(data['time'], format='%d-%m-%Y %H%M', errors='ignore')
    data_ = data.set_index(pd.DatetimeIndex(datetimes))
    
    #remove extra time column with original string objects
    data_time = data_[['day_forecast', 'actual_load']]
    
    return data_time


def combine_annual_data(dictionary):
    """
    Input: a dictionary of dataframes.
    
    Output: a single dataframe
    """
    
    all_data_list = []
    
    for key in dictionary.keys():
        all_data_list.append(dictionary[key])
        
    data_all_years = pd.concat(all_data_list)
    
    return data_all_years



# ### Clean NANs
# 
# This data will be used for predicting day ahead energy demand. In dealing with nan values it is important not to change the structure of the data. 
# 
# Two ways this can occur:
#    1. dropping values changes number of observations in a day. number of daily observations per day needs to line up with the days before and after.
#    2. filling missing values with a single value (i.e. series mean value) is not representiative of the temporal nature of the data

def interpolate_nans(data):
    """
    Inputs:
    - data --- a dataframe of timeseries data
    - columns --- a list of column header names
    
    Process:
    Applies linear interpolation to fill the missing entries per column
    
    output: a dataframe
    """
    
    #try to convert data to float
    try:
        data = data.astype(float)
    except:
        #for typical nans filled with strings replace with nans.
        for char in ['-', '--', '?']:
            data = data.replace('-', np.nan)
        
        #set object type to float
        data = data.astype(float)

    data = data.interpolate(method='linear', axis=0)
    
    return data


def remove_duplicated_rows(data):
    """
    Input a timeseries dataset with multiple rows of the same index value.
    
    Output timeseries dataset with first occurance of duplicated rows.
    """
    #identify the duplicated elements
    duplicated_rows_bool = data.index.duplicated()
    
    #invert the array element wise
    keep_rows = np.invert(duplicated_rows_bool)
    
    #return the original dataframe removing the duplicated values
    return data[keep_rows]


