#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning: Weather
# 
# 
# #### Data Source:
# Weather data for the top 5 largest cities in spain was obtained from OpenWeatherMap. The data contains hourly information on teperature, pressure, rainfall, cloud index, and weather descrption.
# 
# #### Summary of cleaning actions:
# - Add names to the cities
# - Drop columns that contain no data
# - Convert timestamps to datetimes and set a datetime index
# - In columns with partial data, assume Nans are zero values.
# - Set elements to lower case and remove speical characters in categorical columns
# 
# 
# #### Function list:
# 1. get_clean_weather - takes in weather data and returns a cleaned set for the spain top 5 cities data



import pandas as pd

#import data

def clean_weather_data(data):
    """
    Input: hourly bulk data export from OpenWeatherMaps.
    
    Output: cleaned data
    
    """

    
    #add city names
    city_codes = {3128760 : ' Barcelona', 
                  3117735 : 'Madrid', 
                  3128026 : 'Bilbao', 
                  2509954 : 'Valencia', 
                  6361046 : 'Seville'}
    
    data['city_name'] = data['city_id'].replace(city_codes)

    #drop all columns with only NaN values
    data = data.drop(['lat', 
                      'lon', 
                      'sea_level', 
                      'grnd_level', 
                      'rain_24h', 
                      'snow_today',
                      'rain_today', 
                      'snow_1h', 
                      'snow_24h'], axis=1)


    #convert timestamp to datetime object
    times = pd.to_datetime(data['dt'], unit='s', origin='unix')

    #convert the times to local time zone
    data['dt'] = times.dt.tz_localize('UTC').dt.tz_convert('Europe/Madrid').dt.strftime('%Y-%m-%d %H:%M:%S')

    data = data.set_index(pd.DatetimeIndex(data['dt']))

    #replace null values with zeros in columns with relevant informaiton
    nul_cols = ['rain_1h', 'rain_3h', 'snow_3h']
    data[nul_cols] = data[nul_cols].fillna(0)
    
    return data


def clean_descrption_cols(data):
    """
    small function that sets the descrption columns to lower case, and removes special characters from the names.
    
    """
    
    #make each element in the columns lowercase
    data[['weather_main', 'weather_description']] = data[['weather_main', 'weather_description']].apply(lambda x: x.str.lower())
    
    #remove spcial characters
    special_chars = [',', '/', ':', ';', '-']
    
    for char in special_chars:
        data['weather_description'] = data['weather_description'].str.replace(char,' ')
        
    return data


def get_weather_data(path='./data/weather/spain-weather-2013-2019.csv'):

    data = pd.read_csv(path)
    
    weather_data = clean_weather_data(data)
    weather_data = clean_descrption_cols(weather_data)
    
    return weather_data

