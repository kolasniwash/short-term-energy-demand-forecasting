#!/usr/bin/env python
# coding: utf-8

# # Long Short Term Memory Network Model Implementation
# 
# This notebook implements two variants of a LSTM network to predict the next 24 hour ahead energy demand. Predictions are made at the end of each day for the maximum demand for each hour in the day ahead.
# 
# #### Data Structure
# Two model variants are tested. A univariate case that uses only past energy demand to make the prediction. And a multivariate case that uses past energy demand, weather features (temperature, humidity, wind speed, rain, etc), and day of the week as predictors.
# 
# Data structure for the univarate case is described by the following diagram. In this case we are predicting hour-by-hour using previous data from the same hour. In this way each hour becomes a dataset on its own. We can combine these multiple sets into one single block of data with the shape:
# - INPUT(samples, lags, hour slices)
# - OUTPUT (samples, hour slices)
# 
# <img src="img/lstm-data-transform.png" width=800 height=400 align="middle">
# 
# Data structure for the multivariate case is described as follows. In this case we add features to the lags as a flattened 2D vector of the form (lags, features).
# - INPUT (samples, lags & features, hourly slices)
# - OUTPUT (samples, hourly slices)
# 
# <img src="img/lstm-data-input.png" width=600 height=400 align="middle">
# 
# #### Lagged Features
# This workbook uses between 2 and 9 cross validations on small samples (1-3 years) in while maniuplating the number and composition of the lagged features. The default lags are the previous 7 days. First sequence of variants on this are the last 14, 30, 180, and 365 days. The second variant is to use the previous 7 days, and only multiples of 7 of the target day up to a maximum. I.e. A lag structure of a 30 day maxium lookback would be looksbacks of 1, 2, 3, 4, 5, 6, 7, 14, 21, 28. This was done because the autocorrelation between the target day and days at multiples of 7 is highest.
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')

import json
import codecs
import warnings
warnings.filterwarnings("ignore")

import keras
import tensorflow
from keras.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from model_persistence import get_persistence_dataset, train_test_split, calculate_errors, plot_error
from features_preprocessing import make_shifted_features, transform_to_windows




# ### HELPER FUNCTIONS
# ###### Define a function to calculate and capture the mae
# 
# This could be done in keras directly with the model.evaluate feature. However that will return the scaled values. We are interested in comparing our results with other models and therefore choose to use the model.predict.

# In[2]:


def normalize_df(data):
    
    #normalize the dataset for working with the lstm nn
    scaler = MinMaxScaler().fit(data.values)
    data_normd = scaler.transform(data.values)

    data = pd.DataFrame(data_normd, index=data.index, columns=data.columns)
    
    return data, scaler


# In[3]:


def sample_mape(actual, predicted):
    
    #calcualtes the mean absolute percent error per cross validated sample
    #returns as a percentage
    
    return np.mean(np.abs((actual - predicted) / actual)) * 100


# In[4]:


#define a function to calcualte and capture the mae

def get_sample_total_mae(actual, predicted):
    #list to save scores
    maes = []
    mapes = []

    #loop through each crossvalidation sample
    for i in range(actual.shape[0]):
        
        #calcualte the mae and save to list
        mae = mean_absolute_error(actual[i], predicted[i])
        mape = sample_mape(actual[i], predicted[i])
        
        maes.append(mae)
        mapes.append(mape)
        print('Sample {} total MAE {:.2f}, MAPE {:.2f}%'.format(i,mae, mape))
        
    maes_total = np.mean(maes)
    mape_total = np.mean(mapes)
    
    print('Mean crossvalidation MAE {:.2f} MAPE {:.2f}%'. format(maes_total, mape_total))
        
    return maes, mapes


# In[5]:


def inspect_cv_predictions(actuals, predictions):

    #the number of cross validation sets
    plots = predictions.shape[0]
    
    #the first day, middle day, and last days of the validation set
    days = [1, int(predictions.shape[1]/2), predictions.shape[1]-1]
    
    #set figure
    fig, axs = plt.subplots(plots,3, figsize=(15,20))

    #loop through the samples then loop through the days
    for i, axe in zip(range(plots), axs):
        for day, ax in zip(days, axe):
            
            #plot the predictions
            ax.plot(predictions[i][day], label='predicted')
            #plot actual values
            ax.plot(actuals[i][day], label='actual')
            ax.set_title('Cross val set {}, sample day {}'.format(i,day))

    #position the legend in the top left position of the top left chart.
    axs[0][0].legend(loc=2)
    plt.subplots_adjust(hspace=0.3)


# In[100]:


def cv_week_predictions(actuals, predictions, num_days=7, shift=0):

    #the number of cross validation sets
    plots = predictions.shape[0]
    
    #the first day, middle day, and last days of the validation set
    
    days = [x+shift for x in range(num_days)]
    #days = [1, int(predictions.shape[1]/2), predictions.shape[1]-1]
    
    #set figure
    fig, axs = plt.subplots(plots,1, figsize=(15,20))

    #loop through the samples then loop through the days
    for i, ax in zip(range(plots), axs):
        #for day, ax in zip(days, axe):
            
        #plot the predictions
        ax.plot(predictions[i][days].flatten(), label='predicted')
        #plot actual values
        ax.plot(actuals[i][days].flatten(), label='actual')
        ax.set_title('Cross val set {}'.format(i))
        ax.set_xlabel('Hours')
        ax.set_ylabel('MW')

        #position the legend in the top left position of the top left chart.
        axs[i].legend(loc='lower right')
    plt.subplots_adjust(hspace=0.3)


# In[44]:


def split_sequences(sequences, n_steps, extra_lag=False, long_lag_step=7, max_step=30, idx=0, multivar=False):
    """
    Function modified for use from Deep learning time series forecasting by Jason Brownlee
    """
    
    #if not adding extra lag features adjust max_step and n_steps to aling
    if not extra_lag:
        max_step=n_steps
        n_steps+=1
        
    
    X, y = list(), list()
    for i in range(len(sequences)):
        
        # find the end of this pattern
        #end_ix = i + n_steps
        end_ix = i + max_step
        
        #create a list with the indexes we want to include in each sample
        slices = [x for x in range(end_ix-1,end_ix-n_steps, -1)] + [y for y in range(end_ix-n_steps, i, -long_lag_step)]
        
        #reverse the slice indexes
        slices = list(reversed(slices))
        
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break


        # gather input and output parts of the pattern
        seq_x = sequences[slices, :]
        seq_y = sequences[end_ix, :]

        X.append(seq_x)
        y.append(seq_y)
        
    X = np.array(X)
    y = np.array(y)
    
    if multivar:
        #unstack the 3rd dimension and select the first element(energy load)
        y = y[:,idx]
    
    return X, y


# # Multivariable - Multiple Parallel Output LSTM

# In[7]:


###define an LSTM model
#takes in parallel inputs and outputs an equal number of parallel outputs
def lstm_multi_in_parallel_out(n_lags, n_hours, cells=50, learning_rate=5e-3):
    
    #define the model
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(cells, activation='relu', return_sequences=True, input_shape=(n_lags, n_hours)))
    model.add(keras.layers.LSTM(int(cells/2), activation='relu'))
    model.add(keras.layers.Dense(n_hours))
    
    #define the learning rate
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    
    #compile model
    model.compile(optimizer=optimizer, loss='mae')
    
    return model



def get_lstm_multivariable_data_3d(start='2015-01-01', stop='2015-01-05', n_lags=2, extra_lag=False, long_lag_step=7, max_lookback=30):

    #load in the prepared dataset
    all_data = pd.read_csv('./data/lstm/nn_dataset_2015_2018.csv', parse_dates=True, index_col=0)

    #select data time slice
    data = all_data[start: stop].copy()
    
    #reshape the energy load columns to prepare for minmax scaling
    energy = data['actual_load'].values.reshape(-1,1)
    
    #minmax scale the energy column
    scaler = MinMaxScaler().fit(energy)
    data_normd = scaler.transform(energy)
    
    #reset the energy column to the actual loads
    data['actual_load'] = data_normd.copy()
    
    #create single columns with time features
    data.loc[:,'year'] = data.index.year
    data.loc[:,'month'] = data.index.month
    data.loc[:,'day'] = data.index.day
    data.loc[:,'hours'] = data.index.hour
    
    hours_tup = [] 

    #for each unique hour isolate the features and dates
    for h in data.hours.unique():
        
        #boolean mask for each hour of the day
        hour = data[data.hours==h].copy()
        #drop the unneeded columns
        hour.drop(['year', 'month', 'day', 'hours'], axis=1, inplace=True)
        #reshape 2D into a 3D matrix for stacking
        hour = np.reshape(hour.values, (hour.shape[0], hour.shape[1], 1))
        #append each 3d slice into list
        hours_tup.append(hour)
    
    
    tup = tuple(hours_tup)
    
    #stack all the 3D arraysinto 1 single 3D array
    hours_stacked = np.dstack(tup)
    
    #make samples from hours stacked. result is 4D and 2D
    X_4d, y = split_sequences(hours_stacked, 
                           n_lags, 
                           extra_lag=extra_lag, 
                           long_lag_step=long_lag_step, 
                           max_step=max_lookback, 
                           idx=0, 
                           multivar=True)
    #X_4d, y = split_sequences(hours_stacked, n_lags, idx=0, multivar=True)
    
    
    X = []

    #flatten the 2nd and 3rd dimensions together to have a final array of samples, lags & features, hours
    for j in range(len(X_4d)):
        #reshape the inner dimensions
        n = X_4d[j].reshape(-1, hours_stacked.shape[-1])
        X.append(n)
    
    X = np.array(X)
    
    
    return X, y, scaler

# In[47]:


def run_multi_var_lstm_pipe(n_lags=2, n_crossvals=2, epochs=5, lr = 1e-3, extra_lag=False, long_lag_step=7, max_lookback=30, show_verbose=False, period_start = '2017-01-01', period_end = '2017-12-31'):

    
    n_hours = 24


    #load the inital data
    X_multi, y_multi, scaler_multi = get_lstm_multivariable_data_3d(start=period_start, 
                                                                    stop=period_end, 
                                                                    n_lags=n_lags, 
                                                                    extra_lag=extra_lag, 
                                                                    long_lag_step=long_lag_step, 
                                                                    max_lookback=max_lookback)


    n_features = X_multi.shape[1]


    if show_verbose:
        verbose = 1
    else:
        verbose = 0


    print('Crossvalidation run congifuration:')
    print('Number of crossvalidations: {}' .format(n_crossvals))
    print('Number of total feature vectors: {}' .format(n_features))
    #print('Date range from {} to {}'.format(period_start, period_end))


    print('Input shape {}'.format(X_multi.shape))
    print('Output shape {}'.format(y_multi.shape))


    #creates set sequences of the time series to cross validate on. 
    tscv = TimeSeriesSplit(n_splits=n_crossvals)

    # #initalize lists to capture the output
    predictions = []
    actuals = []


    #run the LSTM model on each of the time series splits
    for train, test in tscv.split(X_multi, y_multi):

        lstm_multi = lstm_multi_in_parallel_out(n_features, n_hours, learning_rate=lr)


        lstm_multi.fit(X_multi[train], y_multi[train], epochs=epochs, verbose=verbose, shuffle=False)

        predict = lstm_multi.predict(X_multi[test], verbose=True)


        #inverse transform the predictions and actual values
        prediction = scaler_multi.inverse_transform(predict)
        actual = scaler_multi.inverse_transform(y_multi[test].copy())

        #save the results in a list
        predictions.append(prediction)
        actuals.append(actual)


    #convert results to numpy array for easy manipulation
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    print(predictions.shape)
    print(actuals.shape)
    #calculate and display the crossvalidated mean average errors 
    mae = get_sample_total_mae(actuals, predictions)

    #print a selection of the cross validated predictions. See how the sample predictions evolved.
    inspect_cv_predictions(actuals, predictions)

    return predictions, actuals



########### RUN THE MODEL ON CALL OF THIS FILE
start_date = '2017-01-01'
stop_date = '2017-12-31'
max_lookback=30
long_lag_step=7
n_lags = 7
epochs=50

print('Running Crossvalidation LSTM example with the following parmaters')
print('Type: Univariate')
print('Inital lag {}'.format(n_lags))
print('Long lag step interval of {} days to a maximum of {} days'.format(long_lag_step, max_lookback))
print('Crossvalidation interval start {} and stop {}'.format(start_date, stop_date))
print('Training model on {} epochs'.format(epochs))


predictions, actuals = run_multi_var_lstm_pipe(n_lags=n_lags, 
                        n_crossvals=3, 
                        epochs=50, 
                        lr = 1e-3, 
                        extra_lag=True, 
                        long_lag_step=long_lag_step, 
                        max_lookback=max_lookback, 
                        show_verbose=False, 
                        period_start = '2017-01-01', 
                        period_end = '2017-12-31')

print('Saving outputs...')


prediction_list = predictions.tolist()
actual_list = actuals.tolist()

json_preds = "./results/lstm/multivariate/prediction.json" 
json_actual = "./results/lstm/multivariate/actuals.json"

json.dump(prediction_list, codecs.open(json_preds, 'w', encoding='utf-8'), sort_keys=True, indent=4)
json.dump(actual_list, codecs.open(json_actual, 'w', encoding='utf-8'), sort_keys=True, indent=4)

print('Saved at {}'.format(json_preds))


#cv_week_predictions(actuals, predictions, num_days=7, shift=0)
cv_week_predictions(actuals, predictions, num_days=3*7+1, shift=0)
