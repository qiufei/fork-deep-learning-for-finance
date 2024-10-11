import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def data_preprocessing(data, num_lags, train_test_split):
    # Prepare the data for training
    x = []
    y = []
    for i in range(len(data) - num_lags):
        x.append(data[i:i + num_lags])
        y.append(data[i+ num_lags])
    # Convert the data to numpy arrays
    x = np.array(x)
    y = np.array(y)
    # Split the data into training and testing sets
    split_index = int(train_test_split * len(x))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]

    return x_train, y_train, x_test, y_test

def recursive_mpf(x_test, y_test, num_lags, model, architecture = 'MLP'):
    if architecture == 'MLP':
        # Latest values to use as inputs
        x_test = x_test[-1]
        x_test = np.reshape(x_test, (-1, 1))
        x_test = np.transpose(x_test)
        y_predicted = []
        for i in range(len(y_test)):     
            # Predict over the last x_test values
            predicted_value = model.predict(x_test)
            y_predicted = np.append(y_predicted, predicted_value)
            # Re-inserting the latest prediction into x_test array
            x_test = np.transpose(x_test)
            x_test = np.append(x_test, predicted_value)
            x_test = x_test[1:, ]
            x_test = np.reshape(x_test, (-1, 1))
            x_test = np.transpose(x_test)
        y_predicted = np.reshape(y_predicted, (-1, 1))
    elif architecture == 'LSTM':
        # Latest values to use as inputs
        x_test = x_test[-1]
        x_test = np.reshape(x_test, (-1, 1))
        x_test = np.transpose(x_test)
        x_test = x_test.reshape((-1, num_lags, 1))
        y_predicted = []
        for i in range(len(y_test)):     
            # Predict over the last x_test values
            predicted_value = model.predict(x_test)
            y_predicted = np.append(y_predicted, predicted_value)
            # Re-inserting the latest prediction into x_test array
            x_test = np.transpose(x_test)
            x_test = np.append(x_test, predicted_value)
            x_test = x_test[1:, ]
            x_test = np.reshape(x_test, (-1, 1))
            x_test = np.transpose(x_test)
            x_test = x_test.reshape((-1, num_lags, 1))  
        y_predicted = np.reshape(y_predicted, (-1, 1))
        
    return x_test, y_predicted

def direct_mpf(data, num_lags, train_test_split, forecast_horizon):
    x, y = [], []
    for i in range(len(data) - num_lags - forecast_horizon + 1):
        x.append(data[i:i + num_lags])
        y.append(data[i + num_lags:i + num_lags + forecast_horizon])
    x = np.array(x)
    y = np.array(y)   
    split_index = int(train_test_split * len(x))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:] 
    
    return x_train, y_train, x_test, y_test


def plot_train_test_values(window_total, window_train, y_train, y_test, y_predicted):
    window_test_predict = window_total - window_train
    # reshape 函数的第一个参数是需要重塑的数组 y_predicted，第二个参数是新的形状 (-1, 1)。在新的形状中，-1 表示这一维的大小将根据数组的总长度和其他维度的大小自动计算，以确保元素总数不变。1 表示第二维的大小固定为 1。
    y_predicted = np.reshape(y_predicted, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))
    
    plotting_time_series = np.zeros((window_total, 3))
    # ploting the Training data
    plotting_time_series[0:window_train, 0] = y_train[-window_train:]
    plotting_time_series[window_train:, 0] = np.nan
    # ploting the Test data
    plotting_time_series[:window_train, 1] = np.nan
    plotting_time_series[window_train:, 1] = y_test[0:window_test_predict, 0]
    # ploting the Predicted data
    plotting_time_series[:window_train, 2] = np.nan 
    plotting_time_series[window_train:, 2] = y_predicted[0:window_test_predict, 0] 
    
    plt.plot(plotting_time_series[:, 0], label = 'Training data', color = 'black', linewidth = 2.5)
    plt.plot(plotting_time_series[:, 1], label = 'Test data', color = 'black', linestyle = 'dashed', linewidth = 2)
    plt.plot(plotting_time_series[:, 2], label = 'Predicted data', color = 'red', linewidth = 1)
    plt.axvline(x = window_train, color = 'black', linestyle = '--', linewidth = 1)
    
    plt.grid()
    plt.legend()

def forecasting_threshold(predictions, threshold):
    for i in range(len(predictions)):
        if predictions[i] > threshold:
            predictions[i] = predictions[i]
        elif predictions[i] < -threshold:
            predictions[i] = predictions[i]
        else:
            predictions[i] = 0
    return predictions

def calculate_accuracy(predicted_returns, real_returns):
    predicted_returns = np.reshape(predicted_returns, (-1, 1))
    real_returns = np.reshape(real_returns, (-1, 1))
    hits = sum((np.sign(predicted_returns)) == np.sign(real_returns))
    total_samples = len(predicted_returns)
    accuracy = hits / total_samples
    
    return accuracy[0] * 100

def model_bias(predicted_returns):
    bullish_forecasts = np.sum(predicted_returns > 0)
    bearish_forecasts = np.sum(predicted_returns < 0)
    
    return bullish_forecasts / bearish_forecasts

def calculate_directional_accuracy(predicted_returns, real_returns):
    # Calculate differences between consecutive elements
    diff_predicted = np.diff(predicted_returns, axis = 0)
    diff_real = np.diff(real_returns, axis = 0)
    # Check if signs of differences are the same
    store = []  
    for i in range(len(predicted_returns)):
        try:            
            if np.sign(diff_predicted[i]) == np.sign(diff_real[i]):                
                store = np.append(store, 1)        
            elif np.sign(diff_predicted[i]) != np.sign(diff_real[i]):                
                store = np.append(store, 0)                  
        except IndexError:           
            pass       
    directional_accuracy = np.sum(store) / len(store)
        
    return directional_accuracy * 100


def multiple_data_preprocessing(data, train_test_split):
    data = add_column(data, 4)
    data[:, 1] = np.roll(data[:, 1], 1, axis = 0)
    data[:, 2] = np.roll(data[:, 2], 1, axis = 0)
    data[:, 3] = np.roll(data[:, 1], 1, axis = 0)
    data[:, 4] = np.roll(data[:, 2], 1, axis = 0)
    data[:, 5] = np.roll(data[:, 3], 1, axis = 0)
    data[:, 6] = np.roll(data[:, 4], 1, axis = 0)
    data = data[1:, ]
    x = data[:, 1:]
    y = data[:, 0]
    split_index = int(train_test_split * len(x))
    x_train = x[:split_index]
    y_train = y[:split_index]
    x_test = x[split_index:]
    y_test = y[split_index:]
    
    return x_train, y_train, x_test, y_test

def volatility(data, lookback, close, position):
    data = add_column(np.reshape(data, (-1, 1)), 1)
    for i in range(len(data)):   
        try:           
            data[i, position] = (data[i - lookback + 1:i + 1, close].std())  
        except IndexError:          
            pass   
    data = delete_row(data, lookback)    
     
    return data

def add_column(data, times): 
    for i in range(1, times + 1): 
        new = np.zeros((len(data), 1), dtype = float)     
        data = np.append(data, new, axis = 1)
        
    return data

def delete_column(data, index, times):  
    for i in range(1, times + 1):   
        data = np.delete(data, index, axis = 1)

    return data

def delete_row(data, number): 
    data = data[number:, ]
    
    return data

def compute_diff(data, period):
    data = add_column(np.reshape(data, (-1, 1)), 1)
    for i in range(len(data)):
        data[i, -1] = data[i, 0] - data[i - 1, 0]
    data = delete_column(data, 0, 1)
    
    return data

def ma(data, lookback, close, position):     
    data = add_column(data, 1)    
    for i in range(len(data)):           
            try:                
                data[i, position] = (data[i - lookback + 1:i + 1, close].mean())            
            except IndexError:               
                pass           
    data = delete_row(data, lookback)
    
    return data

def smoothed_ma(data, alpha, lookback, close, position):    
    lookback = (2 * lookback) - 1    
    alpha = alpha / (lookback + 1.0)    
    beta  = 1 - alpha    
    data = ma(data, lookback, close, position)
    data[lookback + 1, position] = (data[lookback + 1, close] * alpha) + (data[lookback, position] * beta)
    for i in range(lookback + 2, len(data)):
            try:
                data[i, position] = (data[i, close] * alpha) + (data[i - 1, position] * beta)
            except IndexError:
                pass
            
    return data

def rsi(data, lookback, close, position):
    data = add_column(data, 5)
    for i in range(len(data)): 
        data[i, position] = data[i, close] - data[i - 1, close]
    for i in range(len(data)):
        if data[i, position] > 0:
            data[i, position + 1] = data[i, position]
        elif data[i, position] < 0:       
            data[i, position + 2] = abs(data[i, position])         
    data = smoothed_ma(data, 2, lookback, position + 1, position + 3)
    data = smoothed_ma(data, 2, lookback, position + 2, position + 4)
    data[:, position + 5] = data[:, position + 3] / data[:, position + 4]   
    data[:, position + 6] = (100 - (100 / (1 + data[:, position + 5])))
    data = delete_column(data, position, 6)
    data = delete_row(data, lookback)

    return data