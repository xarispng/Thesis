import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

def custom_smape(A, F):
	return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def custom_rmse(A, F):
	return np.sqrt(((A - F) ** 2).mean())

def custom_mape(A,F):
	return np.mean(np.abs((A - F)/A))

def arima_func(input_data):

	data = input_data.copy()

	data.to_csv(r'test.csv', index=False)

	robjects.r(''' arima <- function(f) {
								library("readxl")
								suppressPackageStartupMessages(library("forecast"))
								DailyArrivals <- read.csv("test.csv")$trend
								arrivals <- ts(DailyArrivals, frequency = f)
								fit <- auto.arima(arrivals, d=1, D=1)
								pred <- forecast(fit,h=7)$mean
								}''')

	arima = robjects.r['arima']
	b = arima(7)

	predictions = np.exp(b)

	os.remove("test.csv")

	return predictions

def seasonal_lstm(input_data, w, seasonal):

    model = tf.keras.models.load_model('lstm_seasonal')
    
    data = input_data.copy()
    data = data[:,np.newaxis]

    #Scale the all of the data to be values between 0 and 1 
    seasonal = seasonal[:,np.newaxis]
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaler.fit(seasonal)

    scaled_data = scaler.transform(data)

    test = np.reshape(scaled_data, (1,1,w))

    predictions = []
    for i in range(7):
        prediction = model.predict(test)
        predictions.append(prediction)
        temp = test.flatten()
        temp = np.delete(temp, 0)
        temp = np.insert(temp, w-1, prediction)
        test = np.reshape(temp, (1,1,w))

    predictions = scaler.inverse_transform([np.asarray(predictions).flatten()])

    return predictions

def resid_lstm(input_data, w, resid):

    model = tf.keras.models.load_model('lstm_resid')
    
    data = input_data.copy()
    data = data[:,np.newaxis]

    #Scale the all of the data to be values between 0 and 1
    resid = resid[:,np.newaxis]
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaler.fit(resid)
    scaled_data = scaler.transform(data)

    test = np.reshape(scaled_data, (1,1,w))

    predictions = []
    for i in range(7):
        prediction = model.predict(test)
        predictions.append(prediction)
        temp = test.flatten()
        temp = np.delete(temp, 0)
        temp = np.insert(temp, w-1, prediction)
        test = np.reshape(temp, (1,1,w))

    predictions = scaler.inverse_transform([np.asarray(predictions).flatten()])

    return predictions

df = pd.read_csv('DailyArrivals.csv')
df.arrivals = df.arrivals + 1.001

decomposition = seasonal_decompose(df.arrivals, model='additive', extrapolate_trend='freq', period=30)

trend = decomposition.trend
trend = np.log(trend)
seasonal = decomposition.seasonal
resid = decomposition.resid

w = 15
counter = 0
total_rmse = 0
total_mape = 0
total_smape = 0
total_mae = 0
for i in range(270, len(df)-6, 7):
    full = df[:i+7]
    target = df[i:i+7] # target values

    #####  ARIMA TREND #####
    trend_x = trend[:i] # fit values
    trend_pred = arima_func(trend_x) # predictions

    #####  LSTM SEASONAL #####
    seasonal_x = seasonal[i-w:i] # fit values
    seasonal_pred = seasonal_lstm(seasonal_x,w,seasonal).flatten() # predictions

    #####  LSTM RESIDUAL #####
    resid_x = resid[i-w:i] # fit values
    resid_pred = resid_lstm(resid_x,w,resid).flatten() # predictions

    pred = trend_pred + seasonal_pred + resid_pred
    
    d = {'arrivals': pd.Series(pred, index=target.index)}
    df_pred = pd.DataFrame(d)
    '''
    plt.plot(full.arrivals, label='actual')
    plt.plot(df_pred.arrivals, label='forecast')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    '''
    rmse = custom_rmse(target.arrivals, df_pred.arrivals)
    mape = custom_mape(target.arrivals, df_pred.arrivals)
    smape = custom_smape(target.arrivals, df_pred.arrivals)
    mae = mean_absolute_error(target.arrivals, df_pred.arrivals)

    total_rmse = total_rmse + rmse
    total_mape = total_mape + mape
    total_smape = total_smape + smape
    total_mae = total_mae + mae
    counter = counter + 1

print(total_rmse/counter)
print(total_mape/counter)
print(total_smape/counter)
print(total_mae/counter)