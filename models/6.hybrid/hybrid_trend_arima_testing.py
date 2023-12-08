import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.tsa.seasonal import seasonal_decompose

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

df = pd.read_csv('DailyArrivals.csv')
df.arrivals = df.arrivals + 1.001

decomposition = seasonal_decompose(df.arrivals, model='additive', extrapolate_trend='freq', period=30)

trend = decomposition.trend
trend = np.log(trend)
seasonal = decomposition.seasonal
resid = decomposition.resid

counter = 0
total_rmse = 0
total_mape = 0
for i in range(270, len(df)-6, 7):
	full = trend[:i+7]
	x = trend[:i] # fit values
	y = trend[i:i+7] # target values

	pred = arima_func(x) # predictions

	rmse = custom_rmse(np.exp(y.values), pred)
	mape = custom_mape(np.exp(y.values), pred)
	smape = custom_smape(np.exp(y.values), pred)
	
	d = {'arrivals': pd.Series(pred, index=y.index)}
	df_pred = pd.DataFrame(d)

	print(df_pred)
	print(np.exp(y.values))
	
	plt.plot(np.exp(full.values), label='actual')
	plt.plot(df_pred.arrivals, label='forecast')
	plt.legend(loc='upper left', fontsize=8)
	plt.show()
	
	total_rmse = total_rmse + rmse
	total_mape = total_mape + mape
	counter = counter + 1

print(total_rmse/counter)
print(total_mape/counter)