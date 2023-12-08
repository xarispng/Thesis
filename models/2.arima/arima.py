import rpy2.robjects as robjects
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error

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
								DailyArrivals <- read.csv("test.csv")$arrivals
								arrivals <- ts(DailyArrivals, frequency = f)
								fit <- auto.arima(arrivals, d=1, D=1)
								pred <- forecast(fit,h=7)$mean
								x <- list("aic" = fit$aic, "pred" = pred)
								}''')

	arima = robjects.r['arima']
	
	arimas = []
	for i in [7]:
		arimas.append(arima(i))

	aics = []
	preds = []
	for i in range(len(arimas)):
		aics.append(np.array(arimas[i].rx('aic')).flatten())
		preds.append(np.array(arimas[i].rx('pred')).flatten())

	predictions = preds[np.argmin(aics)]
	predictions = np.exp(predictions)

	os.remove("test.csv")

	return predictions

df = pd.read_csv('DailyArrivals.csv')
df.arrivals = np.log(df.arrivals + 1.001)

counter = 0
total_rmse = 0
total_mape = 0
total_smape = 0
total_mae = 0
for i in range(270, len(df)-6, 7):
	full = df[:i+7]
	x = df[:i] # fit values
	y = df[i:i+7] # target values

	pred = arima_func(x) # predictions

	rmse = custom_rmse(np.exp(y.arrivals), pred)
	mape = custom_mape(np.exp(y.arrivals), pred)
	smape = custom_smape(np.exp(y.arrivals), pred)
	mae = mean_absolute_error(np.exp(y.arrivals), pred)
	
	d = {'arrivals': pd.Series(pred, index=y.index)}
	df_pred = pd.DataFrame(d)
	'''
	plt.plot(np.exp(full.arrivals), label='actual')
	plt.plot(df_pred.arrivals, label='forecast')
	plt.legend(loc='upper left', fontsize=8)
	plt.show()
	'''
	total_rmse = total_rmse + rmse
	total_mape = total_mape + mape
	total_smape = total_smape + smape
	total_mae = total_mae + mae
	counter = counter + 1

print(total_rmse/counter)
print(total_mape/counter)
print(total_smape/counter)
print(total_mae/counter)