import rpy2.robjects as robjects
import numpy as np
import os

def arima_func(input_data):

	data = input_data.copy()
	data.arrivals = data.arrivals + 1.001

	data.to_csv(r'.\website\test.csv', index=False)

	robjects.r(''' arima <- function(f) {
								library("readxl")
								suppressPackageStartupMessages(library("forecast"))
								DailyArrivals <- read.csv(".//website//test.csv")$arrivals
								arrivals <- ts(log(DailyArrivals), frequency = f)
								fit <- auto.arima(arrivals, D=1)
								pred <- forecast(fit,h=7)$mean
								}''')

	arima = robjects.r['arima']
	b = arima(5)
	c = arima(7)
	d = arima(10)

	b=(np.asarray(b)+np.asarray(c)+np.asarray(d))/3
	predictions = np.exp(b) - 1

	os.remove("./website/test.csv")

	return predictions