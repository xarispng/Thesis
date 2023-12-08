import numpy as np
import tensorflow as tf
from statsmodels.tsa.seasonal import seasonal_decompose
import rpy2.robjects as robjects
import os
import joblib

def hybrid(input_data):

    data = input_data.copy()
    data.arrivals = data.arrivals + 1

    decomposition = seasonal_decompose(data.arrivals, model='additive', extrapolate_trend='freq', period=10)

    ###########################################################################
    ########                      RESIDUAL                             ########
    ###########################################################################

    resid = decomposition.resid
    resid = resid[:,np.newaxis]

    #Scale the all of the data to be values between 0 and 1 
    scaler_resid = joblib.load('./website/models-and-data/resid_scaler.gz')
    scaled_resid = scaler_resid.transform(resid)

    #Reshape the data into 3-D array
    x_resid = np.reshape(scaled_resid, (1,1,20))

    model_resid = tf.keras.models.load_model('./website/models-and-data/hybrid_residual')

    for i in range(7):
        predictions = model_resid.predict(x_resid)

        temp = x_resid[0,0]
        temp = np.delete(temp, 0)
        temp = np.insert(temp, 19, predictions)
        x_resid = np.reshape(temp, (1,1,20))

    predictions_resid = np.reshape(x_resid[0,0,-7:], (7,1))
    predictions_resid = scaler_resid.inverse_transform(predictions_resid)

    ###########################################################################
    ########                       SEASONAL                            ########
    ###########################################################################

    seasonal = decomposition.seasonal
    seasonal = seasonal[:,np.newaxis]

    #Scale the all of the data to be values between 0 and 1
    scaler_seasonal = joblib.load('./website/models-and-data/seasonal_scaler.gz')
    scaled_seasonal = scaler_seasonal.transform(seasonal)

    #Reshape the data into 3-D array
    x_seasonal = np.reshape(scaled_seasonal, (1,1,20))

    model_seasonal = tf.keras.models.load_model('./website/models-and-data/hybrid_seasonal')

    for i in range(7):
        predictions = model_seasonal.predict(x_seasonal)

        temp = x_seasonal[0,0]
        temp = np.delete(temp, 0)
        temp = np.insert(temp, 19, predictions)
        x_seasonal = np.reshape(temp, (1,1,20))

    predictions_seasonal = np.reshape(x_seasonal[0,0,-7:], (7,1))
    predictions_seasonal = scaler_seasonal.inverse_transform(predictions_seasonal)

    ###########################################################################
    ########                           TREND                           ########
    ###########################################################################

    trend = decomposition.trend
    trend.to_csv(r'.\website\test.csv', index=False)

    robjects.r(''' arima <- function(f) {
                                library("readxl")
                                suppressPackageStartupMessages(library("forecast"))
                                DailyArrivals <- read.csv(".//website//test.csv")$trend
                                arrivals <- ts(log(DailyArrivals), frequency = f)
                                fit <- auto.arima(arrivals, D=1)
                                pred <- forecast(fit,h=7)$mean
                                }''')

    arima = robjects.r['arima']
    b = arima(5)
    c = arima(7)
    d = arima(10)

    b=(np.asarray(b)+np.asarray(c)+np.asarray(d))/3
    predictions_trend = np.exp(b)

    predictions_trend = np.reshape(predictions_trend, (7,1))

    os.remove("./website/test.csv")

    ###########################################################################
    ########                        RESULTS                            ########
    ###########################################################################

    predictions = predictions_resid + predictions_seasonal + predictions_trend - 1

    return predictions