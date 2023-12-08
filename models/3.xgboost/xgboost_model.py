import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import xgboost as xgb
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

def custom_smape(A, F):
    return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def custom_rmse(A, F):
    return np.sqrt(((A - F) ** 2).mean())

def custom_mape(A,F):
    return np.mean(np.abs((A - F)/A))

def make_lags(ts, lags):
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, int(lags) + 1)
        },
        axis=1)

def xgboost_create(X_train, y_train, X_test, y_test):

    reg = xgb.XGBRegressor(n_estimators=200)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=500,
        	verbose=True)

    reg.save_model("xgb.json")

def xgboost_predict(input_data, num_lags):

    reg = xgb.XGBRegressor()
    reg.load_model("xgb.json")

    df = input_data.copy()
    predictions = []
    for i in range(7):
        prediction = reg.predict(df)
        predictions.append(prediction)
        temp = np.array(df.values[-1], dtype=float)
        temp = np.delete(temp, 0)
        temp = np.insert(temp, num_lags-1, prediction)
        for i in range(num_lags, 0, -1):
            df[f'y_lag_{i}'] = temp[num_lags-i]

    d = {'arrivals': pd.Series(predictions, index=range(df.index.values[-1], df.index.values[-1] + 7))}
    predictions_df = pd.DataFrame(data=d)

    return predictions_df

#Load Dataset
df = pd.read_csv("DailyArrivals.csv")
df.drop(columns=['date'], inplace=True)
df.arrivals = df.arrivals + 1.001

#Split dataset
split_size = 270
train, test = df[0:split_size], df[split_size:]

X_train, y_train = train.drop(columns='arrivals'), train.arrivals
X_test, y_test = test.drop(columns='arrivals'), test.arrivals

#Previous Timesteps to use as features
num_lags = 5

#Create training and testing datasets
if(num_lags):
    lags = make_lags(df.arrivals, num_lags)
    lags = lags.fillna(0.0)
    train_lags, test_lags = lags[0:split_size], lags[split_size:len(df)]

    for i in range(num_lags, 0, -1):
        X_train[f'y_lag_{i}'] = train_lags[f'y_lag_{i}']

    for i in range(num_lags, 0, -1):
        X_test[f'y_lag_{i}'] = test_lags[f'y_lag_{i}']

X_train = X_train.iloc[num_lags: , :]
y_train = y_train.iloc[num_lags:]

'''
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(X_train)
    print(X_test)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(y_train)
    print(y_test)
'''

xgboost_create(X_train, y_train, X_test, y_test)
counter = 0
total_rmse = 0
total_mape = 0
total_smape = 0
total_mae = 0
for i in range(270, len(df)-6, 7):
    y = df[i:i+7] # target values
    predictions = xgboost_predict(X_test.loc[[i]], num_lags)

    rmse = custom_rmse(y.arrivals, predictions.arrivals.values)
    mape = custom_mape(y.arrivals, predictions.arrivals.values)
    smape = custom_smape(y.arrivals, predictions.arrivals.values)
    mae = mean_absolute_error(y.arrivals, predictions.arrivals.values)

    total_rmse = total_rmse + rmse
    total_mape = total_mape + mape
    total_smape = total_smape + smape
    total_mae = total_mae + mae
    counter = counter + 1
    
    plt.plot(df.arrivals)
    plt.plot(predictions.arrivals)
    plt.show()
    
print(total_rmse/counter)
print(total_mape/counter)
print(total_smape/counter)
print(total_mae/counter)