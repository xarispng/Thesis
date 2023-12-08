import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error

def custom_smape(A, F):
    return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def custom_rmse(A, F):
    return np.sqrt(((A - F) ** 2).mean())

def custom_mape(A,F):
    return np.mean(np.abs((A - F)/A))

def uni(input_data, w, j):

    model = tf.keras.models.load_model(f'lstm_uni{j}')
    
    data = input_data.copy()

    df = pd.read_csv('DailyArrivals.csv')
    df = df[['arrivals']]
    df.arrivals = df.arrivals + 1.001

    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaler.fit(df)
    scaled_data = scaler.transform(data)

    test = np.reshape(scaled_data, (1,w,1))

    predictions = []
    for i in range(7):
        prediction = model.predict(test)
        predictions.append(prediction)
        temp = test.flatten()
        temp = np.delete(temp, 0)
        temp = np.insert(temp, w-1, prediction)
        test = np.reshape(temp, (1,w,1))

    predictions = scaler.inverse_transform([np.asarray(predictions).flatten()])

    return predictions

df = pd.read_csv('DailyArrivals.csv')
df.drop(columns=['date'], inplace=True)
df.arrivals = df.arrivals + 1.001

w = 15
TOTAL_rmse = 0
TOTAL_mape = 0
TOTAL_smape = 0
TOTAL_mae = 0
for j in range(10):
    counter = 0
    total_rmse = 0
    total_mape = 0
    total_smape = 0
    total_mae = 0
    for i in range(270, len(df)-6, 7):
        x = df[i-w:i] # it values
        y = df[i:i+7] # target values

        pred = uni(x,w,j).flatten() # predictions

        rmse = custom_rmse(y.arrivals, pred)
        mape = custom_mape(y.arrivals, pred)
        smape = custom_smape(y.arrivals, pred)
        mae = mean_absolute_error(y.arrivals, pred)

        total_rmse = total_rmse + rmse
        total_mape = total_mape + mape
        total_smape = total_smape + smape
        total_mae = total_mae + mae
        counter = counter + 1
        '''
        d = {'arrivals': pd.Series(pred, index=y.arrivals.index)}
        predictions = pd.DataFrame(data=d)
        plt.plot(df.arrivals)
        plt.plot(predictions.arrivals)
        plt.show()
        '''
    TOTAL_rmse = TOTAL_rmse + total_rmse/counter
    TOTAL_mape = TOTAL_mape + total_mape/counter
    TOTAL_smape = TOTAL_smape + total_smape/counter
    TOTAL_mae = TOTAL_mae + total_mae/counter

print(TOTAL_rmse/10)
print(TOTAL_mape/10)
print(TOTAL_smape/10)
print(TOTAL_mae/10)