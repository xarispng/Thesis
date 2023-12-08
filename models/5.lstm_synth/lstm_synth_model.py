import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import joblib

def custom_smape(A, F):
    return 1/len(A) * np.sum(2 * np.abs(F-A) / (np.abs(A) + np.abs(F)))

def custom_rmse(A, F):
    return np.sqrt(((A - F) ** 2).mean())

def custom_mape(A,F):
    return np.mean((np.abs(A-F))/np.abs(A))

for j in range(10):
    W = 15
    training_dataset_length = 270

    df = pd.read_csv('DailyArrivals.csv')
    df.drop(columns=['date'], inplace=True)
    df.arrivals = df.arrivals + 1.001

    total_df = df.iloc[:training_dataset_length]
    for i in range(29):
        new_df = pd.read_csv(f'DailyArrivals{i+1}.csv')
        total_df = pd.concat([total_df, new_df])
    total_df.reset_index(drop=True, inplace=True)

    total_df[total_df['arrivals'].isna()==True] = total_df.mean()

    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(total_df)
    joblib.dump(scaler, 'scaler.gz')

    #Splitting the data
    x_train=[]
    y_train = []

    for x in range(30):
        for i in range((x*training_dataset_length)+W, training_dataset_length*(x+1)):
            x_train.append(scaled_data[i-W:i,0])
            y_train.append(scaled_data[i,0])

    #Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data into 3-D array
    x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))

    # The LSTM architecture
    model = Sequential()
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))

    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer=opt, loss='mae')

    es = EarlyStopping(monitor='loss', patience=100)
    history = model.fit(x_train, y_train, epochs=2500, verbose=2, batch_size=x_train.shape[0], callbacks=[es], shuffle=False)

    model.save(f'lstm_synth{j}')