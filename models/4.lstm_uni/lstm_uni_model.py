import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

def smape(A, F):
    return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def custom_rmse(A, F):
    return np.sqrt(((A - F) ** 2).mean())

def custom_mape(A,F):
    return np.mean(np.abs((A - F)/A))

for j in range(10):
    W = 15
    training_dataset_length = 270

    df = pd.read_csv('DailyArrivals.csv')
    df.arrivals = df.arrivals + 1.001
    df.drop(columns='date', inplace=True)

    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaled_data = scaler.fit_transform(df)

    #Train data set
    train_data = scaled_data[0:training_dataset_length, : ]

    #Splitting the data
    x_train=[]
    y_train = []

    for i in range(W, len(train_data)):
        x_train.append(train_data[i-W:i,0])
        y_train.append(train_data[i,0])

    #Convert to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)

    #Reshape the data into 3-D array
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    #Test data set
    test_data = scaled_data[training_dataset_length - W: , : ]

    #splitting the x_test and y_test data sets
    x_test = []
    y_test =  scaled_data[training_dataset_length : , : ] 

    for i in range(W,len(test_data)):
        x_test.append(test_data[i-W:i,0])
        
    #Convert x_test to a numpy array 
    x_test = np.array(x_test)

    #Reshape the data into 3-D array
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    # The LSTM architecture
    model = Sequential()
    model.add(LSTM(units=50))
    model.add(Dropout(0.1))
    model.add(Dense(1))

    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer=opt, loss='mae')

    es = EarlyStopping(monitor='loss', patience=100)
    history = model.fit(x_train, y_train, epochs=2500, verbose=2, callbacks=[es], shuffle=False, batch_size=64, validation_data=(x_test, y_test))

    model.save(f'lstm_uni{j}')