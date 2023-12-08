import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.callbacks import EarlyStopping

W = 15
training_dataset_length = 270

df = pd.read_csv('DailyArrivals.csv')
df.arrivals = df.arrivals + 1.001

decomposition = seasonal_decompose(df.arrivals, model='additive', extrapolate_trend='freq', period=30)

seasonal = decomposition.seasonal
seasonal = seasonal[:,np.newaxis]

#Scale the all of the data to be values between 0 and 1 
scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(seasonal)

#Train data set
train_data = scaled_data

#Splitting the data
x_train=[]
y_train = []

for i in range(W, len(train_data)):
    x_train.append(train_data[i-W:i,0])
    y_train.append(train_data[i,0])

#Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data into 3-D array
x_train = np.reshape(x_train, (x_train.shape[0],1,x_train.shape[1]))

# Initialising the RNN
model = Sequential()
model.add(LSTM(units=50))
model.add(Dropout(0.1))
model.add(Dense(units=1))

opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
model.compile(optimizer=opt, loss='mse')

es = EarlyStopping(monitor='loss', patience=100)
history = model.fit(x_train, y_train, epochs=10000, verbose=2, batch_size=64, callbacks=[es], shuffle=False)

model.save('lstm_seasonal')