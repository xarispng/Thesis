import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler

def uni(input_data):

    model = tf.keras.models.load_model('./website/models-and-data/lstm_uni')
    
    data = input_data.copy()
    data.arrivals = data.arrivals + 1

    df = pd.read_csv('./website/models-and-data/MarinaDataset.csv')
    df = df[['arrivals']]
    df.arrivals = df.arrivals + 1

    #Scale the all of the data to be values between 0 and 1 
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    scaler.fit(df)
    scaled_data = scaler.transform(data)

    test = np.reshape(scaled_data, (1,1,20))

    for i in range(7):
        prediction = model.predict(test)

        temp = test[0,0]
        temp = np.delete(temp, 0)
        temp = np.insert(temp, 19, prediction)
        test = np.reshape(temp, (1,1,20))

    predictions = np.reshape(test[0,0,-7:], (7,1))

    predictions = scaler.inverse_transform(predictions) - 1

    return predictions