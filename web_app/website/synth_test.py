import numpy as np
import tensorflow as tf
import joblib

def synth(input_data):

    model = tf.keras.models.load_model('./website/models-and-data/lstm_synth')

    data = input_data.copy()
    data.arrivals = data.arrivals + 1

    #Scale the all of the data to be values between 0 and 1 
    scaler = joblib.load('./website/models-and-data/synth_scaler.gz')
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