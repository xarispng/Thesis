import numpy as np
import pandas as pd
import xgboost as xgb

def xgboost_func(input_data):

    reg = xgb.XGBRegressor()
    reg.load_model("./website/models-and-data/xgb.json")

    data = input_data.copy()
    data.arrivals = data.arrivals + 1

    df = pd.DataFrame()
    
    for i in range(1, 11):
        df[f'y_lag_{i}'] = data.values[i-1]

    for i in range(7):
        prediction = reg.predict(df)

        temp = np.array(df.values[-1], dtype=float)
        temp = np.delete(temp, 0)
        temp = np.insert(temp, 9, prediction)
        for i in range(1, 11):
            df[f'y_lag_{i}'] = temp[i-1]

    predictions = np.array(df.values[-1][-7:], dtype=float) - 1

    return predictions