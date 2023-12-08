import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")

def custom_smape(A, F):
    return 1/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

def custom_rmse(A, F):
    return np.sqrt(((A - F) ** 2).mean())

def custom_mape(A,F):
    return np.mean(np.abs((A - F)/A))

def expsmooth_func(input_data, length):

    data = input_data.copy()
    #data.arrivals = np.log(data.arrivals)

    pred = []
    t1 = ''
    d1 = True
    s1 = ''
    p1 = 0
    b1 = True
    r1 = True
    best_aic = 10000000
    t_params = ['add']#, 'mul'] #trend
    d_params = [True, False] #damped_trend
    s_params = ['add']#, 'mul'] #seasonal
    p_params = [7, 30, 180] #seasonal_periods
    b_params = [True] #use_boxcox
    r_params = [False] #remove_bias
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            model = ExponentialSmoothing(data.arrivals, trend=t, damped_trend=d, seasonal=s, seasonal_periods=p, use_boxcox=b, initialization_method=None)
                            model_fit = model.fit(optimized=True, remove_bias=r)
                            yhat = model_fit.predict(length, length+6)
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                t1,d1,s1,p1,b1,r1 = t,d,s,p,b,r
                                pred = yhat #np.exp(yhat)

    #predictions = pred.to_numpy()

    return pred

df = pd.read_csv('DailyArrivals.csv')
df.arrivals = df.arrivals + 1.001

counter = 0
total_rmse = 0
total_mape = 0
total_smape = 0
total_mae = 0
for i in range(270, len(df)-6, 7):
    full = df[:i+7]
    x = df[:i] # fit values
    y = df[i:i+7] # target values

    pred = expsmooth_func(x, i) # predictions

    rmse = custom_rmse(y.arrivals, pred)
    mape = custom_mape(y.arrivals, pred)
    smape = custom_smape(y.arrivals, pred)
    mae = mean_absolute_error(y.arrivals, pred)
    '''
    plt.plot(full.arrivals, label='actual')
    plt.plot(pred.fillna(1), label='forecast')
    plt.legend(loc='upper left', fontsize=8)
    plt.show()
    '''
    total_rmse = total_rmse + rmse
    total_mape = total_mape + mape
    total_smape = total_smape + smape
    total_mae = total_mae + mae
    counter = counter + 1

print(total_rmse/counter)
print(total_mape/counter)
print(total_smape/counter)
print(total_mae/counter)