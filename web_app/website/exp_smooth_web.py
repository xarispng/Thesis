import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings("ignore")

def expsmooth_func(input_data):

    data = input_data.copy()
    data.arrivals = data.arrivals + 1.001
    data.arrivals = np.log(data.arrivals)

    pred = []
    t1 = ''
    d1 = True
    s1 = ''
    p1 = 0
    b1 = True
    r1 = True
    best_aic = 1000
    t_params = ['add', 'mul'] #trend
    d_params = [True, False] #damped_trend
    s_params = ['add', 'mul'] #seasonal
    p_params = [5, 7, 10] #seasonal_periods
    b_params = [True, False] #use_boxcox
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
                            yhat = model_fit.predict(20, 20+6)
                            if model_fit.aic < best_aic:
                                best_aic = model_fit.aic
                                t1,d1,s1,p1,b1,r1 = t,d,s,p,b,r
                                pred = np.exp(yhat)

    predictions = pred.to_numpy()

    return predictions