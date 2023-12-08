import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import random

df = pd.read_csv('DailyArrivals.csv')
df.drop(columns=['date'], inplace=True)
df.arrivals = df.arrivals + 1.001

bc_series, fitted_lambda = boxcox(df.arrivals)

d = {'arrivals': pd.Series(bc_series, index=df.index)}
boxcox_df = pd.DataFrame(data=d)

decomposition = seasonal_decompose(boxcox_df.iloc[:270], model='additive', extrapolate_trend='freq', period=30)
#decomposition.plot()
#plt.show()

resid = decomposition.resid
trend = decomposition.trend
seasonal = decomposition.seasonal

blocks = []
block_size = 30
for i in range(270-block_size+1):
    blocks.append(resid[i:i+block_size])

for i in range(29):
    sample_list = random.choices(blocks, k=9)

    synth_resid = sample_list[0]
    for j in range(1, len(sample_list)):
        synth_resid = pd.concat([synth_resid, sample_list[j]])
    synth_resid.reset_index(drop=True, inplace=True)

    synth = trend + seasonal + synth_resid
    synth = inv_boxcox(synth,fitted_lambda)

    d = {'arrivals': pd.Series(synth, index=range(270))}
    synth = pd.DataFrame(data=d)

    synth.to_csv(f'.\DailyArrivals{i+1}.csv', index=False)
    
    '''
    plt.plot(df.iloc[:270])
    plt.plot(synth)
    plt.show()
    '''