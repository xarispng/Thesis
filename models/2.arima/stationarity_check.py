import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy as np

df = pd.read_csv('DailyArrivals.csv')
X = np.log(df.arrivals + 1.001).diff().dropna()

result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))