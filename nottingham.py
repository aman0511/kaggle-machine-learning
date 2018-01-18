import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


df = pd.read_csv('mean-monthly-air-temperature-deg.csv')


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

    #Perform Dickey-Fuller test:
    print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print dfoutput


ts = df['NottinghamTemp']

ts2 = ts - ts.mean()

ts_year = (ts - ts.shift(periods=12)).dropna()

# test_stationarity(ts)
# test_stationarity(ts_year)

# ts_year.plot()
#plot_acf(ts, lags=36)
#plt.show()
#exit()


from statsmodels.tsa.stattools import acf, pacf

# lag_acf = acf(ts_diff, nlags=12)
# lag_pacf = pacf(ts_diff, nlags=12, method='ols')

## Plot ACF:
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')

## Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()
#
# plt.show()
# exit()

from statsmodels.tsa.statespace.sarimax import SARIMAX


sarimax_mdl = SARIMAX(ts2.values, order=(1, 0, 1), seasonal_order=(1, 0, 0, 12))
results = sarimax_mdl.fit(disp=-1)

plt.plot(ts2.values)
plt.plot(results.fittedvalues)
plt.show()




# exit()
#
# def fit_eval_model(order):
#     aic = np.nan
#     try:
#         model = ARIMA(ts_year.values, order=order).fit(disp=0)
#         aic = model.aic
#     except:
#         pass
#     return aic
#
# pdq_vals = [(p,d,q) for p in range(5) for d in range(1) for q in range(5)]
#
# from multiprocessing import Pool
# p = Pool(4)
# res = p.map(fit_eval_model, pdq_vals)
# p.close()
#
# result = pd.DataFrame(res, index=pdq_vals, columns=['aic']).sort_values(by=['aic'])
#
# print result.head(n=75)
