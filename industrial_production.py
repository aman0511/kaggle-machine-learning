import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


##
## https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
##
## https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/
##
## http://www.seanabu.com/2016/03/22/time-series-seasonal-ARIMA-model-in-python/
##
## http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/
##


df = pd.read_csv('industrial_production.csv')


# df['IP'].plot()
# pd.tools.plotting.lag_plot(df['IP'])
# pd.tools.plotting.autocorrelation_plot(df['IP'])
# plot_acf(df['IP'], lags=120)
# plt.show()


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

ts = df['IP']

ts_log = np.log(df['IP'])

ts_minus_rm = (ts_log - ts_log.rolling(window=12).mean()).dropna()
ts_minus_emwa = (ts_log - ts_log.ewm(halflife=12).mean()).dropna()

ts_log_diff = (ts_log - ts_log.shift()).dropna()

ts_diff = (ts - ts.shift()).dropna()

#test_stationarity(df['IP'])
#test_stationarity(ts_log)
#test_stationarity(ts_minus_mean)
#test_stationarity(ts_minus_exp_mean)
#test_stationarity(ts_log_diff)

# test_stationarity(ts_diff)
#
# plt.show()
# exit()

# from statsmodels.tsa.seasonal import seasonal_decompose
# decomposition = seasonal_decompose(ts_log.values, freq=12)
#
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid

# plt.subplot(411)
# plt.plot(ts_log, label='Original (log)')
# plt.legend(loc='best')
# plt.subplot(412)
# plt.plot(trend, label='Trend')
# plt.legend(loc='best')
# plt.subplot(413)
# plt.plot(seasonal,label='Seasonality')
# plt.legend(loc='best')
# plt.subplot(414)
# plt.plot(residual, label='Residuals')
# plt.legend(loc='best')
# plt.tight_layout()

#test_stationarity(ts_log_decompose)

from statsmodels.tsa.stattools import acf, pacf

# lag_acf = acf(ts_log_diff, nlags=20)
# lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
# lag_acf = acf(ts_log_diff, nlags=5)
# lag_pacf = pacf(ts_log_diff, nlags=5, method='ols')


## Plot ACF:
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')

## Plot PACF:
# plt.subplot(122)
# plt.plot(lag_pacf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
# plt.title('Partial Autocorrelation Function')
# plt.tight_layout()

# plt.show()
# exit()

from statsmodels.tsa.arima_model import ARIMA

def fit_plot_arima(model, data):
    results = model.fit(disp=-1)
    plt.plot(data)
    plt.plot(results.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results.fittedvalues-data)**2))

# fit_plot_arima(ARIMA(ts_log_diff.values, order=(2, 0, 2)), ts_log_diff)
# plt.show()
# exit()

arima_model = ARIMA(ts.values, order=(2, 1, 1))
results = arima_model.fit(disp=-1)

predictions_diff = pd.Series(results.fittedvalues, copy=True)

predictions_diff_cumsum = predictions_diff.cumsum()
predictions = pd.Series(ts.ix[0], index=ts.index)
predictions = predictions.add(predictions_diff_cumsum, fill_value=0)

plt.plot(ts.values)
plt.plot(predictions)
#plt.plot(results.fittedvalues)

plt.show()

exit()

def fit_eval_model(order):
    aic = np.nan
    try:
        model = ARIMA(ts_log_diff.values,order=order).fit(disp=0)
        aic = model.aic
    except:
        pass
    return aic


pdq_vals = [(p,d,q) for p in range(4) for d in range(2) for q in range(4)]

from multiprocessing import Pool
p = Pool(4)
res = p.map(fit_eval_model, pdq_vals)
p.close()

result = pd.DataFrame(res,index=pdq_vals,columns=['aic']).sort_values(by=['aic'])

print result.head(n=3)