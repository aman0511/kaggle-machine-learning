import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf



df = pd.read_csv('lenex-corporation-shipment-of-ra.csv')

#print(df.head())

#df['NumRadios'].plot()
#pd.tools.plotting.lag_plot(df['NumRadios'])
#pd.tools.plotting.autocorrelation_plot(df['NumRadios'])
#plot_acf(df['NumRadios'], lags=12)
#plt.show()
#exit()

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


ts = df['NumRadios']
ts_diff = (ts - ts.shift()).dropna()

#test_stationarity(ts_diff)

#plt.show()
#exit()


from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_diff, nlags=12)
lag_pacf = pacf(ts_diff, nlags=12, method='ols')

## Plot ACF:
# plt.subplot(121)
# plt.plot(lag_acf)
# plt.axhline(y=0,linestyle='--',color='gray')
# plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
# plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')
# plt.title('Autocorrelation Function')
#
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

from statsmodels.tsa.arima_model import ARIMA

# def fit_plot_arima(model, data):
#     results = model.fit(disp=-1)
#     plt.plot(data)
#     plt.plot(results.fittedvalues, color='red')
#     plt.title('RSS: %.4f'% sum((results.fittedvalues-data)**2))
#
# fit_plot_arima(ARIMA(ts_diff.values, order=(1, 0, 1)), ts_diff)
# plt.show()
# exit()

# arima_model = ARIMA(ts_diff.values, order=(1, 0, 1))

arima_model = ARIMA(ts_diff.values, order=(1, 1, 3))
results = arima_model.fit(disp=-1)

predictions_diff = pd.Series(results.fittedvalues, copy=True)

predictions_diff_cumsum = predictions_diff.cumsum()
predictions = pd.Series(ts.ix[0], index=ts.index).add(predictions_diff_cumsum, fill_value=0)

plt.plot(df['NumRadios'])
plt.plot(predictions)

plt.show()

exit()


def fit_eval_model(order):
    aic = np.nan
    try:
        model = ARIMA(ts_diff.values,order=order).fit(disp=0)
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

print result.head(n=75)