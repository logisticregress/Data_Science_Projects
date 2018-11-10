# Time Series
a collection of data science and statistical analysis projects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly as pl
import statsmodels
import statsmodels.api as sm
import itertools

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf

jc = 'jcdatapy.csv' # import the data file
data = pd.read_csv(jc) 
print(data)

plt.plot(data.index, data['Sales'])
plt.title('JC Sales Price')
plt.ylabel('Sales ($)');
plt.show()

# Sequence plot assuming unequal scales
plt.close('all')
fig, ax1 = plt.subplots()
t = data['Date']
ax1.plot(t, data['Sales'], 'b-', label='Sales')
fig.autofmt_xdate()
ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')


ax2 = ax1.twinx()
ax2.plot(t, data['Intent'], 'r-', label = 'Intent')

plt.title('Josh Cellars Sequence Plots')
fig.tight_layout()
plt.show()

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')  #parse the time data into the correct format
data = pd.read_csv('jcdatapy.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse) #load the parsed data
data #check that it looks right
data.index #dtype must return datetime64[ns]

ts_sales = data['Sales']  #convert to univariate series object

plt.plot(ts_sales)
plt.show()

rolmean = ts_sales.rolling(12).mean()
plt.plot(ts_sales)
plt.plot(rolmean, color='red')
plt.show()

y = data['Sales'] 

res = sm.tsa.statespace.SARIMAX(ts_sales, order=(2,1,1), seasonal_order=(1,0,0,12), enforce_stationarity=False, enforce_invertability=False).fit()

# in-sample-prediction and confidence bounds
pred = res.get_prediction(start=pd.to_datetime('2016-04-01'), end=pd.to_datetime('2018-07-01'),dynamic=True)

#Get forecast 12 periods into the future
pred_uc = res.get_forecast(steps=12)

#Get Confidence Intervals of forecasts
pred_ci = pred_uc.conf_int()


# Plot time series and long-term forecast
ax = y.plot(label='Observed', figsize=(16, 8), color='#006699');
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color='#ff0066')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='#ff0066', alpha=.25);
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left')
plt.show()

print(res.summary())
#print(ts_sales)
res.plot_diagnostics(figsize=(15,12))
plt.show()

pred = res.get_prediction(start=pd.to_datetime('2016-04-01'),dynamic=False)
pred_ci = pred.conf_int()

ax = y['2018':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='OSaF', alpha = .7)

ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:, 0], 
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.legend()
plt.show()
