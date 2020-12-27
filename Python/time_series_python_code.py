#importing data for time series 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly as pl
import statsmodels
import statsmodels.api as sm
import itertools

#modules for time series analysis
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from datetime import datetime
from statsmodels.tsa.stattools import acf, pacf

# Import the time series data
jc = 'jcdatapy.csv'
data = pd.read_csv(jc, index_col=0)
data

# Create the initial sequence plot 
plt.plot(data.index, data['Sales'])
plt.title('JC Sales Price')
plt.ylabel('Sales ($)');
plt.show()

# Sequence plot assuming equal scales 
plt.figure(figsize=(10,8))
plt.plot(data['Date'], data['Sales'], 'b-', label = 'Sales')
plt.plot(data['Date'], data['Intent'], 'r-', label = 'Intent')
plt.legend();


# Sequence plot assuming unequal scales
plt.close('all')
fig, ax1 = plt.subplots()
t = data['Date']
ax1.plot(t, data['Sales'], 'b-', label='Sales')
fig.autofmt_xdate()
ax1.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')


ax2 = ax1.twinx()
ax2.plot(t, data['Intent'], 'r-', label = 'Intent')

plt.title('JC Sequence Plots')
fig.tight_layout()
plt.show()

#Plots using Plotly (currently can't make this work) 
plotly.tools.set_credentials_file(username='rclukey', api_key='••••••••••')



#time Series analysis

dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')  #parse the time data into the correct format
data = pd.read_csv('jcdatapy.csv', parse_dates=['Date'], index_col='Date',date_parser=dateparse) #load the parsed data
data #check that it looks right
data.index #dtype must return datetime64[ns]

ts_sales = data['Sales']  #convert to univariate series object

#Sequence plot
plt.plot(ts_sales)
plt.show()

#Check for stationarity with the Dickey-Fuller Test

def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()  #  rolmean = ts_sales.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')  # orig = plt.plot(ts_sales, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')  # mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Perform Dickey-Fuller test:
    #print 'Results of Dickey-Fuller Test:'
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)


#make a graph that shows the moving average
rolmean = ts_sales.rolling(12).mean()
plt.plot(ts_sales)
plt.plot(rolmean, color='red')
plt.show()


#differencing the Data
ts_sales_D = ts_sales - ts_sales.shift()
plt.plot(ts_sales_D)
plt.show()


#Testing again for stationarity
ts_sales_D.dropna(inplace=True)
dftest = adfuller(ts_sales_D, autolag='AIC')


#Lag ACF
lag_acf = acf(ts_sales_D, nlags=20)
lag_pacf = pacf(ts_sales_D, nlags=20, method='ols')


plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_sales_D)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_sales_D)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_sales_D)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_sales_D)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

plt.show()




#using SARIMAX (this actually works)

y = data['Sales'] 

res = sm.tsa.statespace.SARIMAX(ts_sales, order=(1,1,1), seasonal_order=(1,0,0,12), enforce_stationarity=True, enforce_invertability=True).fit()

# in-sample-prediction and confidence bounds
pred = res.get_prediction(start=pd.to_datetime('2016-04-01'), end=pd.to_datetime('2018-07-01'),dynamic=True)

#Get forecast 12 periods into the future
pred_uc = res.get_forecast(steps=12)

#Get Confidence Intervals of forecasts
pred_ci = pred_uc.conf_int()


#Plot time series and long-term forecast
ax = y.plot(label='Observed', figsize=(16, 8), color='#006699');
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', color='#ff0066')
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='#ff0066', alpha=.25);
ax.set_xlabel('Date');
ax.set_ylabel('Sales');
plt.legend(loc='upper left')
plt.show()








