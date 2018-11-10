Library(forecast)
jd_ts = ts(jd, frequency = 12)  #convert dataset to time series
autoplot(jd_ts)

#Seasonal decomposition
sfit = stl(jd_ts[,3], s.window="period")
plot(sfit)


#univariate time series
fit = auto.arima(jd$Sales)
jd_pred = forecast(fit, n = 12)
plot(jd_pred)

#multivariate time series 
exo = cbind(jd$Intent, jd$Spend)   # create the data matrix for predictors 
fitx = auto.arima(jd$Sales, xreg = exo)

#Summary Table
summary(fitx)

#Check Residuals 
checkresiduals(fitx)

#Forecast (12 months ahead)
fcast = forecast(fitx, xreg=jd1)
autoplot(fcast)

#Plot 
autoplot(fcast)
