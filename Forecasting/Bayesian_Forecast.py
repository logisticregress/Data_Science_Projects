def bayes_forecast(data, brand, Y, prior, var_names = [], forecast_start=None, forecast_end=None, discount=.98, return_all=False, X=None, 
        seasPeriods=[52], seasHarmComponents=[[1,2]], nsamps=3000, ntrend=2):
    data = data.copy()
    data = data[data['Brand Name'] == brand]
    start = 0
    Y = data[Y].values[start:]
    if not var_names: 
        X = None
    else:
        X = data[var_names].values[start:]
    family = 'normal'
    prior_length = prior
    if forecast_start is None: forecast_start = 0
    if forecast_end is None: forecast_end = len(Y)-1
    s0 = 0.01
    
    if ntrend == 1:
        deltrend = discount
    elif ntrend == 2:
        deltrend = [discount, 0.97]
     
    dates = data.Date[start:]
    date = pd.to_datetime(data.Date).dt.date
    k = 1
    ## Run the analysis
    mod, samples = analysis(Y, X=X,
    family=family,
    forecast_start=forecast_start,
    forecast_end=forecast_end,
    seasPeriods=seasPeriods,
    nsamps=nsamps,
    seasHarmComponents=seasHarmComponents,
    ntrend=ntrend,
    k=k,
    prior_length=prior_length,
    deltrend=deltrend,
    delregn=discount,
    delVar=discount,
    s0=s0)
    
    # set confidence interval
    credible_interval=95
    alpha = (100-credible_interval)/2
    upper=np.percentile(samples, [100-alpha], axis=0).reshape(-1)
    lower=np.percentile(samples, [alpha], axis=0).reshape(-1)
    
    ## If return_all=True, return a lot of information
    if return_all:
        return mod, samples, Y[forecast_start:forecast_end+1], date, prior_length, lower, upper

    forecast = median(samples)

        
        
        
def future(mod, h, date):
    b_samples = mod.forecast_path(h, nsamps=2000)
    b_forecast = median(b_samples)
    f_date = pd.date_range(start=(dates.max()+pd.DateOffset(1)), freq='D', periods=h)
    # set confidence interval
    credible_interval=95
    alpha = (100-credible_interval)/2
    b_upper=np.percentile(b_samples, [100-alpha], axis=0).reshape(-1)
    b_lower=np.percentile(b_samples, [alpha], axis=0).reshape(-1)
    
    return f_date, b_forecast, b_lower, b_upper 


def f_plot():
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Bayesian Forecast Showing'+' '+str(h)+' '+'Month Forecast for '+brand, fontsize=20)
    plt.title('MAPE: '+str(mape), fontsize=16)
    plt.plot(dates[pl:], forecast[pl:], c='blue', linewidth=.5, linestyle='--')
    plt.plot(dates[pl:], Y_plot[pl:], c='black', linewidth=.5)
    plt.plot(f_date, b_forecast, c='blue')
    plt.fill_between(dates[pl:], lower[pl:], upper[pl:], color='blue', alpha=.1)
    plt.fill_between(f_date, b_lower, b_upper, color='blue', alpha=.33)
    plt.xlabel('date', fontsize=16)
    plt.ylabel(y+' Score', fontsize=16)
        
        
