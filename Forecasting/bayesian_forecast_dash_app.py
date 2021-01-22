# -*- coding: utf-8 -*-
import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.exceptions import PreventUpdate
import psycopg2
import gunicorn
import joblib
import base64
from io import BytesIO
import datetime
import os.path
import io
import plotly
from dash.dependencies import Input, Output, State
from os import listdir
from os.path import isfile, join
import pmdarima as pmd
import pandas as pd
import numpy as np
import scipy.stats as st
import plotly.graph_objs as go
import sqlalchemy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, f1_score
# these need to be added to requirements.txt
#import boto3
#from botocore.client import Config
# these are for PyBats -- need to add to requirements.textimport pybats
import pybats
from pybats.analysis import analysis
from pybats.point_forecast import median, mean
#from pybats.plot import plot_data_forecast, ax_style



app = dash.Dash(__name__)

application = app.server

# Bayesian Forecasting Function
def test_variables(data, var_names = [], river_lags=[1], off_river_lags = [1],
                   forecast_start=None, forecast_end=None, discount=.98,
                   return_all = False, Y=['River'],
                   seasPeriods=[5], seasHarmComponents=[[1,2]],
                   nsamps=1000, ntrend=2, binary=False):

    data = data.copy()

    ## Define the start point. Can't start at time 0 in the dataset, because predictors are lagged values.
    start_orl = max(off_river_lags) if len(off_river_lags) > 0 else 0
    start_rl = max(river_lags) if len(river_lags) > 0 else 0
    start = max(start_rl, start_orl)


    ## Select Y variable
    if binary:
        data[Y] = (data[Y].diff() > 0).astype(int)
        Y = data[Y].values[start:]
        family = 'bernoulli'
    else:
        Y = data[Y].values[start:]
        family = 'normal'

    ## Select X variables
    X_vars = data[var_names].values[start:]

    if len(river_lags) > 0:
        X_river_lags = np.c_[[data.River.values[start-l:-l] for l in river_lags]].T
    else:
        X_river_lags = data[[]].values[start:]

    if len(off_river_lags) > 0:
        X_offriver_lags = np.c_[[data.Off_River.values[start-l:-l] for l in off_river_lags]].T
    else:
        X_offriver_lags = data[[]].values[start:]

    X = np.c_[X_river_lags, X_offriver_lags, X_vars]

    ## Select the dates
    dates = data.Date[start:]

    ## Set any parameters that aren't arguments passed into the function
    k = 1
    prior_length = 90
    if forecast_start is None: forecast_start = prior_length + 50
    if forecast_end is None: forecast_end = len(Y)-1
    s0 = 0.005

    if ntrend == 1:
        deltrend = discount
    elif ntrend == 2:
        deltrend = [discount, 0.95]

    ## Run the analysis
    mod, samples = analysis(Y, X,
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


    mod.r_lags = river_lags
    mod.or_lags = off_river_lags


    ## If return_all=True, return a lot of information
    if return_all:
        return mod, samples, Y[forecast_start:forecast_end+1], dates

    forecast = median(samples)

    ## Otherwise, just return the MAD
    return np.round(MAD(Y[forecast_start:forecast_end+1], forecast), 4)


# Bayesian Function to Predict Independent Variables
def forecast_path_ar(mod_R, mod_OR, k, lagged_R, lagged_OR, makeX_R, makeX_OR,
                     # These are known variables in the models for River and Off River
                     X_R=None, X_OR=None,
                     # These are variables in the River model, interacting with lag-1 River and lag-1 Off River
                     X_R_interact_R=None, X_R_interact_OR=None,
                     # These are variables in the Off River model, interacting with lag-1 River and lag-1 Off River
                     X_OR_interact_R=None, X_OR_interact_OR=None,

                     nsamps=1, max_lag=5):
    ### MakeX_R is a function to make the predictor variables for River
    ### MakeX_OR is a function to make the predictor variables for Off River

    xdf = []

    samps_river = np.zeros([nsamps, k])
    samps_offriver = np.zeros([nsamps, k])

    # Setting up an array to hold a simulated path into the future
    R_simulated_path = np.zeros(k + max_lag)
    OR_simulated_path = np.zeros(k + max_lag)

    # Filling in the known, past values, to initialize the simulated path
    R_simulated_path[:max_lag] = lagged_R[-max_lag:].reshape(-1)
    OR_simulated_path[:max_lag] = lagged_OR[-max_lag:].reshape(-1)

    # Setting up the X_R, and X_OR in case they are blank
    if X_R is None: X_R = [np.array([]) for i in range(k)]

    if X_OR is None: X_OR = [np.array([]) for i in range(k)]

    if X_R_interact_R is None: X_R_interact_R = [np.array([]) for i in range(k)]

    if X_R_interact_OR is None: X_R_interact_OR = [np.array([]) for i in range(k)]

    if X_OR_interact_R is None: X_OR_interact_R = [np.array([]) for i in range(k)]

    if X_OR_interact_OR is None: X_OR_interact_OR = [np.array([]) for i in range(k)]


    # Save the model coefficients, because they will be changed in this function
    a_R = np.copy(mod_R.a)
    R_R = np.copy(mod_R.R)
    if type(mod_R) == pybats.dglm.dlm:
        s_R = mod_R.s
        n_R = mod_R.n
    t_R = mod_R.t

    a_OR = np.copy(mod_OR.a)
    R_OR = np.copy(mod_OR.R)
    if type(mod_OR) == pybats.dglm.dlm:
        s_OR = mod_OR.s
        n_OR = mod_OR.n
    t_OR = mod_OR.t

    # For each forecast sample we want to draw...
    for n in range(nsamps):

        # Reset the model coefficients to their original values
        mod_R.a = a_R.copy()
        mod_R.R = R_R.copy()
        if type(mod_R) == pybats.dglm.dlm:
            mod_R.s = s_R
            mod_R.n = n_R

        mod_OR.a = a_OR.copy()
        mod_OR.R = R_OR.copy()
        if type(mod_OR) == pybats.dglm.dlm:
            mod_OR.s = s_OR
            mod_OR.n = n_OR


        # For each time step into the future...
        for i in range(k):

            # Simulate a value 1-step ahead of Off-River
            X = np.array(makeX_OR(R_simulated_path[:max_lag + i], OR_simulated_path[:max_lag + i],
                                  X_OR[i], X_OR_interact_R[i], X_OR_interact_OR[i],
                                  mod_OR.r_lags, mod_OR.or_lags))
            samps_offriver[n, i] = OR_simulated_path[max_lag + i] = mod_OR.forecast_marginal(k=1, X=X, nsamps=1)

            # Simulate a value 1-step ahead of River
            X = np.array(makeX_R(R_simulated_path[:max_lag + i], OR_simulated_path[:max_lag + i],
                                 X_R[i], X_R_interact_R[i], X_R_interact_OR[i],
                                 mod_R.r_lags, mod_R.or_lags))
            samps_river[n, i] = R_simulated_path[max_lag + i] = mod_R.forecast_marginal(k=1, X=X, nsamps=1)

            # Update the Off River model based on the simulated values
            y = samps_offriver[n, i]
            X = np.array(makeX_OR(R_simulated_path[:max_lag + i + 1], OR_simulated_path[:max_lag + i + 1],
                                  X_OR[i], X_OR_interact_R[i], X_OR_interact_OR[i],
                                  mod_OR.r_lags, mod_OR.or_lags))
            mod_OR.update(y, X)

            # Update the River model based on the simulated values
            y = samps_river[n, i]
            X = np.array(makeX_R(R_simulated_path[:max_lag + i + 1], OR_simulated_path[:max_lag + i + 1],
                                 X_R[i], X_R_interact_R[i], X_R_interact_OR[i],
                                 mod_R.r_lags, mod_R.or_lags))
            mod_R.update(y, X)

            # Undo discounting of information, if desired
            if not mod_R.discount_forecast: mod_R.R = mod_R.R - mod_R.W
            if not mod_OR.discount_forecast: mod_OR.R = mod_OR.R - mod_OR.W

            xdf.append(X)

    mod_R.a = a_R.copy()
    mod_R.R = R_R.copy()
    if type(mod_R) == pybats.dglm.dlm:
        mod_R.s = s_R
        mod_R.n = n_R
    mod_R.t = t_R

    mod_OR.a = a_OR.copy()
    mod_OR.R = R_OR.copy()
    if type(mod_OR) == pybats.dglm.dlm:
        mod_OR.s = s_OR
        mod_OR.n = n_OR
    mod_OR.t = t_OR

    xf_tuple = tuple(xdf)
    xf_df = pd.DataFrame(xf_tuple)

    return samps_river #, xf_df

def makeX_R(lagged_R, lagged_OR, X, X_interact_R, X_interact_OR, r_lags, or_lags):
    # Returning lag1_river and lag1_offriver
    out = []

    for rl in r_lags:
        out.append(lagged_R[-rl])
    for orl in or_lags:
        out.append(lagged_OR[-orl])

    X_interact_R = X_interact_R * lagged_R[-1]
    X_interact_OR = X_interact_OR * lagged_OR[-1]

    out.extend(X.tolist()) # These are the additional known variables
    out.extend(X_interact_R)
    out.extend(X_interact_OR)
    return out

def makeX_OR(lagged_R, lagged_OR, X, X_interact_R, X_interact_OR, r_lags, or_lags):
    # Returning lag1_offriver
    out = []

    for rl in r_lags:
        out.append(lagged_R[-rl])
    for orl in or_lags:
        out.append(lagged_OR[-orl])

    X_interact_R = X_interact_R * lagged_R[-1]
    X_interact_OR = X_interact_OR * lagged_OR[-1]

    out.extend(X.tolist()) # These are the additional known variables
    out.extend(X_interact_R.tolist())
    out.extend(X_interact_OR.tolist())
    return out


def get_confusion_matrix(y_true, samples):
    forecast = median(samples)
    increase = y_true[1:] > y_true[:-1]
    prediction = forecast[1:] > forecast[:-1]
    diff_a = y_true[1:] - y_true[:-1]
    diff_p = forecast[1:] - forecast[:-1]
    confusion = np.zeros([2,2])
    confusion[0,0] = sum([i and p for i,p in zip(increase, prediction)])
    confusion[1,1] = sum([not i and not p for i,p in zip(increase, prediction)])
    confusion[0,1] = sum([not i and p for i,p in zip(increase, prediction)])
    confusion[1,0] = sum([i and not p for i,p in zip(increase, prediction)])
    return confusion, increase, diff_a, diff_p

def get_auc(y_true, samples):
    forecast = median(samples)
    increase = y_true[1:] > y_true[:-1]
    prediction = forecast[1:] > forecast[:-1]
    return roc_auc_score(increase, prediction)


# Function to update the forecasts on the database
def update_data(df, contents, filename):
    if contents is not None:
        new_data = parse_contents(contents, filename)
        df_new = pd.DataFrame(new_data)
        df_all = pd.concat([df, df_new]).reset_index(drop=True)
        return df_all
    else:
        return df

def parse_contents(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')))
            elif 'xlsx' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

        return df
    else:
        return [{}]


# Function to Create Technical Inidicators
def mov_avg(var, size):
    window_size = size

    numbers_series = pd.Series(var)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()

    ma = moving_averages.tolist()
    #without_nans = moving_averages_list[window_size - 1:]
    return ma



def RSI(series,n):
    ewma = pd.Series.ewm
    n = 14
    delta = series.diff()
    u = delta * 0
    d = u.copy()
    i_pos = delta > 0
    i_neg = delta < 0
    u[i_pos] = delta[i_pos]
    d[i_neg] = delta[i_neg]
    rs = ewma(u, span=n).mean() / ewma(d, span=n).mean()
    return 100 - 100 / (1 + rs)




# Get the Data
def get_data():
    engine = sqlalchemy.create_engine("...")
    con = engine.connect()
    df = pd.read_sql("corn_data_v2", con=engine)
    #df = df.dropna(inplace=True)
    df_geo = pd.read_sql("corn_geo", con=engine)
    con.close()
    engine.dispose()
    return df, df_geo


app.layout = html.Div([

    html.Div([
        html.H2('CMV Corn Spot Price Forecast Tool',)

    ], className='row'),

    html.Div([

                        html.Div(
                            [
                                html.P("To update the forecast dataset, drag new data here:"),
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(['Drag and Drop or ',
                                    html.A('Select Files')
                                    ]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px'
                                        },
                                        # Allow multiple files to be uploaded
                                        multiple=False
                                        ),

                                html.Hr(),
                                html.Br(),
                                html.Div([html.P("Number of Days in Dataset:  "), html.Div(id='num_days')], className='row'),
                                html.Br(),
                                html.Div([html.P("Last day of valid date:  "), html.Div(id='last_day')], className='row'),
                                html.Br(),
                                html.Div([html.P("Number of new days added:  "), html.Div(id='num_new_days')], className='row'),
                                ],
                                className='pretty_container three columns'
                            ),
                            html.Div(
                                [
                                dcc.Tabs([
                                    dcc.Tab(label='Model Summary', children=[
                                            html.Div([html.Div([html.P("Model Accuracy:"), html.Div(id='auc')], className='mini_container'),
                                                      html.Div([html.P("Predicted Direction Tomorrow:"), html.Div(id='direction')], className='mini_container'),
                                                      html.Div([html.P("Probability of Direction:"), html.Div(id='probability')], className='mini_container'),
                                                      html.Div([html.P("Model Was Correct:"), html.Div(id='actual_direc')], className='mini_container')
                                                      ], className='row'),
                                            html.Div(
                                                [

                                                    html.Div(
                                                        [
                                                            html.P("This is the sequence plot showing the actual values for the River and Off-River gorups.\
                                                            We're using basis data starting in January 2017 until the\
                                                            current day.  You can also select from a number of Technical Indicators as overlays to the graph."),
                                                            html.Div([dcc.Dropdown(id='select_days',
                                                                            options=[
                                                                                {'label': 'All', 'value': ''},
                                                                                {'label': '5 days', 'value': -5},
                                                                                {'label': '30 days', 'value': -30},
                                                                                {'label': '3 Months', 'value': -90},
                                                                                {'label': '1 year', 'value': -365},
                                                                                {'label': '2 years', 'value': -730},

                                                                            ],
                                                                            placeholder="Select Number of Days",
                                                                            style={"margin-right": "15px"}
                                                                            ),
                                                            html.P("Technical Indicators:"),
                                                            dcc.Dropdown(id='tech_ind',
                                                                            options=[
                                                                                {'label': '7MA', 'value': 'ma7'},
                                                                                {'label': '21MA', 'value': 'ma21'},
                                                                                {'label': 'EXP_MA', 'value': 'ema'},
                                                                                {'label': 'RSI', 'value': 'rsi'},
                                                                                {'label': 'Bollinger', 'value': '20sd'},
                                                                                {'label': 'MACD', 'value': 'macd'},

                                                                            ],
                                                                            value='ma7',
                                                                            style={"margin-left": "5px"}
                                                                            )
                                                            ], className='row'),
                                                            dcc.Graph(
                                                                    id='graph',
                                                            )
                                                        ],

                                                        id="countGraphContainer",
                                                        className='pretty_container eight columns'),

                                                ],
                                                id="tripleContainer",
                                                className='row'),


                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.P("Model 1: Bayesian Gaussian Model uses 1-step-ahead forecasting for validation.\
                                                                    The historical period can be adjusted up to 60 days.\
                                                                    The graph below shows that the model has learned the\
                                                                    dynamics of this time series quite well."),
                                                            dcc.Graph(
                                                                id='vgraph',
                                                            )
                                                        ], className='pretty_container six columns'),
                                                    html.Div(
                                                        [
                                                            html.P("This is scatter plot of the predicted forecast for the next 3 days by the probability.\
                                                                    The vertical line in the middle indicates no change, and the blue band shows the average\
                                                                    change for the previous 5 days to provide a sense of the normal amount of variation."),
                                                            dcc.Graph(
                                                                id='fgraph',
                                                            )
                                                        ], className='pretty_container six columns'),
                                                ], className='row'),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.P("Model 2: Binary Model. This model uses a Bayesian Algorithm to predict the probability of an increase\
                                                                    or decrease in the basis.  Values above .5 indicate an increase, while values below .5 \
                                                                    indicate a decrease."),
                                                            dcc.Graph(
                                                                id='binary_plot',
                                                            )
                                                        ], className='pretty_container six columns'),
                                                    html.Div(
                                                        [
                                                            html.P("Correlation of River vales and Off-River Lagged 1 day."),
                                                            dcc.Graph(
                                                                id='correlation',
                                                            )
                                                        ], className='pretty_container six columns'),
                                                ], className='row'),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.P("This summary table shows the predicted values\
                                                            and the day-to-day change in predicted values ('diff').\
                                                            It also shows the probability that the change is an\
                                                            increase or a decrease."),
                                                            html.Div(id='forecast_table')
                                                        ], className='pretty_container six columns'),
                                                    html.Div(
                                                        [
                                                            html.P("The ROC Curve is a method to gauge the accuracy of the model.  The curve should bend\
                                                                    away from the diagonal line, which indicates chance.  This current model performs better\
                                                                    than chance in predicting the next day's increase or decrease in basis."),
                                                            html.Div(id='cm'),
                                                            dcc.Graph(
                                                                id='diff_graph',
                                                            )
                                                        ], className='pretty_container six columns'),
                                                ], className='row'),

                                    ],
                                    id="rightCol",
                                    className='eight columns'),

                            dcc.Tab(label='Raw Data', children=[
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.P("These are the locations of the Corn Elevators."),
                                                dcc.Graph(
                                                    id='geograph',
                                                    )
                                            ],

                                    id="geoGraphContainer",
                                    className='pretty_container twelve columns'),

                                    ],
                                    id="GeoContainer",
                                    className='pretty_container eight columns'),
                                html.Div([
                                    html.Div(id='all_data')
                                ], className='pretty_container ten columns')
                            ], className='twelve columns'),


                        ]),
                ], className='row'),
                html.Div(id='table', className='twelve columns'),



        ])

], id="mainContainer", style={"display": "flex","flex-direction": "column"}
)

# Show Uploaded Data in Data Table
@app.callback(Output('table', 'children'),
             [Input('upload-data', 'contents'),
              Input('upload-data', 'filename')])

def generate_table(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            table = html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns])
            ])

        return table

# Forecast algorithm callback
@app.callback([Output('graph', 'figure'),
               Output('vgraph', 'figure'),
               Output('fgraph', 'figure'),
               Output('diff_graph', 'figure'),
               Output('binary_plot', 'figure'),
               Output('correlation', 'figure'),
               Output('geograph', 'figure'),
               Output('forecast_table', 'children'),
               Output('num_days', 'children'),
               Output('last_day', 'children'),
               Output('num_new_days', 'children'),
               Output('auc', 'children'),
               Output('all_data', 'children'),
               Output('cm', 'children'),
               Output('direction', 'children'),
               Output('probability', 'children'),
               Output('actual_direc', 'children')],
               [Input('upload-data', 'contents'),
                Input('upload-data', 'filename'),
                Input('select_days', 'value'),
                Input('tech_ind', 'value')])


def forecast(contents, filename, s_days, ti):

    df_original, df_geo = get_data()
    df_t = df_original.copy()  # create a copy of the dataframe
    print(df_t)
    df_t = df_t.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)  # drop any duplicates
    if contents is not None:
        df1 = update_data(df_t, contents, filename)
        df = df1.dropna(inplace=False)
        print("loaded datafarme", df)
        if df.shape[0] > df_t.shape[0]:
             engine = sqlalchemy.create_engine("...")
             con = engine.connect()
             df.to_sql("corn_data_v2", con=engine, method='multi', if_exists="replace", index=False)
             engine.dispose()
             con.close()
             new_days = df.shape[0] - df_t.shape[0]
        else:
            new_days = 0
        new_days = df.shape[0] - df_t.shape[0]
    else:
        df = df_t
        new_days = 0

    days = 3  # sets the number of forecast days
    vdays = 20 # sets the number of validation days


    ndays = str(days)
    validdays = str(vdays)

    # Create Technical Inidicators
    df['ma7'] = mov_avg(df['River'], 7)             # 9-day Moving Average
    df['ma21'] = mov_avg(df['River'], 21)           # 21-day Moving Average
    df['ema'] = df['River'].ewm(com=0.5).mean()     # Exponential Moving Average, alph = .5
    df['rsi'] = RSI(df.River, 14)                   #
    df['20sd'] = df['River'].rolling(window=20).std()
    emafast = df['River'].ewm(span=12, adjust=False).mean()
    emaslow = df['River'].ewm(span=26, adjust=False).mean()
    df['macd'] = emafast-emaslow

    # Summary Statistics
    last_day = pd.to_datetime(df['Date']).dt.date.max()
    num_days = df.shape[0]

    # STEP 1: DATA PREP ##################################################################################################

    y = df['River']
    y2 = df['Off_River']

    # date/time and axis
    x_axis = df['Date']
    date = pd.to_datetime(df['Date']).dt.date
    f_axis = pd.bdate_range(start=date.max(), periods=4)[-3:]

    # Interactions of p and lag-1 River
    for i in range(1, 6):
        n = df.shape[0]
        tmp = np.zeros(n)
        tmp[1:] = df['p' + str(i)][:-1] * df['River'][:-1]
        df['p' + str(i) + '_lag1_r'] = tmp

    # Interactions of p and lag-1 Off River
    for i in range(1, 6):
        n = df.shape[0]
        tmp = np.zeros(n)
        tmp[1:] = df['p' + str(i)][:-1] * df['Off_River'][:-1]
        df['p' + str(i) + '_lag1_or'] = tmp

    k = 3
    X_R = np.zeros([k, 7])
    X_R[:,0] = 1 # Setting gs=1 because we're in the growing season
    X_R[:,2] = df.g1[-1:]
    X_R[:,3] = df.g2[-1:]
    X_R[:,4] = df.g3[-1:]
    X_R[:,5] = df.g4[-1:]
    X_R[:,6] = df.g5[-1:]


    X_R_interact_R = np.zeros([k, 5])
    X_R_interact_R[:,2]=1
    X_R_interact_OR = X_R_interact_R


    # STEP 2: BAYESIAN TIMES SERIES MODEL ################################################################################

    # First run the river analysis, returning the model
    var_names = ['gs', 'drop', 'g1', 'g2', 'g3', 'g4', 'g5',
                 'p1_lag1_r', 'p2_lag1_r', 'p3_lag1_r', 'p4_lag1_r', 'p5_lag1_r',
                 'p1_lag1_or', 'p2_lag1_or', 'p3_lag1_or', 'p4_lag1_or', 'p5_lag1_or']
    r_lags = [1] # I have removed lag-1 river because it's redundant
    or_lags = [1] # I have removed lag-1 off river because it's redundant
    mod_R, samples_R, plot_Y_R, forecast_dates_R = test_variables(df, var_names, r_lags, or_lags, seasPeriods=[5], seasHarmComponents=[[1]],
                                                                  return_all=True)

    # And then the off-river analysis, returning the model
    var_names_or = []
    r_lags_or = []
    or_lags_or = [2]
    mod_OR, samples_OR, plot_Y_OR, forecast_dates_OR = test_variables(df, var_names_or, r_lags_or, or_lags_or, seasPeriods=[5], seasHarmComponents=[[1]],
                                                                      return_all=True, Y=['Off_River'])

    forecast_bayesian = median(samples_R)
    #df_forecast = pd.DataFrame(forecast_bayesian)
    #df_forecast.to_excel("df_forecast.xlsx")


    nsamps = 1000
    lagged_R = plot_Y_R  # These are the values of River
    lagged_OR = plot_Y_OR # These are the values of Off River
    forecast_samples = forecast_path_ar(mod_R, mod_OR, k=k, lagged_R=lagged_R, lagged_OR=lagged_OR,
                                        makeX_R=makeX_R, makeX_OR=makeX_OR,
                                        X_R = X_R,
                                        X_R_interact_R = X_R_interact_R,
                                        X_R_interact_OR = X_R_interact_OR,
                                        nsamps=nsamps)

    # set confidence interval
    credible_interval=95
    alpha = (100-credible_interval)/2
    b_upper=np.percentile(samples_R, [100-alpha], axis=0).reshape(-1)
    b_lower=np.percentile(samples_R, [alpha], axis=0).reshape(-1)

    #future = mod.forecast_path(k=days, X=z_new[-days:].values, nsamps=1000)
    f = median(forecast_samples)
    #predicted = pd.DataFrame(f)
    #predicted.to_excel("forecast_values.xlsx")
    f_upper=np.percentile(forecast_samples, [100-alpha], axis=0).reshape(-1)
    f_lower=np.percentile(forecast_samples, [alpha], axis=0).reshape(-1)

    f_ci = pd.DataFrame([f_lower, f_upper]).T
    f_ci.columns=['low', 'high']

    # Get Probabilities
    y_last = y[-1:]
    y_new = np.concatenate((y_last, f), axis=0)
    y_pred_diff = y_new - y_last.values
    #y_pred_diff = pd.Series(y_new).substract(y_last)
    z = np.arange(1,4,1)
    fut_preds_df = pd.DataFrame(forecast_samples, columns=[z])
    #fut_preds_df.to_excel('future_pred.xlsx')

    # get the last value of the dataset
    lv = df.River[-3:].mean()
    last_val = df.River[-1:].values
    direc = f[0] - last_val

    # create an array that repeats the value the number of foreast days
    d_ar = pd.DataFrame([last_val]*5).values.ravel()

    # zip the array with the columns of the forecast samples
    d = dict(zip(fut_preds_df,d_ar))

    # calculate the number of samples above the last day in the data set for each forecast day
    if direc < 0:
        prob_dec = pd.Series([fut_preds_df[k].lt(v).sum()/fut_preds_df.shape[0] for k,v in d.items()],index=d.keys())
        p = prob_dec.values.ravel()
        prob = prob_dec
        print("prob",prob)
    else:
        prob_inc = pd.Series([fut_preds_df[k].gt(v).sum()/fut_preds_df.shape[0] for k,v in d.items()],index=d.keys())
        p = prob_inc.values.ravel()
        prob = prob_inc
        print("prob",prob)




    # CONFUSION MATRIX AND ROC CURVE ############################################################################################
    confusion, increase, diff_a, diff_p = get_confusion_matrix(plot_Y_R, samples_R)
    cmn = pd.DataFrame(np.round(confusion/len(plot_Y_R), 2))
    cmn.columns = ['increase', 'decrease']
    cmn.reset_index()
    cmn.reindex(index=['pred. increase', 'predicted decrease'])

    # AUC
    #roc = "{0:.0%}".format(np.round(get_auc(plot_Y_R, samples_R),2))
    roc = get_auc(plot_Y_R, samples_R)

    # Confusion Matrix
    cm = np.round(confusion/len(plot_Y_R), 2)
    cm_acc = cm[0][0] + cm[1][1]

    if roc > cm_acc:
        auc = np.round(roc,3)
    else:
        auc = np.round(cm_acc,3)

    # ROC Curve
    fpr, tpr, _ = roc_curve(increase, diff_p)

    # Create the Summary Table
    empty_list = [0]
    for i in p:
        empty_list.append(i)
    pl = pd.Series(empty_list)


    forecast_axis = pd.bdate_range(start=date.max(), periods=4).strftime("%m/%d/%Y")
    f_all = np.concatenate((last_val, f), axis=0)
    yn = pd.DataFrame(f_all, forecast_axis).reset_index().round(6)
    yn_df = pd.concat((yn, pl), axis=1)
    yn_df.columns = (['date', 'forecast', 'probability'])
    yn_df['diff_previous'] = yn_df.forecast.diff().round(6)

    if direc > 0:
        direction = 'increase'
    else:
        direction = 'decrease'

    fs =["add new data..."]
    if contents is not None:
        chk_p = forecast_bayesian[-1] - df['River'].iloc[-2]
        chk_act = last_val - df['River'].iloc[-2]
        if (chk_p > 0 and chk_act > 0):
            fs = 'yes'
        elif (chk_p < 0 and chk_act < 0):
            fs = 'yes'
        else:
            fs = 'no'
    else:
        fs =["add new data..."]


    # BINARY Model ###########################################################################################################
    var_names_b = ['gs', 'drop', 'g1', 'g2', 'g3', 'g4', 'g5']
    r_lags_b = [1]
    or_lags_b = [2]

    mod_R_b, samples_R_b, plot_Y_R_b, forecast_dates_R_b = test_variables(df, var_names_b, r_lags_b, or_lags_b, seasPeriods=[5], seasHarmComponents=[[1]],
                                                                  return_all=True, binary=True)

    # And then the off-river analysis, returning the model
    var_names_b_or = []
    r_lags_b_or = []
    or_lags = [2]
    mod_OR_b, samples_OR_b, plot_Y_OR_b, forecast_dates_OR_b = test_variables(df, var_names_b_or, r_lags_b_or, or_lags, seasPeriods=[5], seasHarmComponents=[[1]],
                                                                      return_all=True, Y=['Off_River'])

    X_R_b = np.zeros([k, 7])
    X_R_b[:,0] = 1 # Setting gs=1 because we're in the growing season
    X_R_b[:,2] = df.g1[-1:]
    X_R_b[:,3] = df.g2[-1:]
    X_R_b[:,4] = df.g3[-1:]
    X_R_b[:,5] = df.g4[-1:]
    X_R_b[:,6] = df.g5[-1:]

    bsamps = 500
    lagged_R_b = plot_Y_R_b  # These are the values of River
    lagged_OR_b = plot_Y_OR_b # These are the values of Off River
    forecast_samples_b = forecast_path_ar(mod_R_b, mod_OR_b, k=k, lagged_R=lagged_R_b, lagged_OR=lagged_OR_b,
                                        makeX_R=makeX_R, makeX_OR=makeX_OR,
                                        X_R = X_R_b,
                                        nsamps=bsamps)

    binary_for = mean(samples_R_b)

    binary_future = mean(forecast_samples_b)
    #print(binary_future, "binary_future")
    if direc > 0:
        probability = prob_inc.iloc[0]
        #print("probability", probability.iloc[0])
    else:
        probability = prob_dec.iloc[0]
        #print("probability", probability.iloc[0])

    bf = binary_future
    #########################################################################################################

    # ARIMA #################################################################################################

    arima_mod = joblib.load('full_arima.pkl')
    m_x = pmd.auto_arima(df.Off_River)
    x_fut = m_x.predict(n_periods = 3)

    arima_forecast = arima_mod.predict(n_periods=3, exogenous=x_fut.reshape(-1,1))

    ########################################################################################################

    # this puts the above code into an html table to be displayed
    cm = html.Div([
        dash_table.DataTable(
            data=cmn.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in cmn.columns])
    ])

    forecast_table = html.Div([
        dash_table.DataTable(
            data=yn_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in yn_df.columns])
    ])

    all_data = html.Div([
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns],
            style_table={'maxHeight': '300px', 'overflowY': 'scroll'})
    ])


    # PLOTS: ########################################################################################################
    # Plot 1 (fig1): the main plot showing the in-sample fit based on the SARIMAX model with exogenous predictors
    # Plot 2 (fig2)
    # Plot 3 (fig3)
    # Plot 4 (fig4)
    # Plot 5 (fig5)
    #################################################################################################################

    # Main Plot Showing the River Series
    if s_days == '':
        s_days = len(y)*-1
    else:
        s_days

    print("s_days", s_days)
    River = go.Scatter(
                x = x_axis,
                y = y[s_days:],
                line=dict(color='rgb(0,176,246)', width=1),
                mode='lines',
                name='River Group',
                #fill='tonexty',
                #fillcolor='rgba(179, 199, 255, 0.3)'
                )
    Off_River = go.Scatter(
                x = x_axis,
                y = y2[s_days:],
                line=dict(color='rgb(255,55,246)', width=1),
                mode='lines',
                name='Off River Group',
                #fill='tonexty',
                #fillcolor='rgba(179, 199, 255, 0.3)'
                )
    tech = go.Scatter(
                x = x_axis,
                y = df[ti][s_days:],
                line=dict(color='rgb(0,64,255)', width=1),
                mode='lines',
                name=ti,
                #fill='tonexty',
                #fillcolor='rgba(179, 199, 255, 0.3)'
                )

    layout = go.Layout(title='Actual Data:',
                       xaxis_title='Date',
                       plot_bgcolor='rgba(0,0,0,0)',
                       font=dict(family='Helvetica', size=10, color='#7f7f7f'))
    fig = go.Figure(data = [River, Off_River, tech], layout=layout)

    # Second Plot Showing the 1-step ahead Validation
    trace_y = go.Scatter(
                x = x_axis[-vdays:],
                y = y[-vdays:],
                line=dict(color='rgb(0,176,246)', width=1),
                mode='lines',
                name='Actual Values'
                )
    trace_b_low = go.Scatter(
                name = '95% low',
                x = x_axis[-vdays:],
                y = b_lower[-vdays:].ravel(),
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),

    )
    trace_b_high = go.Scatter(
                name = '95% high',
                x = x_axis[-vdays:],
                y = b_upper[-vdays:].ravel(),
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                fillcolor='rgba(27, 199, 0, 0.1)',
                fill='tonexty'
    )
    trace_b = go.Scatter(
                x = x_axis[-vdays:],
                y = forecast_bayesian[-vdays:].ravel(),
                line=dict(color='rgb(27, 199, 0)', width=1),
                mode='lines+markers',
                name='In Sample Predicted',
                fill='tonexty',
                fillcolor='rgba(27, 199, 0, 0.1)'
                )
    f_lower = go.Scatter(
                x = f_axis.strftime("%m/%d/%Y"),
                y = f_ci['low'][:days],
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                name = 'forecast 95% Low'
                )
    f_upper = go.Scatter(
                name = 'forecast 95% High',
                x = f_axis.strftime("%m/%d/%Y"),
                y = f_ci['high'][:days],
                mode='lines',
                marker=dict(color='#444'),
                line=dict(width=0),
                fillcolor='rgba(255, 126, 0, 0.1)',
                fill='tonexty'
                )

    forecast = go.Scatter(
                x = f_axis.strftime("%m/%d/%Y"),
                y = f[:days],
                line=dict(color='rgb(255, 0, 0)', width=1, dash='dash'),
                mode='lines+markers',
                name='Bayesian Forecast',
                fill='tonexty',
                fillcolor='rgba(255, 126, 0, 0.1)'
                )

    arima = go.Scatter(
                x = f_axis.strftime("%m/%d/%Y"),
                y = arima_forecast[:days],
                line=dict(color='rgb(125, 0, 125)', width=1, dash='dash'),
                mode='lines+markers',
                name='ARIMA Forecast',
                #fill='tonexty',
                #fillcolor='rgba(255, 126, 0, 0.1)'
                )


    layout2 = go.Layout(title='Forecast Validation from Hold-out Sample: '+validdays,
                        xaxis_title='Date',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Helvetica', size=10, color='#7f7f7f'))
    fig2 = go.Figure(data=[trace_y, trace_b_low, trace_b, trace_b_high, f_lower, forecast, f_upper, arima], layout=layout2)


    # generate the forecast DIFFERENCE plot
    y_std = np.std(y[-3:].diff())
    ypd = y_pred_diff[~np.isnan(y_pred_diff)]

    pred_diff = go.Scatter(
                x=ypd[1:],
                y=prob,
                line=dict(color='rgb(255,50,0)', width=1),
                mode='markers',
                name='Bayesian Forecast',
                yaxis='y1'
                )
    pred_mean = go.Scatter(
                x=pd.Series(ypd[1:].mean()),
                y=pd.Series(prob.mean()),
                line=dict(color='rgb(0,0,255)', width=1),
                mode='markers',
                name='Average Forecast',
                yaxis='y1'
                )
                # probability line
    horiz1      = go.Scatter(
                x=[-.005, .005],
                y=[.5]*len(x_axis),
                mode='lines',
                line=dict(color='black', dash='dash', width=.5),
                name='chance 50%'
                )
                # no change line
    vert1       = go.Scatter(
                x=[0,0],
                y=[0,1],
                mode='lines',
                line=dict(color='black', dash='dash', width=.5),
                name='No Change'
                )
    s1       = go.Scatter(
                x=[y_std,y_std],
                y=[0,1],
                mode='lines',
                line=dict(color='blue', dash='dash', width=.5),
                name='average variation',
                fill='tonextx',
                fillcolor='rgba(0, 0, 255, 0.1)'
                )
    s2       = go.Scatter(
                x=[-y_std,-y_std],
                y=[0,1],
                mode='lines',
                line=dict(color='blue', dash='dash', width=.5),
                name='average variation',
                fill='tonextx',
                fillcolor='rgba(0, 0, 255, 0.1)'
                )


    #prob_increase = go.Scatter(
    #            x=f_axis.strftime("%Y/%m/%d"),
    #            y=prob_inc[:days],
    #            line=dict(color='rgb(255,50,0)', width=1),
    #            mode='lines+markers',
    #            name='Probability of Increase',
    #            yaxis='y2'
    #            )

    layout3 = go.Layout(title='Predicted Change in Basis for the next : '+ndays+' days',
                        xaxis_title='Predicted Change in Basis',
                        yaxis=dict(title="Probability of Change", range=[0, 1], side='left'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Helvetica', size=10, color='#7f7f7f'))
    fig3 = go.Figure(data=[pred_diff, horiz1, vert1, pred_mean, s1, s2], layout=layout3)

    # Generate Bar Plot
    trace1 = go.Scatter(
            x=fpr,
            y=tpr,
    )
    trace_diag = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line=dict(color='red', width=.5, dash='dash'),
            mode='lines'

    )
    traces = [trace1, trace_diag]
    layout4 = go.Layout(title='ROC',
                        yaxis_title='True Positive Rate',
                        xaxis_title='False Positive Rate',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Helvetica', size=10, color='#7f7f7f'))

    fig4 = go.Figure(data=traces, layout=layout4)

    # Binary Plot
    binary_model = go.Scatter(
                x=x_axis[-vdays:],
                y=binary_for[-vdays:].ravel(),
                line=dict(color='rgb(27, 199, 0)', width=1),
                mode='lines+markers',
                name='Probability of Increase'
                )
    actual_diff = go.Scatter(
                x=x_axis[-vdays:],
                y=plot_Y_R_b.ravel()[-vdays:],
                line=dict(color='rgb(0,0,255)', width=1),
                mode='markers',
                name='Actual of Increase'
                )
    binary_future = go.Scatter(
                x=f_axis.strftime("%Y/%m/%d"),
                y=bf[-days:],
                line=dict(color='rgb(255,50,0)', width=1),
                mode='markers',
                name='Forecast Probabilty'
                )
    horiz       = go.Scatter(
                x=x_axis[-vdays:],
                y=[.5]*len(x_axis[-vdays:]),
                mode='lines',
                line=dict(color='black', dash='dash', width=.5),
                name='chance 50%'
                )

    layout6 = go.Layout(title='Binary Prediction Model',
                        xaxis_title='Date',
                        yaxis=dict(title="Probability of Increase in Basis", side='left'),
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Helvetica', size=10, color='#7f7f7f')
                        )

    fig6 = go.Figure(data = [binary_model, actual_diff, horiz, binary_future], layout=layout6)



    # Correlation Plot
    correlation = go.Scatter(
                x = df.River[1:],
                y = df.Off_River[:-1],
                marker=dict(color=df.index.to_numpy()[1:]),
                mode='markers'
                )

    layout7 = go.Layout(title='Correlation Between River and Off-River Lags',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Helvetica', size=10, color='#7f7f7f'))
    fig7 = go.Figure(data = [correlation], layout=layout7)

    scl = [[0,"rgb(100,0,200)"],[1,"rgb(100,0,225)"],[2,"rgb(100,0,250)"],[3,"rgb(75,0,200)"],[4,"rgb(50,0,200)"],[5,"rgb(25,0,200)"],[6,"rgb(0,0,200)"],[7,"rgb(255, 0, 0)"]]

    geo_fig = go.Figure(data=go.Scattergeo(
            lon = df_geo['long'],
            lat = df_geo['lat'],
            text = df_geo['ID'],
            mode = 'markers',
            marker = dict(
                        color = df_geo['prox'],
                        colorscale = 'Plasma',
                        reversescale = True,
                        opacity = 0.7,
                        size = 2)
            ))

    geo_fig.update_layout(
            title = 'Corn Elevators',
            geo_scope='usa',
        )

    return fig, fig2, fig3, fig4, fig6, fig7, geo_fig, forecast_table, num_days, last_day, new_days, auc, all_data, cm, direction, probability, fs


if __name__ == '__main__':
    application.run(debug=True, port=8080)


# All Python and Statistical Algorithms written by Ryan J. Clukey, User Metrics, LLC. 2020.
