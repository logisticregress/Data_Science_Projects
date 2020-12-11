import lifetimes
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lifetimes.datasets import load_dataset
import lifetimes 
from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_probability_alive_matrix, plot_frequency_recency_matrix
from lifetimes.generate_data import beta_geometric_nbd_model
from lifetimes import GammaGammaFitter
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases, plot_period_transactions,plot_history_alive
import psycopg2
import sqlalchemy as sa
#from sqlalchemy import create_engine
import boto3
import seaborn as sns

# To perform KMeans clustering 
from sklearn.cluster import KMeans

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
fig = plt.figure(figsize=(20,10))

### DATABASE CONNECTION SCRIPT - REQUIRES SSH Connection
#
#
#
#
#######################################################

sql = ("""SELECT customer_id, email
                 , date_trunc('month', CAST(order_date as DATE)) as date
                 , count(product_amount) as orders
                 , avg(product_amount) as amount
        FROM uos.orders
        WHERE date >= '2016-01-01'
        AND product_amount >= 0
        GROUP BY customer_id, email, date
""")

# Functions

def get_data(s, rs_dict):
    engine = sa.create_engine('redshift+psycopg2://'+rs_dict['user']+':'+rs_dict['password']+
                          '@'+rs_dict['host']+':'+rs_dict['port']+'/'+rs_dict['dbname'], connect_args={'sslmode': 'prefer'})
    con = engine.raw_connection()
    df = pd.read_sql(sql, con)
    df.dropna(axis=0,inplace=True)
    df1 = df.sample(frac=s, replace=False)
    df1['date'] = pd.to_datetime(df.date)
    df1['email'] = df1['email'].str.lower()
    datemax = df1.date.max()
    return df1, datemax
    
    
    
def rfm_model(data, end_date, f, p):
    rfm1 = lifetimes.utils.summary_data_from_transaction_data(
    data,
    'customer_id',
    'date',
    monetary_value_col='amount',
    observation_period_end=end_date,
    freq=f)
    rfm1 = rfm1[rfm1.monetary_value < 600]
    bgf = BetaGeoFitter(penalizer_coef=p)
    bgf.fit(rfm1['frequency'], rfm1['recency'], rfm1['T'])
    return rfm1, bgf


def rfm_predict(rfm_table, bgf_model, t):
    rfm_table['predicted_num_trxn'] = bgf_model.conditional_expected_number_of_purchases_up_to_time(t, rfm_table['frequency'], rfm_table['recency'], rfm_table['T'])
    rfm_table['probability_alive'] = bgf_model.conditional_probability_alive(rfm_table['frequency'], rfm_table['recency'], rfm_table['T'])
    rfm_table.sort_values(by='predicted_num_trxn')
    return rfm_table


def gg_model(rfmmod, bgf, p, f):
    # Build the Model
    ret_cust = rfmmod[(rfmmod['frequency'] > 0) & (rfmmod['monetary_value'] > 0)]
    ggf = GammaGammaFitter(penalizer_coef = p)
    ggf.fit(ret_cust['frequency'],ret_cust['monetary_value'])
    pred_clt = ggf.customer_lifetime_value(
        bgf, 
        ret_cust['frequency'],
        ret_cust['recency'],
        ret_cust['T'],
        ret_cust['monetary_value'],
        time=12, # months
        freq=f,
        discount_rate=0.01)
    ret_cust['predicted_cltv'] = pred_clt
    ret_cust['exp_profit'] = ggf.conditional_expected_average_profit(ret_cust['frequency'],ret_cust['monetary_value'])
    ret_cust = ret_cust.sort_values('predicted_cltv', ascending=False).round(3)
    return ret_cust


def merge(df_rfm, df_holdout, thresh):
    df_new = pd.merge(df_rfm, df_holdout, on='customer_id')
    df_new.sort_values('predicted_cltv', ascending=False)
    df_new_red = df_new[(df_new.predicted_cltv < thresh) & (df_new.amount < thresh)]
    return df_new, df_new_red
    
def send_data(results):
    engine = sa.create_engine('redshift+psycopg2://'+rs_dict['user']+':'+rs_dict['password']+
                          '@'+rs_dict['host']+':'+rs_dict['port']+'/'+rs_dict['dbname'], connect_args={'sslmode': 'prefer'})
    con = engine.raw_connection()
    results.to_sql('predict_ltv', con=engine, schema='drp_staging', method='multi', chunksize=50000, index=False, if_exists='replace')
    print("database updated!")



def get_holdout(df):
    hold_end = df.date.max()
    hold_begin = hold_end-pd.to_timedelta(365, unit='d')
    df_hold1 = df[(df.date > hold_begin) & (df.date < hold_end)]
    df_hold2 = df_hold1.groupby('customer_id').agg({'orders':'count', 'amount':'sum'}).reset_index()
    df_hold2.columns = ['customer_id', 'trxn', 'amount']
    return df_hold2, hold_begin, hold_end





# Get the data. Set the first parameter to get a random sample of the dataset.
# (use this for extremely large datasets.)
df, max_date = get_data(1, rs_dict)

# set values 
#time horizon in the unit frequency (i.e. Days, Months, Weeks, etc.)
t = 12

# Frequecy 
frq = 'M'

# Holdout dataset
df_hold, h_beg, h_end = get_holdout(df)



# get stop date for last year
max_last = max_date-pd.to_timedelta(365, unit='d')
print(max_last)
# run NB the model: 
rfm_mod, bgf_mod = rfm_model(df, max_last, frq, 0.01)
bgf_mod.summary


# set the time period for predicting transactions / probability alive
t = 12

rfm_pred = rfm_predict(rfm_mod, bgf_mod, t)
print('Predicted rfm shpae:', rfm_pred.shape)

# This is the final CLTV Predictions that need to be pushed back to the database.
# this has the 0 frequency transactions removed...
rc = gg_model(rfm_pred, bgf_mod, 0.01, frq)
rc


print(f"Expected Average Sales: {rc['exp_profit'].mean()}")
print(f"Actual Average Sales: {rc['monetary_value'].mean()}")



df2 = df.drop_duplicates(subset=['email']) 
df2.shape#df.drop_duplicates(subset=['brand'])
df_final = rc.merge(df2[['customer_id','email']], on='customer_id')
df_final = df_final.merge(rfm_cluster[['customer_id','clusterID']], on='customer_id')

df_final['wholesaler'] = np.where(df_final['predicted_cltv'] < 1000, 0, 1)
df_final['churn_group'] = np.where(df_final['probability_alive'] < .5, 0, 1)

df_final


# Plots and Validation

plot_period_transactions(bgf_mod)



cal_hold = calibration_and_holdout_data(df, 
                                   'customer_id', 
                                   'date',
                                   calibration_period_end='2018-12-31', #3 years calibration
                                   observation_period_end='2020-12-31', #2 year holdout
                                   freq = frq)

# plots the efficiacy of the model using the hold-out period
plt.rcParams['figure.figsize'] = (20,10)
bgf = BetaGeoFitter()
bgf.fit(cal_hold['frequency_cal'], cal_hold['recency_cal'], cal_hold['T_cal'])
plot_calibration_purchases_vs_holdout_purchases(bgf, cal_hold)

fig = plt.figure(figsize=(8,6))
plot_frequency_recency_matrix(bgf_mod)


fig = plt.figure(figsize=(8,6))
plot_probability_alive_matrix(bgf_mod)






























