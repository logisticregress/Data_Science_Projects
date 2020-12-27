########################################################################
#
# Author: R. Clukey 
# Date: May 9, 2020 
# Description: Python Record Linkage using Fuzzy Matching to dedupe customer_id records
# EC2 Cron job parameters: crontab -e 0 0 * * 7 python3 ~/Householding_python_v2.py

#* * * * * command to be executed
#- - - - -
#| | | | |
#| | | | ----- Day of week (0 - 7) (Sunday=0 or 7)
#| | | ------- Month (1 - 12)
#| | --------- Day of month (1 - 31)
#| ----------- Hour (0 - 23)
#------------- Minute (0 - 59)
#

# to disable: crontab -l | awk '{print "# "$1}' | crontab 

########################################################################
## this is the code to copy the file to the EC2 instance:
## scp -P 19022 -i /users/ryanclukey/ec2-householding.pem /users/ryanclukey/documents/python_projects/Record_Matching/Householding_python_v2.py ...:

# for the .YML file
## scp -P 19022 -i /users/ryanclukey/ec2-householding.pem /users/ryanclukey/documents/python_projects/Record_Matching/python_db.yml ...:


#This method uses de-duplication algorithms which are part of the Python 
#RecordLinkage toolkit to identify records which are likely duplicates 


#Load the necessary libraries 
import pandas as pd
import numpy as np
import psycopg2
import time
import datetime as dt
import sqlalchemy as sa
import snowflake.connector as sf
import sqlalchemy as sa
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import yaml
from yaml import load, dump
import boto3
import psycopg2
import recordlinkage as rl
import sys
#import jellyfish.cjellyfish
from recordlinkage.preprocessing import clean
from recordlinkage.preprocessing import phonenumbers
import argparse


#############

parser = argparse.ArgumentParser("Database Credentials as Parameters")
parser.add_argument("dbtype", help="Redshift or Snowflake",type=str)
parser.add_argument("dbname", help="the name of the databse",type=str)
parser.add_argument("host", help="the host",type=str)
parser.add_argument("port", help="the port",type=str)
parser.add_argument("user", help="the user name",type=str)
parser.add_argument("password", help="the password",type=str)
parser.add_argument("account", help="Snowflake needs the account name... ",type=str)
parser.add_argument("warehouse", help="choose the warehouse name... ",type=str)
args = parser.parse_args()

# Server Credentials...


# The main dedupe function that runs all the sub functions. This also determines if the main dedupe should 
# be run, or only the incremental. 
def df_chunk(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def run_dedupe(engine, db_name, sql_full, sql_inc):
    pyth = engine.dialect.has_table(engine, 'python_cstmr_lkp', schema = 'drp_staging')
    if pyth == True: # if the Python table already exists
        df = []  
        for chunk in pd.read_sql(sql_inc, con, chunksize=100000):
            df.append(chunk)

        # Start appending data from list to dataframe
        df_rs = pd.concat(df, ignore_index=True)
        print("starting incremental match process...")
        df_new, df_old = split_records(df_rs)
        if df_new.shape[0] > 0:
            df_new_c = clean_data(df_new)
            df_old_c = clean_data(df_old)
            matches_new = incremental_match(df_new_c, df_old_c)
            new_merged = incremental_merge(matches_new, df_old_c, df_new_c)
            if db_name == 'redshift':
                new_merged.to_sql('python_cstmr_lkp', con=engine, schema='drp_staging', method='multi', chunksize=100000, index=False, if_exists='append')
            elif db_name == 'snowflake':
                new_merged.to_sql('python_cstmr_lkp', con=engine, schema='drp_staging', method='multi', chunksize=10000, index=False, if_exists='append')
            print("number of new records:", len(new_merged))
            print("number total records:", len(df_rs))
            return new_merged
        else: 
            print("no records to match")
            pass
    elif pyth == False: # if the Python table doesn't exist 
        df = []
        complete_all = []
        n = 500000
        count_chunk = 0
        for chunk in pd.read_sql(sql_full, con, chunksize=500000):
            print("starting chunked match process...")
            print("Chunk Number:", count_chunk)
            df_chk = pd.DataFrame(chunk)
            count_chunk = count_chunk+1
            df_c = clean_data(df_chk)
            print("data cleaned...")
            df_c = df_c[df_c['customer_id'].notna()]
            df1, matched, dupe_emails = match_data_chunk(df_c, 1)
            complete = merge_sets(df1, matched, dupe_emails)
            print("data merged...")
            complete_all.append(complete)
            print("chunk ", count_chunk, " done.")
        all_records = pd.concat(complete_all, ignore_index=True)
        print("total records processed: ", all_records.shape[0])
        if db_name == 'redshift':
            all_records.to_sql('python_cstmr_lkp', con=engine, schema='drp_staging', method='multi', chunksize=100000, index=False, if_exists='replace')
        elif db_name == 'snowflake':
            all_records.to_sql('python_cstmr_lkp', con=engine, schema='drp_staging', method='multi', chunksize=10000, index=False, if_exists='replace')
        return all_records


# FUNCTION DEFINITION: Initial Matching 

#Cleaning function - removes unecessary characters, spacing and symbols from the fields
def clean_data(df):
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df[df['customer_id'].notna()]
    df['phone_clean'] = phonenumbers(df['phone'])
    df['first_name_clean'] = clean(df['first_name'])
    df['last_name_clean'] = clean(df['last_name'])
    df['address_clean'] = clean(df['address'])
    df['city_clean'] = clean(df['city'])
    df['state_clean'] = clean(df['state'])
    df['zip_clean'] = clean(df['zip'].str.split('-').str[0])
    return df 

# this function removes the deuplicate emails from the dataframe, and then
#the dataframe output from the previous cleaning function should be entered into the function below

def match_data_chunk(df, s):
    count_pass = 0
    start = time.process_time()
    all_matches = []
    df0=df[df['email'].isnull() | ~df[df['email'].notnull()].duplicated(subset='email',keep=False)]
    mask = df.email.duplicated(keep=False)
    d = df[mask]
    dupe_emails = d[['customer_id', 'email']].dropna(subset=['email'])
    df1 = df0.sample(frac=s, replace=False, random_state=42)
    df1.set_index('customer_id', inplace=True)
    dupe_indexer = rl.Index()

    #set the blocking system
    dupe_indexer.block(left_on=['first_name_clean','last_name_clean','zip_clean','state_clean'])
    dupe_indexer.block(left_on=['first_name_clean','last_name_clean'])
    dupe_indexer.block(left_on=['zip_clean','first_name_clean'])
    dupe_indexer.block(left_on=['zip_clean','last_name_clean'])
    #dupe_indexer.block(left_on=['zip_clean', 'state_clean'])
    dupe_indexer.block(left_on=['last_name_clean', 'state_clean'])
    #dupe_indexer.block(left_on=['zip_clean', 'phone_clean'])
    dupe_candidate_links = dupe_indexer.index(df1)
    print("total candidate links:", len(dupe_candidate_links))

    # split the pair indedx for iteration
    s = rl.index_split(dupe_candidate_links, 20)
    for chunk in s:    
        compare_dupes = rl.Compare(n_jobs=8)
        compare_dupes.string('first_name_clean','first_name_clean', method='jarowinkler', threshold=0.92, label='first_name_cl')
        compare_dupes.string('last_name_clean','last_name_clean', method='jarowinkler', threshold=0.85, label='last_name_cl')
        compare_dupes.string('email', 'email', method='jarowinkler', threshold=0.90, label='email_cl')
        compare_dupes.string('address_clean','address_clean', method='jarowinkler', threshold=0.7, label='address_cl')
        compare_dupes.string('city_clean','city_clean', method='jarowinkler',threshold=0.85, label='city_cl')
        compare_dupes.string('state_clean','state_clean', method='jarowinkler',threshold=0.85, label='state_cl')
        compare_dupes.exact('zip_clean','zip_clean',label='zip_cl')
        compare_dupes.string('phone_clean','phone_clean', method='jarowinkler', threshold=0.95, label='phone_cl')

        # create the deduped feature set - this takes a while...
        dupe_features = compare_dupes.compute(chunk, df1)

        # select those features that match

        # Business rule: of any 3 of email, address, city, state, zip, or phone match, then code it as a "duplicate"

        pdm = dupe_features[dupe_features.sum(axis=1) > 2].reset_index()  #this limits the dataset to rows that had more than 2 matched columns
        pdm['score'] = pdm.loc[:, 'email_cl':'phone_cl'].sum(axis=1) #this sums the columns and creates a "score" column 
        pdm['duplicate'] = np.where((pdm['email_cl'] == 1) | (pdm['first_name_cl'] == 1 |  (pdm['last_name_cl'] == 1)) & ((pdm['score'] > 3) | (pdm['phone_cl'] + pdm['zip_cl'] == 2)),1,0) #creates an indicator based on the threshold rule of > 2 matches (1=yes, 0=no)
        
        ne = pdm[pdm['duplicate']==1] #filter out non-matching rows
        ne.sort_values(by=['score'], ascending=False) #sort the results by the score column 

        matches = ne[['customer_id_1','customer_id_2','email_cl', 'phone_cl', 'address_cl','city_cl','state_cl','zip_cl','score','duplicate']].sort_values(by=['score'], ascending=False)
        all_matches.append(matches)
        count_pass = count_pass+1
        elapsed = time.process_time() - start
        print(count_pass, format(elapsed, '.3f'), ": seconds")
    all_matches = pd.concat(all_matches, ignore_index=True)
    
    return df1, all_matches, dupe_emails


# Function to merge the various dataframes back into a single dataframe containing all the duplicate email records
def merge_sets(df, matches, dupes):
    df_email = df['email'].reset_index()
    cid = matches[['customer_id_1', 'customer_id_2']]
    print('cid:',len(cid), 'df_email:',len(df_email))
    
    #create a merged dataframe of customer ids and emails
    merged_df = pd.merge(cid, df_email, left_on='customer_id_1', right_on='customer_id', how='left')
    # create two lists from each column.  they need to have the columns renamed in order for the concatenate to work 
    c_id1 = merged_df[['customer_id_1', 'email']].rename({'customer_id_1': 'id', 'email': 'email'}, axis=1)
    c_id2 = merged_df[['customer_id_2', 'email']].rename({'customer_id_2': 'id', 'email': 'email'}, axis=1)
    c_all = pd.concat([c_id1, c_id2], sort=False)
    c_all_deduped = c_all.drop_duplicates(['id'], keep='last').rename({'id': 'customer_id', 'email': 'email'}, axis=1)
    temp = df_email[~df_email['customer_id'].isin(c_all_deduped['customer_id'])] #removes matching records matching the deduped list
    y = temp.append(c_all_deduped) #append the new dataframe to the existing one (with original cases removed)
    complete = y.append(dupes)
    complete.insert(0, 'timestamp', dt.datetime.now().replace(microsecond=0))
    #complete['email'].fillna(y['customer_id'], inplace=True) #replace missing emails with the customer_id 
    
    return complete



# INCREMENTAL FUNCTIONS 
def split_records(df):
    df_new = df.loc[df['new_recs'] == '1'] 
    df_old = df.loc[df['new_recs'] == '0']
    return df_new, df_old


def incremental_match(df_n, df_o):
    indexer = rl.Index()
    indexer.block(left_on=['first_name_clean','last_name_clean','zip_clean','state_clean'])
    indexer.block(left_on=['first_name_clean','last_name_clean'])
    indexer.block(left_on=['zip_clean','first_name_clean'])
    indexer.block(left_on=['zip_clean','last_name_clean'])
    indexer.block(left_on=['zip_clean', 'state_clean'])
    indexer.block(left_on=['last_name_clean', 'state_clean'])
    indexer.block(left_on=['zip_clean', 'phone_clean'])
    candidate_links = indexer.index(df_n, df_o)


    compare_cl = rl.Compare(n_jobs=8)
    compare_cl.string('first_name_clean','first_name_clean', method='jarowinkler', threshold=0.92, label='first_name_cl')
    compare_cl.string('last_name_clean','last_name_clean', method='jarowinkler', threshold=0.85, label='last_name_cl')
    compare_cl.string('email', 'email', method='jarowinkler', threshold=0.90, label='email_cl')
    compare_cl.string('address_clean','address_clean', method='jarowinkler', threshold=0.70, label='address_cl')
    compare_cl.string('city_clean','city_clean', method='jarowinkler',threshold=0.85, label='city_cl')
    compare_cl.string('state_clean','state_clean', method='jarowinkler',threshold=0.85, label='state_cl')
    compare_cl.exact('zip_clean','zip_clean',label='zip_cl')
    compare_cl.string('phone_clean','phone_clean', method='jarowinkler', threshold=0.95, label='phone_cl')

    # create the deduped feature set - this takes a while...
    cl_features = compare_cl.compute(candidate_links, df_n, df_o)
    cl_summary = cl_features.sum(axis=1).value_counts().sort_index(ascending=False)

    df_match = cl_features[cl_features.sum(axis=1) > 2].reset_index()  #this limits the dataset to rows that had more than 2 matched columns
    df_match['score'] = df_match.loc[:, 'email_cl':'phone_cl'].sum(axis=1) #this sums the columns and creates a "score" column 

    #creates an indicator based on the threshold rule of > 2 matches (1=yes, 0=no)
    df_match['duplicate'] = np.where((df_match['email_cl'] == 1) | (df_match['first_name_cl'] == 1 |  (df_match['last_name_cl'] == 1)) & ((df_match['score'] > 3) | (df_match['phone_cl'] + df_match['zip_cl'] == 2)),1,0) 
    ne = df_match[df_match['duplicate']==1] #filter out non-matching rows
    ne.sort_values(by=['score'], ascending=False) #sort the results by the score column 
    return ne


def incremental_merge(matches, dfa, dfb):
    y = dfa.reset_index() #existing records to match 
    z = dfb.reset_index() # new records to assign new email to
    m_new1 = pd.merge(matches, z[['index', 'customer_id']], left_on='level_0', right_on='index')[['level_0', 'level_1', 'customer_id']]
    m_new2 = pd.merge(m_new1, y[['index','unique_id','customer_id']], left_on='level_1', right_on='index', how='left')[['customer_id_x','unique_id']]
    m_new2a = m_new2.drop_duplicates(['customer_id_x'], keep='last').rename({'customer_id_x': 'customer_id', 'unique_id': 'email'}, axis=1)
    temp = z[~z['customer_id'].isin(m_new2a['customer_id'])][['customer_id', 'email']]
    merged = temp.append(m_new2a)
    merged.insert(0, 'timestamp', dt.datetime.now().replace(microsecond=0))
    return merged


# SQL Scripts ################################################################################################
sql_rs = """select
      c.shop_id
    , c.store_source
    , c.customer_id
    , c.email as email
    , c.first_name as first_name
    , c.last_name as last_name
    , c.phone_number as phone
    , c.address1 as address
    , c.state as state
    , c.city as city
    , c.zipcode as zip
    , c.country_code as country
from uos.customers c
group by
         c.shop_id
        , c.store_source
        , c.customer_id
        , c.email
        , c.first_name
        , c.last_name
        , c.phone_number
        , c.address1
        , c.state
        , c.city
        , c.zipcode
        , c.country_code"""


rs_py = """with CTE1 AS (
    select
      c.shop_id
    , c.store_source
    , c.customer_id
    , c.email as email
    , c.first_name as first_name
    , c.last_name as last_name
    , c.phone_number as phone
    , c.address1 as address
    , c.state as state
    , c.city as city
    , c.zipcode as zip
    , c.country_code as country
    , p.timestamp
    , p.email as unique_id
    , c.updated_at
    , '1' as new_recs from uos.customers c
    left join drp_staging.python_cstmr_lkp p on c.customer_id = p.customer_id
    where c._loaded_at >= (select max(timestamp) as last_match_date from drp_staging.python_cstmr_lkp)
    AND p.customer_id IS NULL
    group by
          c.shop_id
        , c.store_source
        , c.customer_id
        , c.email
        , c.first_name
        , c.last_name
        , c.phone_number
        , c.address1
        , c.state
        , c.city
        , c.zipcode
        , c.country_code
        , p.timestamp
        , p.email
        , c.updated_at
        , new_recs
),
    CTE2 AS (
         select
          c.shop_id
        , c.store_source
        , c.customer_id
        , c.email as email
        , c.first_name as first_name
        , c.last_name as last_name
        , c.phone_number as phone
        , c.address1 as address
        , c.state as state
        , c.city as city
        , c.zipcode as zip
        , c.country_code as country
        , p.timestamp
        , p.email as unique_id
        , c.updated_at
        , '0' as new_recs from uos.customers c
         right join drp_staging.python_cstmr_lkp p on c.customer_id = p.customer_id
        group by
          c.shop_id
        , c.store_source
        , c.customer_id
        , c.email
        , c.first_name
        , c.last_name
        , c.phone_number
        , c.address1
        , c.state
        , c.city
        , c.zipcode
        , c.country_code
        , p.timestamp
        , p.email
        , c.updated_at
        , new_recs

     )
select * from CTE1 union select * from CTE2
"""



# SQL Query for Snowflake
sql_sf = """select
      c.__shop_id
    --, c.store_source
    , c.customer_id
    , c.email as email
    , c.first_name as first_name
    , c.last_name as last_name
    , c.phone_number as phone
    , c.address1 as address
    , c.state as state
    , c.city as city
    , c.zipcode as zip
    , c.country_code as country
from uos.customers c
group by
         c.__shop_id
        --, c.store_source
        , c.customer_id
        , c.email
        , c.first_name
        , c.last_name
        , c.phone_number
        , c.address1
        , c.state
        , c.city
        , c.zipcode
        , c.country_code"""

# incremental Snowflake SQL Query 
sf_py = """with CTE1 AS (
    select
    --c.__shop_id
    --, c.store_source
      c.customer_id
    , c.email as email
    , c.first_name as first_name
    , c.last_name as last_name
    , c.phone_number as phone
    , c.address1 as address
    , c.state as state
    , c.city as city
    , c.zipcode as zip
    , c.country_code as country
    , p.timestamp
    , p.email as unique_id
    , c.updated_at
    , '1' as new_recs from uos.customers c
    left join drp_staging.python_cstmr_lkp p on c.customer_id = p.customer_id
    where c.__loaded_at >= (select max(timestamp) as last_match_date from drp_staging.python_cstmr_lkp)
    AND p.customer_id IS NULL
    group by
        --c.__shop_id
        --, c.store_source
          c.customer_id
        , c.email
        , c.first_name
        , c.last_name
        , c.phone_number
        , c.address1
        , c.state
        , c.city
        , c.zipcode
        , c.country_code
        , p.timestamp
        , p.email
        , c.updated_at
        , new_recs
),
    CTE2 AS (
         select
        --c.__shop_id
        --, c.store_source
          c.customer_id
        , c.email as email
        , c.first_name as first_name
        , c.last_name as last_name
        , c.phone_number as phone
        , c.address1 as address
        , c.state as state
        , c.city as city
        , c.zipcode as zip
        , c.country_code as country
        , p.timestamp
        , p.email as unique_id
        , c.updated_at
        , '0' as new_recs from uos.customers c
         right join drp_staging.python_cstmr_lkp p on c.customer_id = p.customer_id
        group by
        --c.__shop_id
        --, c.store_source
          c.customer_id
        , c.email
        , c.first_name
        , c.last_name
        , c.phone_number
        , c.address1
        , c.state
        , c.city
        , c.zipcode
        , c.country_code
        , p.timestamp
        , p.email
        , c.updated_at
        , new_recs

     )
select * from CTE1 union select * from CTE2
"""




# STEP 2: Load the data from passing parameters 
try:
    # Establish the REDSHIFT Engine
    if args.dbtype == 'redshift':
        db_name = args.dbtype
        engine = sa.create_engine('redshift+psycopg2://'+args.user+
                                                ':'+args.password+'@'+args.host+
                                                ':'+args.port+'/'+args.dbname,
                                                  connect_args={'sslmode': 'prefer'})
        con = engine.raw_connection()

        # Run the dedupe function
        print("last revision: 6/15/2020 | 1:15pm") 
        print("starting:", args.dbtype, ":", args.user)
        start = time.process_time()
        run_dedupe(engine, db_name, sql_rs, rs_py)
        print("success::  ", time.process_time() - start)

        #close the DB connection 
        con.close()
        engine.dispose()

    elif args.dbtype == 'snowflake':
        db_name = args.dbtype
        url_sf = URL(
            host=args.host,
            user=args.user,
            password=args.password,
            account=args.account, 
            database=args.dbname,
            warehouse=args.warehouse
            )

        engine = create_engine(url_sf, connect_args = {'client_session_keep_alive': True})
        con = engine.connect()

        # Run the dedupe function 
        print("starting:", args.dbtype, ":", args.user)
        start = time.process_time()
        run_dedupe(engine, db_name, sql_sf, sf_py)
        print("success::  ", time.process_time() - start)

        #close the DB connection 
        con.close()
        engine.dispose()


except MemoryError:
    print(args.user, " FAIL: not enough memory")
    pass
