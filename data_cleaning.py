import pandas as pd
import numpy as np

'''
This script has functions to clean the test and train data
as well as engineer new features, for use in the plots and churn_model
scripts
'''
def create_target(df):
    # creating dependent churn variables
    # labelled customers churned if they hadn't used the service in the last
    # month

    condition = df['last_trip_date'] < '2014-06-01' 
    df['churn'] = 1
    df.ix[~condition, 'churn'] = 0
    return df

def clean_data(df):
    '''
    INPUT: UNCLEANED PANDAS DF with target label
    OUTPUT: CLEANED PANDAS DF with null value
    '''

    # convert to datetime
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    # convert to 1/0
    df['luxury_car_user'] = df['luxury_car_user'].astype(int)

    # add missing value to phone device
    df['phone'].fillna(value='missing', inplace=True)

    # Filling missing values for avg_rating_of_driver
    df['avg_rating_of_driver'].fillna(-1,inplace = True)
    
    # Filling missing values for avg_rating_by_driver
    df['avg_rating_by_driver'].fillna(-1,inplace = True)
    
    # Create new colums indicating ratings or non-ratings
    condition_1 = df['avg_rating_of_driver'] == -1 
    df['rating_of_driver'] = 0
    df.ix[~condition_1, 'rating_of_driver'] = 0
    condition_2 = df['avg_rating_by_driver'] == -1 
    df['rating_by_driver'] = 0
    df.ix[~condition_2, 'rating_by_driver'] = 0
    
    return df

