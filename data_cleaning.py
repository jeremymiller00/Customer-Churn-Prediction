import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

'''
This script has functions to clean the test and train data
as well as engineer new features, for use in the plots and churn_model
scripts
'''
class DataCleaning(BaseEstimator, TransformerMixin):
#     def get_params(self, **kwargs):
#         pass
        
    def fit(self, df, y):
        return self

    def transform(self, df):
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
        df.loc[~condition_1, 'rating_of_driver'] = 0
        condition_2 = df['avg_rating_by_driver'] == -1 
        df['rating_by_driver'] = 0
        df.loc[~condition_2, 'rating_by_driver'] = 0
        
        cols_to_be_kept = ['avg_dist', 'avg_rating_by_driver', 'rating_by_driver',\
                           'avg_rating_of_driver', 'rating_of_driver',\
                           'avg_surge','city', 'phone', 'surge_pct','trips_in_first_30_days',\
                           'luxury_car_user', 'weekday_pct']
        X = df[cols_to_be_kept]

        cat_cols = ['phone', 'city']
        for col in cat_cols:
            df[col] = df[col].astype('category')
        X = pd.get_dummies(X, columns=cat_cols)
        return X