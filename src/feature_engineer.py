import pandas as pd
import numpy as np


def feature_engineer(data):

    y = data['churn']
    data.drop(['churn'],axis=1,inplace=True)

    cols_to_be_kept = ['avg_dist', 'avg_rating_by_driver', 'rating_by_driver',\
                       'avg_rating_of_driver', 'rating_of_driver',\
                       'avg_surge','city', 'phone', 'surge_pct','trips_in_first_30_days',\
                       'luxury_car_user', 'weekday_pct']
    X = data[cols_to_be_kept]

    cat_cols = ['phone', 'city']
    for col in cat_cols:
        data[col] = data[col].astype('category')
    X = pd.get_dummies(X, columns=cat_cols)
    
    return X, y
