""" This solution makes heavy use of sklearn's Pipeline class.
    You can find documentation on using this class here:
    http://scikit-learn.org/stable/modules/pipeline.html
"""
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from data_cleaning import DataCleaning, accuracy, recall
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    # creating dependent churn variables
    # labelled customers churned if they hadn't used the service in the last
    # month

    condition = df['last_trip_date'] < '2014-06-01' 
    df['churn'] = 1
    df.loc[~condition, 'churn'] = 0
    y = df['churn']
    clean = DataCleaning()
    df = clean.transform(df)
    
    #p = Pipeline([
    #    ('dc', DataCleaning()),
    #    ('rf', RandomForestClassifier())
    #])
    
    # GridSearch for RF
    params = {'n_estimators': [100, 200, 500],
             'max_depth': [3, 5, 7],
             'max_features': ['auto', 'sqrt', 'log2']}

    gb_params = {'learning_rate': [0.01, 0.1, 1],
                'n_estimators' : [50, 100],
                'subsample' : [0.5, 1],
                'max_depth' : [3, 5],
                'max_features' : ['auto', 'sqrt', 'log2']
    
    }

    lr_params = { 'C' : [1, 2, 3, 4, 5]
                
    }

    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    lr = LogisticRegression()

    acc_scorer = make_scorer(accuracy)

    # gscv = GridSearchCV(estimator=rf,
    #                     param_grid=params,
    #                     n_jobs=-1,
    #                     scoring=acc_scorer,
    #                     cv=10)
    '''
    gscv = GridSearchCV(estimator=gb,
                    param_grid=gb_params,
                    n_jobs=-1,
                    scoring=acc_scorer,
                    cv=10)
    '''    
    gscv = GridSearchCV(estimator=lr,
                param_grid=lr_params,
                n_jobs=-1,
                scoring=acc_scorer,
                cv=10)
    
    clf = gscv.fit(df, y)
    
    model = clf.best_estimator_
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=66)
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    
    '''
    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(clf.best_score_))

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = clf.predict(test)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)
    '''
