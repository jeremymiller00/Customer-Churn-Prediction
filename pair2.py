""" This solution makes heavy use of sklearn's Pipeline class.
    You can find documentation on using this class here:
    http://scikit-learn.org/stable/modules/pipeline.html
"""
from datetime import timedelta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from data_cleaning import DataCleaning

class DataType(BaseEstimator, TransformerMixin):
    """Cast the data types of the id and data source columns to strings
    from numerics.
    """
    col_types = {'str': ['MachineID', 'ModelID', 'datasource']}

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass


class FilterColumns(BaseEstimator, TransformerMixin):
    """Only keep columns that don't have NaNs.
    """
    def fit(self, X, y):
        pass

    def transform(self, X):
        pass


class ReplaceOutliers(BaseEstimator, TransformerMixin):
    """Replace year made when listed as earlier than 1900, with
    mode of years after 1900. Also add imputation indicator column.
    """
    def fit(self, X, y):
        pass

    def transform(self, X):
        pass


class ComputeAge(BaseEstimator, TransformerMixin):
    """Compute the age of the vehicle at sale.
    """
    def fit(self, X, y):
        pass

    def transform(self, X):
        pass


class ComputeNearestMean(BaseEstimator, TransformerMixin):
    """Compute a mean price for similar vehicles.
    """
    def __init__(self, window=5):
        self.window = window

    def get_params(self, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def transform(self, X):
        pass


class ColumnFilter(BaseEstimator, TransformerMixin):
    """Only use the following columns.
    """

    def fit(self, X, y):
        # Get the order of the index for y.
        pass

    def transform(self, X):
        pass


def accuracy(y_hat, y):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    """
    return np.mean(y_hat == y)


if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    # creating dependent churn variables
    # labelled customers churned if they hadn't used the service in the last
    # month

    condition = df['last_trip_date'] < '2014-06-01' 
    df['churn'] = 1
    df.ix[~condition, 'churn'] = 0
    y = df['churn']
    
    p = Pipeline([
        ('dc', DataCleaning()),
        ('rf', RandomForestClassifier())
    ])
    '''
    # GridSearch
    params = {'n_estimators': [100, 200, 500],
             'max_depth': [3, 5, 7],
             'max_features': ['auto', 'sqrt', 'log2']}
    '''
    params = {'rf__n_estimators': [100, 200, 500]}

    # Turns our rmsle func into a scorer of the type required
    # by gridsearchcv.
    acc_scorer = make_scorer(accuracy)

    gscv = GridSearchCV(estimator=p,
                        param_grid=params,
                        n_jobs=-1,
                        scoring=acc_scorer,
                        cv=10)
    
    clf = gscv.fit(df, y)

    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(clf.best_score_))

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = clf.predict(test)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)