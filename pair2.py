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
from sklearn.ensamble import RandomForestClassifier
import numpy as np
import pandas as pd


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


def rmsle(y_hat, y):
    """Calculate the root mean squared log error between y
    predictions and true ys.
    """
    pass


if __name__ == '__main__':
    df = pd.read_csv('data/churn_train.csv')
    
    df = df.set_index('SalesID').sort_index()
    y = df.SalePrice

    # This is for predefined split... we want -1 for our training split,
    # 0 for the test split.
    cv_cutoff_date = pd.to_datetime('2011-01-01')
    cv = -1*(pd.to_datetime(df.saledate) < cv_cutoff_date).astype(int)

    cross_val = PredefinedSplit(cv)

    p = Pipeline([
        ('filter', FilterColumns()),
        ('type_change', DataType()),
        ('replace_outliers', ReplaceOutliers()),
        ('compute_age', ComputeAge()),
        ('nearest_average', ComputeNearestMean()),
        ('columns', ColumnFilter()),
        ('rf', RandomForestClassifier())
    ])
    df = df.reset_index()

    # GridSearch
    params = {'n_estimators': [100, 200, 500],
             'max_depth': [3, 5, 7],
             'max_features': ['auto', 'sqrt', 'log2']}

    # Turns our rmsle func into a scorer of the type required
    # by gridsearchcv.
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    gscv = GridSearchCV(p, params,
                        n_jobs=-1,
                        scoring=rmsle_scorer,
                        cv=cross_val)
    clf = gscv.fit(df.reset_index(), y, n_jobs=-1)

    print('Best parameters: {}'.format(clf.best_params_))
    print('Best RMSLE: {}'.format(clf.best_score_))

    test = pd.read_csv('data/test.csv')
    test = test.sort_values(by='SalesID')

    test_predictions = clf.predict(test)
    test['SalePrice'] = test_predictions
    outfile = 'data/solution_benchmark.csv'
    test[['SalesID', 'SalePrice']].to_csv(outfile, index=False)