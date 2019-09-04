#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 19:04:18 2019

@author: ZLC
"""

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('stroke.csv')

#dropping nan and categoricals written in multiple ways + highly correlated features
dataset = dataset.loc[:, dataset.isnull().mean() <= .25]
dataset.drop(['area_1','area_2', 'area_3', 'area_4', 'area_5', 'pgisc_walk2', 'pgisp_walk2', 'site2', 'answer_int', 'answer_euro', 'answer_pbsi',
              'i_pbsi', 'answer_apathy', 'higheris_apathy', 'answer_RNL', 'higheris_rnln',
              'answer_gds', 'higheris_GDS', 'gdsprof', 'answer_gs', 'g_dist',
              'higheris_gs', 'i_gait', 'answer_pgi', 'area_6', 'inOA', 'inoa1', 'inoa2', 'inoa3',    
              'inoa4', 'inoa5', 'inoaprofilem', 'rehab_yes', 'language' ], axis = 1, inplace = True)

category = dataset.select_dtypes(exclude=["number","bool_",])

#imputer for missing data
from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    

x = pd.DataFrame(dataset)
df = DataFrameImputer().fit_transform(x)

#independent and dependent variables
X = df.iloc[:, df.columns != 'pbsi'].values
y = df.iloc[:, df.columns == 'pbsi'].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_X.fit_transform(y_test)

#Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#metrics
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(regressor.score(X_test, y_test))

feature_imp = pd.Series(regressor.feature_importances_).sort_values(ascending=False)
print(feature_imp)
