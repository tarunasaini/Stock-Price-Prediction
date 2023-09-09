import pandas as pd
import numpy as np
import graphviz 
from datetime import datetime
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
#from TechnicalIndicators import TechnicalIndicators
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate
from xgboost.sklearn import XGBClassifier

class Model():
    
    def __init__(self):
        '''
        Define the Classifiers to be Used for 
        @Classifiers:
                    List of Tuples
        @Pipeline: Channel of Estimators
        @Employ the use of GridSearchCV
        Predicting Returns
        '''
        self.KERNELS = ['linear', 'rbf']
        self.GAMMA = [0.0001, 0.001, 0.01, 1]
        self.CRITERION = ['gini', 'entropy']
        self.RANGE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.N_VALIDATION = 2
        self.BEST_ACCURACY = 0.0
        self.BEST_GRIDSEARCH = ''
        self.BEST_CLASSIFIER = 0
        
        self.pipe_SVC = Pipeline([('normalizer', StandardScaler()), ('clf', SVC())])

        self.pipe_LogisticRegression = Pipeline([('normalizer', StandardScaler()), ('clf', LogisticRegression())])
        self.pipe_DecisionTreeClassifier = Pipeline([('normalizer', StandardScaler()), ('clf', DecisionTreeClassifier())])
    
        self.pipe_RandomForestClassifier = Pipeline([('normalizer', StandardScaler()), ('clf', RandomForestClassifier())])
        
        self.pipe_SVC_params = [{'clf__kernel': self.KERNELS,
                                'clf__C': self.RANGE,
                                'clf__gamma': self.GAMMA}]
        
        self.pipe_RandomForestClassifier_params = [{'clf__criterion': self.CRITERION,
                                                     'clf__max_depth': np.arange(2,10),
                                                     'clf__min_samples_split': np.arange(2,10),
                                                     'clf__min_samples_leaf': np.arange(2,10),
                                                     'clf__n_estimators': np.arange(1,2)}]
        self.pipe_DecisionTreeClassifier_param = [{'clf__max_depth': np.arange(2,10),
                                                    }]
        self.pipe_LogisticRegression_params = [{'clf__penalty': ['l1', 'l2'],
                                        		'clf__C': [1.0, 0.5, 0.1], 'clf__solver': ['liblinear']}]
        
        
    def optimize(self, X_train, X_test, Y_train, Y_test):
        '''
        Here we call the GridSearchCV class to get
        the best parameters or better still optimized parameters
        for our data.
        Remember the Gridsearch is done througk the pipeline.
        '''
        
        self.grid_RandomForestClassifier = GridSearchCV(estimator = self.pipe_RandomForestClassifier, 
                                                        param_grid = self.pipe_RandomForestClassifier_params,
                                                        scoring='accuracy',	cv = self.N_VALIDATION)

        self.grid_SVC = GridSearchCV(estimator = self.pipe_SVC, param_grid = self.pipe_SVC_params,
                                             scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_LogisticRegression = GridSearchCV(estimator = self.pipe_LogisticRegression, 
                                                    param_grid = self.pipe_LogisticRegression_params,
                                                    scoring='accuracy',	cv = self.N_VALIDATION)
        
        self.grid_DecisionTreeClassifier = GridSearchCV(estimator = self.pipe_DecisionTreeClassifier,
                                                        param_grid = self.pipe_DecisionTreeClassifier_param,
                                                        scoring='accuracy',	cv = self.N_VALIDATION)
        self.All_grids = {'grid_RandomForestClassifier': self.grid_RandomForestClassifier,
                          'grid_SVC': self.grid_SVC, 
                          'grid_LogisticRegression': self.grid_LogisticRegression,
                          'grid_DecisionTreeClassifier': self.grid_DecisionTreeClassifier,
                          }
        
        print('--------------------------------------------------------')
        print('\tPerforming optimization...')
        for classifier_grid_name, classifier_grid in self.All_grids.items():
            print('--------------------------------------------------------')
            print('Classifier: {}'.format(classifier_grid_name))	
        	# Fit grid search	
            classifier_grid.fit(X_train, Y_train)
        	# Best params
            print('Best params: {}'.format(classifier_grid.best_params_))
        	# Best training data accuracy
            print('Best training accuracy: {}'.format(classifier_grid.best_score_))
        	# Predict on test data with best params
            Y_Prediction = classifier_grid.predict(X_test)
        	# Test data accuracy of model with best params
            print('Test set accuracy score for best params: {}'.format(accuracy_score(Y_test, Y_Prediction)))
            print('--------------------------------------------------------')
        	# Track best (highest test accuracy) model
            if accuracy_score(Y_test, Y_Prediction) > self.BEST_ACCURACY:
                self.BEST_ACCURACY = accuracy_score(Y_test, Y_Prediction)
                self.BEST_GRIDSEARCH = classifier_grid
                self.BEST_CLASSIFIER = classifier_grid_name
        print('\nClassifier with best test set accuracy: {}'.format(self.BEST_CLASSIFIER))
        
        return self.BEST_GRIDSEARCH
