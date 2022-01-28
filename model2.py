from sklearn.linear_model import LinearRegression
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from applicationlogging import logger
from sklearn.model_selection import train_test_split, GridSearchCV
def trainingModel(self):

        # Logging the start of Training
    self.train= self.pd.read_csv('new_train.csv') # loading the training data

    Y = self.train['count']
    X = self.train.drop(columns = ['count'])

        
    X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.25,random_state=255)

    Rf = RandomForestRegressor(n_estimators = 100,criterion='mse',random_state=255,max_depth=50,min_samples_split=10, verbose=3,oob_score=True)
    Rf.fit(X_train,y_train)
    filename = 'finalized_bike_model.pickle'
    pickle.dump(Rf, open(filename, 'wb'))

    # saving the model to the local file system
