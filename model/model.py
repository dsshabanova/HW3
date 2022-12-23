
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from connector.connector import get_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import sys
import pandas as pd
from pathlib import Path
if sys.path[0][-3:] != 'HW3':
    sys.path[0] = str(Path(sys.path[0]).parent)
from connector.connector import get_data
from conf.tuning import settings
from util.utility import spliting_data
from util.utility import save_model
from conf.tuning import logging as lg
from conf.settings_model import models
from sklearn.metrics import accuracy_score


import logging 


def train_test_split(df):
    X = df.iloc[:, :-1]
    y = df['target']
    logging.info("Split into train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    return X_train, X_test, y_train, y_test


def train_model(Model, Data:pd.DataFrame, target:str, list_of_param: list) -> None:
    """
    This function find the best model with the given set of parametries and data, after that save it in dir
    """
    featured_model = models[Model] #finding models in dictionary
    splited_data = spliting_data(Data.drop(target, axis=1), Data[target]) #split the data
    lg.info('Data was splitted')
    searcher = GridSearchCV(featured_model(), list_of_param, scoring="neg_root_mean_squared_error", cv=settings.cv) #give gyperparameters to our GridSearch
    searcher.fit(splited_data[0], splited_data[2]) #fit GridSearch
    lg.info(f'Model fitted, best score = {accuracy_score(splited_data[3], searcher.predict(splited_data[1]))}')
    best_model = searcher.best_estimator_ #getting best model
    save_model(settings.dir+f'/model={Model}.pkl', best_model) #save best model
    lg.info(f'Best model was saved')





def train_model(X_train, y_train,Data:pd.DataFrame,target, params):
    splited_data = spliting_data(Data.drop(target, axis=1), Data[target])
    model = RandomForestClassifier(random_state = 42)
    model.set_params(**params)
    logging.info("Train Random Forest")
    model.fit(X_train, y_train)
    save_model('model/conf/RandomForesеt.pkl', model)
    searcher = GridSearchCV(model(), params, scoring="neg_root_mean_squared_error", cv=settings.cv)
    searcher.fit(splited_data[0], splited_data[2])
    logging.info("Gridsearch")
    return model

def train_model2(X_train, y_train,Data:pd.DataFrame,target, params):
    splited_data = spliting_data(Data.drop(target, axis=1), Data[target])
    model = LogisticRegression(solver = 'liblinear', random_state = 2)
    model.set_params(**params)
    logging.info("Train Logisti Regression")
    model.fit(X_train, y_train)
    save_model('model/conf/LogisticRegression.pkl', model)
    searcher = GridSearchCV(model(), params, scoring="neg_root_mean_squared_error", cv=settings.cv)
    searcher.fit(splited_data[0], splited_data[2])
    logging.info("Gridsearch")
    return model

from util.util import save_model, load_model



def prediction(model, values):
    model = load_model(settings.dir, model)
    train_model(model, data, settings.target, settings[model]) 
    logging.info("Prediction")
    return model.predict(values)