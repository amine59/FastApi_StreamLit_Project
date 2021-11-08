# 1. Library imports
#https://betterdatascience.com/how-to-build-and-deploy-a-machine-learning-model-with-fastapi/

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pydantic import BaseModel
import joblib

class IrisSpecies(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisModelRandomForest:
    def __init__(self):
        self.df = pd.read_csv('C:/Users/client/Desktop/Amaris\iris.csv')
        self.model_fname_ = 'iris_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self._train_model()
            joblib.dump(self.model, self.model_fname_)

    def _train_model(self):
        X = self.df.drop('species', axis=1)
        y = self.df['species']
        rfc = RandomForestClassifier()
        model = rfc.fit(X, y)
        return model

    def predict_RandomForest(self, sepal_length, sepal_width, petal_length, petal_width):
        data_in = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = self.model.predict(data_in)
        probability = self.model.predict_proba(data_in).max()
        return prediction[0], probability

class IrisModelLogisticRegression:
    def __init__(self):
        self.df = pd.read_csv('C:/Users/client/Desktop/Amaris\iris.csv')
        self.model_fname_ = 'iris_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self.train_model()
            joblib.dump(self.model, self.model_fname_)


        self.X = self.df.drop('species', axis=1)
        self.y = self.df['species']
        self.iris_type = {
                    0: 'setosa',
                    1: 'versicolor',
                    2: 'virginica'
                }

    def train_model(self) -> LogisticRegression:
        LogisticRegression(solver='lbfgs',
                                  max_iter=1000,
                                  multi_class='multinomial').fit(self.X, self.y)

    def predict_LogisticRegression(self,features: dict):
        X = [features['sepal_length'], features['sepal_width'], features['petal_length'], features['petal_width']]
        prediction = self.model.predict_proba([X])
        return {self.iris_type[np.argmax(prediction)], round(max(prediction[0]), 2)}

class DecisionTreeClassifier:
    def __init__(self):
        self.df = pd.read_csv('C:/Users/client/Desktop/Amaris\iris.csv')
        self.model_fname_ = 'iris_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self.train_model()
            joblib.dump(self.model, self.model_fname_)


        self.X = self.df.drop('species', axis=1)
        self.y = self.df['species']
        self.iris_type = {
                    0: 'setosa',
                    1: 'versicolor',
                    2: 'virginica'
                }

    def train_model(self) -> DecisionTreeClassifier:
        DecisionTreeClassifier().fit(self.X, self.y)

    #self.clf = train_model()

    def predict_DecisionTree(self,features: dict):
        X = [features['sepal_length'], features['sepal_width'], features['petal_length'], features['petal_width']]
        prediction = self.model.predict_proba([X])
        return {self.iris_type[np.argmax(prediction)], round(max(prediction[0]), 2)}


class SupportVectorMachines:
    def __init__(self):
        self.df = pd.read_csv('C:/Users/client/Desktop/Amaris\iris.csv')
        self.model_fname_ = 'iris_model.pkl'
        try:
            self.model = joblib.load(self.model_fname_)
        except Exception as _:
            self.model = self.train_model()
            joblib.dump(self.model, self.model_fname_)


        self.X = self.df.drop('species', axis=1)
        self.y = self.df['species']
        self.iris_type = {
                    0: 'setosa',
                    1: 'versicolor',
                    2: 'virginica'
                }

    def train_model(self) -> SVC:
        SVC(gamma='auto').fit(self.X, self.y)

    #self.clf = train_model()

    def predict_SupportVectorMachines(self,features: dict):
        X = [features['sepal_length'], features['sepal_width'], features['petal_length'], features['petal_width']]
        prediction = self.model.predict_proba([X])
        return {self.iris_type[np.argmax(prediction)], round(max(prediction[0]), 2)}