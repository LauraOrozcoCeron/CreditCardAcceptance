import pandas as pd 
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

class ModelComparison:
    def __init__(self, df_train: pd.DataFrame, y_train, metric_score: str) -> None:
        self.df_train = df_train
        self.y_train = y_train
        self.metric_score = metric_score


    def selection_model_classifier(self):
        models = []
        models.append(('LOG', LogisticRegression(random_state=123)))
        models.append(('RF', RandomForestClassifier(min_samples_leaf= 10, class_weight="balanced",random_state = 123, criterion="gini", max_depth=6, max_features=5, n_estimators=150)))
        models.append(('GB', GradientBoostingClassifier(random_state=123,learning_rate= 0.3, max_depth=3, n_estimators= 500)))
        models.append(('XGB', XGBClassifier(random_state=123, eta=0.01,gamma=0.05,learning_rate=0.3,max_depth=3, n_estimators=500, reg_alpha=0, reg_lambda= 0.1 )))


        X = self.df_train.values

        # evaluate each model in turn
        results = []
        names = []
        scoring = self.metric_score
        for name, model in models:
            cv = RepeatedKFold(n_splits=10,  n_repeats=3,random_state=123)
            cv_results = cross_val_score(model, X, self.y_train, cv=cv, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        # boxplot algorithm comparison
        fig = plt.figure()
        fig.suptitle('Algorithm Comparison')
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        plt.show()

        #Devuelve el score y desviacion estandar para cada modelo evaluado
