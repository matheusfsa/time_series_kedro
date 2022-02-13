import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from statsmodels.tsa.forecasting.theta import ThetaModel

from typing import Dict, Tuple


class Theta(RegressorMixin, BaseEstimator):
    def __init__(self, period=None, method=False):
        self.period = period
        self.method = method

    def fit(self, y, X=None):
        
        if self.method == "multiplicative":
            y += 1
        self._model = ThetaModel(y,
                                period=self.period,
                                method=self.method).fit()
        
        return self
        
        
    def predict(self, n_periods, X=None):
        y_pred = self._model.forecast(n_periods)
        if self.method == "multiplicative":
            return y_pred - 1
        else:
            return y_pred
    
                    
    def get_params(self, deep=True):
        
        self.parameters = {'method':self.method,
                           'period':self.period}
    
        return self.parameters

    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

