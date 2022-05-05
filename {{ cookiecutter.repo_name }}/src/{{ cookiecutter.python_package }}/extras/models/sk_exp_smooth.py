import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETSModel

from typing import Dict, Tuple


class ExponentialSmoothing(RegressorMixin, BaseEstimator):
    '''

    '''
    def __init__(self, trend=None, damped_trend=False, seasonal=None, seasonal_periods=None):
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        

            
    def fit(self, y, X=None):
        if not self.trend  and self.damped_trend:
            raise ValueError("Exponential Smoothing cannot be fitted with 'damped trend=True' when the 'trend' parameter is None.")
        
        if self.trend == "mul" or self.seasonal == "mul":
            y += 1
        self._model = ETSModel(y,
                            trend = self.trend,
                            damped_trend = self.damped_trend,
                            seasonal = self.seasonal,
                            seasonal_periods = self.seasonal_periods).fit(optimized=True)
        
        return self
        
        
    def predict(self, n_periods, X=None):
        y_pred = self._model.forecast(n_periods)
        if self.trend == "mul" or self.seasonal == "mul":
            return y_pred - 1
        else:
            return y_pred
    
                    
    def get_params(self, deep=True):
        
        self.parameters = {'trend': self.trend,
                           'damped_trend': self.damped_trend,
                           'seasonal': self.seasonal,
                           'seasonal_periods': self.seasonal_periods}
    
        return self.parameters

    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

