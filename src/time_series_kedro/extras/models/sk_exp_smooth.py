import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

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
                              seasonal_periods = self.seasonal_periods).fit(optimized=True, disp=False)
        
        return self
        
        
    def predict(self, n_periods, X=None):
        
        y_pred = self._model.forecast(n_periods)
        if self.trend == "mul" or self.seasonal == "mul":
            return y_pred - 1
        else:
            return y_pred
    
    def predict_in_sample(self, 
                          start=None,
                          end=None,
                          dynamic=False,
                          index=None,
                          method=None,
                          simulate_repetitions=1000):
        
        pred = self._model.get_prediction(start=start,
                                         end=end,
                                         dynamic=dynamic,
                                         index=index,
                                         method=method,
                                         simulate_repetitions=simulate_repetitions)
        pred = pred.summary_frame()
        pred = pred.rename(columns={"mean": "yhat", "pi_lower": "yhat_lower", "pi_upper": "yhat_upper"})
        if self.trend == "mul" or self.seasonal == "mul":
            pred["yhat"] -= 1
            pred["yhat_upper"] -= 1
            pred["yhat_lower"] -= 1
        return pred
                    
    def get_params(self, deep=True):
        
        self.parameters = {'trend':self.trend,
                           'damped_trend':self.damped_trend,
                           'seasonal':self.seasonal,
                           'seasonal_periods':self.seasonal_periods}
    
        return self.parameters

    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

