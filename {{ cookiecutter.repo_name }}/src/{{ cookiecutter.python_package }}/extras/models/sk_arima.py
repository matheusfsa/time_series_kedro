import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
import pmdarima as pm

from typing import Dict, Tuple


class ARIMA(RegressorMixin, BaseEstimator):
    '''

    '''
    def __init__(self, order=(0, 0, 0), seasonal_order=(0, 0, 0, 1)):
        self.order = order
        self.seasonal_order = seasonal_order

            
    def fit(self, y, X=None):
        self._initial_date = y.index[0]
        self._model = pm.ARIMA(order=self.order, seasonal_order=self.seasonal_order, suppress_warnings=True)
        self._model.fit(y, X=X)
        return self
        
        
    def predict(self, n_periods, X=None):
        
        return self._model.predict(n_periods, X=X)
    
    def predict_in_sample(self, 
                          start=None,
                          end=None,
                          dynamic=False,
                          index=None,
                          method=None,
                          simulate_repetitions=1000):
        
        pred = self._model.predict_in_sample(start=start,
                                             end=end,
                                             dynamic=dynamic,
                                             index=index,
                                             method=method,
                                             simulate_repetitions=simulate_repetitions,
                                             return_conf_int=True)
        output = pd.DataFrame(index=pd.date_range(start=self._initial_date, periods=end+1, freq='MS'))
        output["yhat"] = pred[0]
        output["yhat_lower"] = pred[1].T[0]
        output["yhat_upper"] = pred[1].T[1]
        return output
                    
    def get_params(self, deep=True):
        
        self.parameters = {'order':self.order,
                           'seasonal_order':self.seasonal_order}
    
        return self.parameters

    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

