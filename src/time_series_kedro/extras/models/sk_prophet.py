import os
import sys

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from prophet import Prophet as FacebookProphet
from typing import Dict, Tuple
import os

class Prophet(RegressorMixin, BaseEstimator):
    '''

    '''
    def __init__(self, period=365.25, seasonality="additive", changepoint_prior_scale=0.3, seasonality_prior_scale=0.2):
        self.period = period
        self.seasonality = seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.train_size = 0
            
    def fit(self, y, X=None):
        
        self._model = FacebookProphet(
                        yearly_seasonality=False,
                        weekly_seasonality=False,
                        daily_seasonality=False,
                       )
        self._model.add_seasonality(name='custom', 
                              period=self.period, 
                              fourier_order=10, 
                              prior_scale=self.seasonality_prior_scale,
                             mode=self.seasonality)  
        df_train = pd.DataFrame({"ds":y.index, "y": y.values})
        if X is not None:
            X = X.rename_axis("ds")
            self._regressors_names = X.columns
            X = X.reset_index()
            df_train = pd.merge(df_train, X, on="ds", how="left")
            df_train[self._regressors_names] = df_train[self._regressors_names].fillna(0)
            for regressor in self._regressors_names:
                self._model.add_regressor(regressor)

        self.train_size = df_train.shape[0]
        with suppress_stdout_stderr():
            self._model.fit(df_train)
        return self
        
        
    def predict(self, n_periods, X=None):
        future = self._model.make_future_dataframe(periods=n_periods, freq='D')
        if X is not None:
            X = X.rename_axis("ds").reset_index()
            future = pd.merge(future, X, on="ds", how="left")
            future[self._regressors_names] = future[self._regressors_names].fillna(0)
        pred = self._model.predict(future).set_index("ds")['yhat'].iloc[-n_periods:]
        return pred
                    
    def get_params(self, deep=True):
        self.parameters = {'period':self.period,
                           'seasonality':self.seasonality,
                           'changepoint_prior_scale':self.changepoint_prior_scale,
                           'seasonality_prior_scale':self.seasonality_prior_scale}
    
        return self.parameters

    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self


class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])