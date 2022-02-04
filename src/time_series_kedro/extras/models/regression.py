import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import  make_pipeline

import logging


class RegressionModel(RegressorMixin, BaseEstimator):
    '''

    '''
    def __init__(self, base_estimator, lags=1, poly_degree=1, **kwargs):
        self._base_estimator = base_estimator
        self.lags = lags
        self.poly_degree = poly_degree
        self.params = ['lags','poly_degree'] + list(kwargs.keys())
        for parameter, value in kwargs.items():
            setattr(self, parameter, value)

    def _create_lagged_data(
        self, 
        target_series, 
        past_covariates, 
        future_covariates, 
        max_samples_per_ts
    ):
        n_in = self.lags
        n_out = 1
        data = target_series.copy()
        
        n_vars = 1 if len(data.shape) == 1 else data.shape[-1]
        
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        agg.dropna(inplace=True)
        X = agg.iloc[:, :-1].values
        y = agg.iloc[:, -1].values
        return X, y        

    def fit(self, y, X=None):
        logging.disable(logging.ERROR)
        
        ts = y.values
        
        self._train_series = ts.copy()
        self._lagged_data = self._create_lagged_data(ts, 
                                                    past_covariates=None, 
                                                    future_covariates=None, 
                                                    max_samples_per_ts=None)
        
        model_params = self.get_params().copy()
        del model_params["lags"]
        del model_params["poly_degree"]
        
        steps = []
        if self.poly_degree > 1:
            steps.append(PolynomialFeatures(self.poly_degree))
        steps.append(self._base_estimator(**model_params))
        self._model = make_pipeline(*steps)

        X, y = self._lagged_data
        self._model.fit(X, y)
        logging.disable(logging.NOTSET)
        return self
        
    
    def predict(self, n_periods, X=None):
        logging.disable(logging.ERROR)
        X = self._lagged_data[0][-1, :]
        preds = []
        for i in range(n_periods):
            pred = self._model.predict(X.reshape(1, -1))
            preds.append(pred[0])
            X = np.roll(X, -1)
            X[-1] = pred
        logging.disable(logging.NOTSET)
        return np.array(preds)
    
    
    def get_params(self, deep=True):
        
        self.parameters = {}
        for p in self.params:
            self.parameters[p] = getattr(self, p)
    
        return self.parameters
    
    
    
    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            
        return self

class RandomForestForecaster(RegressionModel):
    def __init__(self, lags=1, **kwargs):
        super().__init__(base_estimator=RandomForestRegressor, lags=lags, **kwargs)


class SVRForecaster(RegressionModel):
    def __init__(self, lags=1, **kwargs):
        super().__init__(base_estimator=LinearSVR, lags=lags, **kwargs)  
        
class AdaForecaster(RegressionModel):
    def __init__(self, lags=1, **kwargs):
        
        super().__init__(base_estimator=AdaBoostRegressor, lags=lags, **kwargs)