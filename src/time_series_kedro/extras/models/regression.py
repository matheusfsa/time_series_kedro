import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import  make_pipeline
from numpy.lib.stride_tricks import sliding_window_view
import logging
import warnings
warnings.filterwarnings("ignore")

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
    ):
        lagged_data = sliding_window_view(target_series, self.lags + 1)
        X = lagged_data[:, :-1]
        y = lagged_data[:, -1]
        return X, y        

    def fit(self, y, X=None):
        logging.disable(logging.ERROR)
        
        ts = y.values
        
        self._train_series = ts.copy()
        X_train, y_train = self._create_lagged_data(ts)
        if X is not None:
            X_train = np.concatenate((X_train, X.values[self.lags:,]), axis=1)
            """
            Exog window
            
            exog_lagged_data = []
            for exog in X.columns:
                X_exog, y_exog= self._create_lagged_data(X[exog])
                exog_lagged_data = np.concatenate((X_exog, y_exog.reshape(-1, 1)), axis=1)
                X_train = np.concatenate([X_train, exog_lagged_data], axis=1)
            """
        self._lagged_data = (X_train, y_train)
        model_params = self.get_params().copy()
        del model_params["lags"]
        del model_params["poly_degree"]
        
        steps = []
        if self.poly_degree > 1:
            steps.append(PolynomialFeatures(self.poly_degree))
        steps.append(self._base_estimator(**model_params))
        self._model = make_pipeline(*steps)
        print(self._lagged_data[0].shape)
        X, y = self._lagged_data
        self._model.fit(X, y)
        logging.disable(logging.NOTSET)
        return self
        
    
    def predict(self, n_periods, X=None):
        logging.disable(logging.ERROR)
        X_hist = np.zeros(self.lags)
        X_hist[:self.lags] = self._lagged_data[0][-1, :self.lags]
        X_hist[-1] = self._lagged_data[1][-1]
        preds = []

        for i in range(n_periods):
            if X is not None:
                exog_values = X.iloc[i, :] 
                X_pred = np.concatenate((X_hist, X.iloc[i, :]))
                pred = self._model.predict(X_pred.reshape(1, -1))
            else:
                pred = self._model.predict(X_hist.reshape(1, -1))
            preds.append(pred[0])
            X_hist = np.roll(X_hist, -1)
            X_hist[self.lags-1] = pred
            
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

class RidgeForecaster(RegressionModel):
    def __init__(self, lags=1, **kwargs):
        
        super().__init__(base_estimator=Ridge, lags=lags, **kwargs)

class LassoForecaster(RegressionModel):
    def __init__(self, lags=1, **kwargs):
        super().__init__(base_estimator=Lasso, lags=lags, **kwargs)