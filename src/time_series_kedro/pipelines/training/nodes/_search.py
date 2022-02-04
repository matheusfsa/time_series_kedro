import numpy as np
import pandas as pd
from typing import Union, Tuple, List
from pmdarima import model_selection
from sklearn import base
from itertools import product
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.base import BaseEstimator
import time_series_kedro.extras.models as used_models
from time_series_kedro.extras.utils import ld2dl
import time
import warnings
warnings.filterwarnings("ignore")
import logging

logger = logging.getLogger(__name__)
class TSModelSearchCV(BaseEstimator):
    
    def __init__(self, estimator, params_grid, cv_split, fit_error='warn', n_jobs=-1, verbose=1, score="rmse"):
        
        self.estimator = estimator
        self.params_grid = params_grid
        self.cv_split = cv_split
        self.fit_error = fit_error
        self.verbose = verbose
        self.score = score
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = max(min(n_jobs, cpu_count()), 1)
    
    
    def fit(self, y, X=None):
        
        if isinstance(self.params_grid, dict):
            params = [dict(zip(self.params_grid, t)) for t in product(*self.params_grid.values())] 
        else: 
            params = self.params_grid
                
        n_folds = len(list(self.cv_split._iter_train_test_indices(y, X)))
        n_params =  len(params)
        n_fits = n_folds * n_params
        
        if self.verbose > 0:
            print(f"Fitting {n_folds} folds for each of {n_params} candidates, totalling {n_fits} fits")
        
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch="2*n_jobs", verbose=self.verbose)
        
        out = parallel(delayed(_fit_and_pred)(fold = fold,
                                              estimator = base.clone(self.estimator),
                                              y = y,
                                              X = X,
                                              parameters = parameters,
                                              train = train,
                                              test = test,
                                              verbose = self.verbose,
                                              fit_error = self.fit_error,
                                              score = self.score
                                              )
                                              for (cand_idx, parameters), (fold, (train, test)) in product(
                                                  enumerate(params), enumerate(self.cv_split.split(y, X)))
        )
        out = pd.DataFrame(ld2dl(out))
        out["estimator"] = out.estimator.apply(str)
        estimators_results = out.groupby("estimator").mean().metric
        best_estimator_str = estimators_results.idxmin()
        self._best_score = estimators_results.min()
        self._best_estimator =  eval(f"used_models.{best_estimator_str}")
        return self

def _safe_indexing(
    X: Union[pd.Series, pd.DataFrame, np.array], 
    indices: Union[List[int], np.array]) -> Union[pd.Series, pd.DataFrame, np.array]:
    """Slice an array or dataframe."""
    
    # slicing dataframe
    if hasattr(X, 'iloc'):
        return X.iloc[indices]
    
    # slicing 2D array
    if hasattr(X, 'ndim') and X.ndim == 2:
        return X[indices, :]
    
    # list or 1d array
    return X[indices]

def _safe_split(
    y: Union[pd.Series, np.array], 
    X: Union[pd.Series, pd.DataFrame, np.array, None], 
    train: Union[List[int], np.array], 
    test: Union[List[int], np.array]) -> Tuple:
    """Performs the CV indexing given the indices"""
    
    y_train, y_test = _safe_indexing(y, train), _safe_indexing(y, test)
    
    if X is None:
        X_train = X_test = None
    else:
        X_train, X_test = _safe_indexing(X, train), _safe_indexing(X, test)
        
    return y_train, y_test, X_train, X_test

def _fit_and_pred(
    fold, 
    estimator, 
    y, 
    X, 
    parameters, 
    train, 
    test, 
    verbose, 
    fit_error,
    score="rmse"):
    """Fit estimator and compute scores for a given dataset split."""
    
    msg = 'fold=%i' % fold
    if verbose > 1:
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    start_time = time.time()
    y_train, y_test, X_train, X_test = _safe_split(y, X, train, test)
    
    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = base.clone(v, safe=False)
        
        estimator = estimator.set_params(**cloned_parameters)

    try:
        
        estimator.fit(y_train, X=X_train) 
        fit_time = time.time() - start_time

        start_time = time.time()
        preds = estimator.predict(len(test), X=X_test)
        preds[preds < 0] = 0
        if score == "rmse":
            metric = np.sqrt(mean_squared_error(test, preds))
        if score == "mape":
            metric = mean_absolute_percentage_error(test + 1, preds + 1)
        preds = np.round(preds, 0)
        
        if not np.isnan(preds).all():
            status = 'success'
        else:
            status = 'failed'
        message = None
        pred_time = time.time() - start_time

    except Exception as e:
        fit_time = time.time() - start_time
        start_time = time.time()
        preds = np.empty(len(test))
        preds[:] = np.nan
        status = 'failed'
        message = str(e)
        pred_time = time.time() - start_time
        metric = np.nan
        if fit_error == 'raise':
            raise
            
        elif fit_error == 'warn':
            warnings.warn("Estimator fit failed.")
            
        elif fit_error == 'ignore':
            pass

    if verbose > 2:
        total_time = pred_time + fit_time
        msg += ", [time=%.3f sec]" % (total_time)
        print(msg)  
    ret = {
            'fold': fold,
            'estimator': estimator,
            'training_status': status,
            'error_message': message,
            'metric': metric,
          }
    
    return ret

