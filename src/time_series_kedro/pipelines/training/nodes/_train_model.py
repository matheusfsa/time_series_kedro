
from typing import Any, Dict, List, Union

import numpy as np
from ._params_search import build_params_search
from ._search import TSModelSearchCV
from time_series_kedro.extras.utils import model_from_string
from pmdarima.model_selection import RollingForecastCV
from sklearn.base import clone, BaseEstimator
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

def train_model(
    series_data: pd.DataFrame,
    serie_id: Union[str, List],
    serie_target: str,
    date_col: str,
    model: Dict[str, Any],
    stride: int,
    fr_horizon: int,
    initial: Union[float, int],
    n_jobs: int = -1,
    score: str = "rmse",
    ) -> pd.DataFrame:
    """
    This node train and search the best estimators for each serie.
    Args:
        series_data: DataFrame with train time series.
        serie_id: Column or list of columns that identify series.
        serie_data: DataFrame with time series.
        serie_target: Target column name.
        date_col: Period column name.
        model: Dict with model definition
        stride: Stride of Cross Validation
        fr_horizon: Forecast horizon of Cross Validation
        initial: initial size of train dataset
        n_jobs: number of jobs in Cross Validation process
        score: used metric
    Returns:
        Best Estimators for each serie
    """
    
    model_groups_params = model["params"]
    
    estimator = model_from_string(model["model_class"], model["default_args"])
    
    best_estimators = series_data.groupby(serie_id).apply(lambda serie_data: _search(serie_data, 
                                                                                    estimator, 
                                                                                    model_groups_params, 
                                                                                    serie_target, 
                                                                                    date_col, 
                                                                                    stride, 
                                                                                    fr_horizon, 
                                                                                    initial,
                                                                                    n_jobs,
                                                                                    score))
    return best_estimators

def _search(
    serie_data: pd.DataFrame,
    estimator_base: BaseEstimator,
    model_groups_params: Dict[str, Any],
    serie_target: str,
    date_col: str,
    stride: int,
    fr_horizon: int,
    initial: Union[float, int],
    n_jobs: int,
    score: str):
    """
    This node train and search the best estimators for a serie.
    Args:
        series_data: DataFrame with train time series.
        serie_id: Column or list of columns that identify series.
        serie_data: DataFrame with time series.
        serie_target: Target column name.
        date_col: Period column name.
        model: Dict with model definition
        stride: Stride of Cross Validation
        fr_horizon: Forecast horizon of Cross Validation
        initial: initial size of train dataset
        n_jobs: number of jobs in Cross Validation process
        score: used metric
    Returns:
        Best Estimator
    """

    serie_group = serie_data.group.iloc[0]
    
    model_group = None
    if serie_group in model_groups_params:
        model_group  = serie_group
    if "all" in model_groups_params:
        model_group = "all"

    if model_group is not None:
        params_search = build_params_search(model_groups_params[model_group]["params_search"])
        estimator = clone(estimator_base)
        ts = serie_data.set_index(date_col)[serie_target]
        start_point = int(initial) if initial > 1 else int(initial*ts.shape[0])
        cv = RollingForecastCV(step=stride, h=fr_horizon, initial=start_point)
        search = TSModelSearchCV(clone(estimator), params_search, cv_split=cv, n_jobs=n_jobs, verbose=0, score=score)
        search.fit(ts)
        result = pd.Series({"estimator": search._best_estimator, "metric": search._best_score})
        
    else:
        result = pd.Series({"estimator": None, "metric": np.nan})
    return result   

def model_selection(serie_id, *best_estimators):

    estimators = pd.concat(best_estimators)
    estimators = estimators.reset_index().groupby(serie_id).apply(lambda data: data.set_index("estimator").metric.idxmin())
    estimators.name = "best_estimator"
    estimators = estimators.reset_index()
    
    return estimators