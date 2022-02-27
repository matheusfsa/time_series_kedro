
from tabnanny import verbose
from typing import Any, Dict, List, Optional, Union
from xmlrpc.client import boolean
from jmespath import search

import numpy as np
from ._params_search import build_params_search
from ._search import TSModelSearchCV
from time_series_kedro.extras.utils import model_from_string
from pmdarima.model_selection import RollingForecastCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.base import clone, BaseEstimator
import pandas as pd
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
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
    train_start: Optional[Dict],
    use_exog: bool, 
    exog_info: Optional[Dict],
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
    if train_start is not None:
        if train_start["by"] ==  "offset":
            train_start_date = series_data.date.max() - DateOffset(**train_start["date"])
        elif train_start["by"] == "date":
            train_start_date = train_start["date"]
        else:
            raise ValueError(f"Filter by {train_start['by']} was not implemented")
        series_data = series_data[series_data.date >= train_start_date]
    model_groups_params = model["params"]
    exog_list = []
    if exog_info is not None:
        for exog_name in exog_info:
            exog_list += exog_info[exog_name]["target_columns"]
    
    estimator = model_from_string(model["model_class"], model["default_args"])
    best_estimators = pd.DataFrame()
    for serie_idx, serie_data in tqdm(series_data.groupby(serie_id), total=series_data[serie_id].drop_duplicates().shape[0]):
        serie_result = _search(serie_data, estimator, model_groups_params, 
                               serie_target, date_col, stride, 
                               fr_horizon, initial,n_jobs,score, 
                               use_exog, exog_list)
        for id_col, id in zip(serie_id, serie_idx):
            serie_result[id_col] = id
        best_estimators = pd.concat((best_estimators, serie_result), ignore_index=True)
    return best_estimators

def get_scoring(score):
    if score == "rmse":
        return make_scorer(mean_squared_error, squared=True)
    if score == "mape":
        cost = lambda y_true, y_pred: mean_absolute_percentage_error(y_true + 1, y_pred + 1)
        return make_scorer(cost, squared=True)
    if score == "rmsle":
        cost = lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))
        return make_scorer(cost, squared=True)

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
    score: str,
    use_exog: bool,
    exog_columns: Optional[List[str]]):
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
    
    if serie_group in model_groups_params:
        model_group  = serie_group
    elif "all" in model_groups_params:
        model_group = "all"
    else:
        model_group = None
    if model_group is not None:
        params_search = build_params_search(model_groups_params[model_group]["params_search"])
        estimator = clone(estimator_base)
        ts = serie_data.set_index(date_col)[serie_target]
        start_point = int(initial) if initial > 1 else int(initial*ts.shape[0])
        cv = RollingForecastCV(step=stride, h=fr_horizon, initial=start_point)
        search = TSModelSearchCV(clone(estimator), params_search, cv_split=cv, n_jobs=n_jobs, verbose=0, score=score)
        if use_exog:
            X = serie_data[exog_columns + [date_col]].set_index(date_col) if len(exog_columns) else None
        else:
            X = None
        search.fit(np.log1p(ts), X=X)
        result = pd.DataFrame({"estimator": [search._best_estimator], "metric": [search._best_score]})
        
    else:
        result = pd.DataFrame({"estimator": [None], "metric": [np.nan]})
    return result   

def model_selection(serie_id, *best_estimators):

    estimators = pd.concat(best_estimators)
    estimators = estimators.reset_index().groupby(serie_id).apply(lambda data: data.set_index("estimator").metric.idxmin())
    estimators.name = "best_estimator"
    estimators = estimators.reset_index()
    
    return estimators