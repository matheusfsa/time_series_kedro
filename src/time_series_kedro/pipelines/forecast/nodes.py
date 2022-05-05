"""
This is a boilerplate pipeline 'forecast'
generated using Kedro 0.17.6
"""
from typing import Any, Union, List, Optional, Dict
import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from tqdm import tqdm
import time_series_kedro.extras.models as models


def forecast(
    series_data: pd.DataFrame,
    best_estimators: pd.DataFrame,
    serie_target: str,
    date_col: str,
    fr_horizon: int,
    serie_freq: int,
    train_start: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
    """
    This node execute forecast for each time serie.
    Args:
        series_data: DataFrame with train time series.
        best_estimators: DataFrame with best estimators for each serie.
        serie_id: Column or list of columns that identify series.
        serie_target: Target column name.
        date_col: Period column name.
        fr_horizon: Forecast horizon of Cross Validation
    Returns:
        DataFrame with forecast
    """
    if train_start is not None:
        if train_start["by"] ==  "offset":
            train_start_date = series_data.date.max() - DateOffset(**train_start["date"])
        elif train_start["by"] == "date":
            train_start_date = train_start["date"]
        else:
            raise ValueError(f"Filter by {train_start['by']} was not implemented")
        series_data = series_data[series_data.date >= train_start_date]
    series_data = pd.merge(series_data, best_estimators, on="serie_id")
    tqdm.pandas()
    forecast_results = series_data.groupby("serie_id").progress_apply(lambda data: _forecast(data, 
                                                                                  serie_target,
                                                                                  date_col,
                                                                                  fr_horizon,
                                                                                  serie_freq))
    forecast_results = forecast_results.reset_index(level=-1, drop=True).reset_index()         
    #forecast_results = pd.merge(test_data, forecast_results, on=serie_id + [date_col], validate="1:1")
    return forecast_results

def _forecast(
    data: pd.DataFrame,
    serie_target: str,
    date_col: str,
    fr_horizon: int,
    serie_freq: str,
    ) -> pd.DataFrame:
    """
    This node execute forecast for a time serie.
    Args:
        series_data: DataFrame with time serie.
        serie_target: Target column name.
        date_col: Period column name.
        fr_horizon: Forecast horizon of Cross Validation
    Returns:
        DataFrame with forecast
    """
    ts = data.set_index(date_col)[serie_target]
    estimator = eval(f"models.{data.best_estimator.iloc[0]}")
    estimator.fit(ts)

    y_pred = estimator.predict(fr_horizon)
    y_pred[y_pred < 0] = 0
    result = pd.DataFrame(data={"sales": y_pred, 
                                "date":pd.date_range(start=ts.index[-1], 
                                                     periods=y_pred.shape[0] + 1, 
                                                     freq=serie_freq)[1:]})
    
    return result                                                

def get_submission_file(
    forecast_results: pd.DataFrame,
    test_data: pd.DataFrame,
    serie_id: Union[str, List],
    serie_target: str,
    date_col: str,
):  
    """
    This node generate submission file.
    Args:
        forecast_results: DataFrame with forecast
    Returns:
        Submission file
    """
    test_data["serie_id"] = list(map(str, zip(*[test_data[c] for c in serie_id])))
    forecast_results = pd.merge(test_data, forecast_results, on=["serie_id", date_col], validate="1:1")
    return forecast_results[['id',serie_target]]
