"""
This is a boilerplate pipeline 'forecast'
generated using Kedro 0.17.6
"""
from typing import Union, List
import numpy as np
import pandas as pd
import time_series_kedro.extras.models as models
from tqdm import tqdm

def forecast(
    series_data: pd.DataFrame,
    test_data: pd.DataFrame,
    best_estimators: pd.DataFrame,
    serie_id: Union[str, List],
    serie_target: str,
    date_col: str,
    fr_horizon: int,
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
    series_data = pd.merge(series_data, best_estimators, on=serie_id)
    tqdm.pandas()
    forecast_results = series_data.groupby(serie_id).progress_apply(lambda data: _forecast(data, 
                                                                                  serie_target, 
                                                                                  date_col, 
                                                                                  fr_horizon))
    forecast_results = forecast_results.reset_index(level=-1, drop=True).reset_index()         
    forecast_results = pd.merge(test_data, forecast_results, on=["store_nbr", "family", "date"], validate="1:1")

    return forecast_results

def _forecast(
    data: pd.DataFrame,
    serie_target: str,
    date_col: str,
    fr_horizon: int,
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
                                                     freq="D")[1:]})
    
    return result                                                

def get_submission_file(
    forecast_results: pd.DataFrame
):  
    """
    This node generate submission file.
    Args:
        forecast_results: DataFrame with forecast
    Returns:
        Submission file
    """
    return forecast_results[['id','sales']]