"""
Prepare time series functions.
"""
import logging
from typing import List, Union
import pandas as pd

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

def prepare_time_series(
    data: pd.DataFrame,
    date_col: str,
    serie_target: str,
    serie_id: Union[str, List[str]],
    ) -> pd.DataFrame:
    """
    This node prepare time series, ensuring that all series have all periods
    (adding duplicate periods and adding periods without observations) and
    filling null values.

    Args:
        data: Dataframe with time series.
        date_col: Period column name.
        serie_target: Target column name.
        serie_id: Column or list of columns that identify series.

    Returns:
        Data with prepared time series
    """
    data["serie_id"] = list(map(str, zip(*[data[c] for c in serie_id])))
    series_data = data.pivot_table(columns="serie_id", values=serie_target, index=date_col)
    new_idx = pd.Index(pd.date_range(series_data.index.min(), series_data.index.max()), name="date")
    series_data = series_data.reindex(new_idx)
    series_data = _rolling_mean(series_data)
    series_data = pd.melt(series_data, value_name=serie_target, ignore_index=False).reset_index()
    return series_data

def _rolling_mean(data: pd.DataFrame, window_size: int = 2) -> pd.DataFrame:
    """
    Fills Na values with the mean of the nearest values.

    Args:
        data: Original Series.
        n: Window size.
    Return:
        DataFrame with missing values filled.
    """
    out = np.copy(data)
    filler = np.full((window_size//2, out.shape[1]), np.nan)
    rolling_mean = np.vstack((filler, out, filler))
    rolling_mean = sliding_window_view(rolling_mean, window_size+1, 0)
    rolling_mean = np.nanmean(rolling_mean, axis=2)
    out[np.isnan(out)] = rolling_mean[np.isnan(out)]
    out[np.isnan(out)] = 0
    out_data = pd.DataFrame(data=out, columns=data.columns, index=data.index)
    return out_data

def add_exog(data, exog_info, *exogs):
    """
    FThis node add exogenous variables to series Dataframe.
    Args:
        data: Original Series.
        exog_info: Exogenous descriptions.
        exogs: DataFrames with exogs data.
    Return:
        DataFrame with endogenous and exogenous data.
    """
    test_exog = pd.DataFrame()
    for i, (exog_name, exog_data) in enumerate(zip(exog_info.keys(), exogs)):
        merge_cols = exog_info[exog_name]["merge_columns"]
        target_cols = exog_info[exog_name]["target_columns"]
        train_exog_data = exog_data[exog_data.date <= data.date.max()]
        test_exog_data = exog_data[exog_data.date > data.date.max()]
        data = pd.merge(data, train_exog_data[merge_cols + target_cols], on=merge_cols)
        if i > 0:
            test_exog = pd.merge(test_exog, test_exog_data[merge_cols + target_cols], on=merge_cols)
        else:
            test_exog = test_exog_data[merge_cols + target_cols]
        data[target_cols] = data[target_cols].fillna(0)
        test_exog[target_cols] = test_exog[target_cols].fillna(0)
    return data, test_exog
