from typing import List, Optional, Union
import pandas as pd

#from time_series_kedro.extras.utils import rolling_fill
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)

def prepare_time_series(
    data: pd.DataFrame, 
    date_col: str, 
    serie_target: str, 
    serie_id: Union[str, List[str]],
    sampling: Optional[int] = None,
    random_state: int = 42
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
    if sampling:
        np.random.seed(random_state)
        series = np.random.choice(data[serie_id[0]].unique(), min(sampling, data[serie_id[0]].nunique()), replace=False)
        data = data[data[serie_id[0]].isin(series)]
        logger.info(f"# Series after sampling: {data[serie_id].drop_duplicates().shape[0]}")
    data = data.groupby([date_col] + serie_id).sum()[serie_target].reset_index()
    tqdm.pandas()
    data = data.groupby(serie_id).progress_apply(lambda serie_data: _build_series(serie_data, serie_target, date_col))
    return data.reset_index()


def _build_series(
    serie_data: pd.DataFrame, 
    serie_target: str, 
    date_col: str) -> pd.DataFrame:
    """
    This function prepare a time series, ensuring that all series have all periods 
    (adding duplicate periods and periods without observations) and 
    filling null values.

    Args:
        serie_data: Dataframe with time series.
        serie_target: Target column name.
        date_col: Period column name.

    Returns:
        Data with prepared time serie
    """
    
    serie = serie_data.set_index(date_col)[[serie_target]]
    full_serie = serie.reindex(pd.Index(pd.date_range(serie.index.min(), serie.index.max()), name="date"))
    full_serie[serie_target] = _rolling_fill(full_serie[serie_target], n=2)
    return full_serie


# Fill function
def _rolling_fill(
    data: pd.Series,
    n: int
) -> pd.Series:

    """
    Fills Na values with the mean of the nearest values.

    Args:
        data: Original Series.
        n: Window size.
    Return:
        Series with missing values filled. 
    """

    data[data < 0] = 0

    out = np.copy(data)
    w_size = n//2

    # Create sliding window view -> [[x[i]-1, x[i], x[i+1]] for i in range(x.shape)]
    rolling_mean = np.hstack((np.full(w_size, np.nan), out, np.full(w_size, np.nan)))
    axis = 0 if len(rolling_mean.shape) == 1 else 1
    rolling_mean = np.nanmean(sliding_window_view(rolling_mean, (n+1,), axis=axis), axis=1)
    # Get Mean e filling nan values
    out[np.isnan(out)] = rolling_mean[np.isnan(out)]
    out[np.isnan(out)] = 0

    return out