from cgi import test
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.stattools import adfuller

def train_test_split(
    data: pd.DataFrame,
    serie_id: Union[List[str], str],
    test_size: int
):
    """
    This node calculates metrics to assess the quality of the series.

    Args:
        data: Dataframe with time series.
        serie_id: Column or list of columns that identify series.
        serie_target: Target column name.
        test_size: test size
    Returns:
        Train and test DataFrames
    """
    train_data = data.groupby("serie_id").apply(lambda data: data.iloc[:-test_size,:])
    test_data = data.groupby("serie_id").apply(lambda data: data.iloc[-test_size:,:])
    return train_data, test_data


