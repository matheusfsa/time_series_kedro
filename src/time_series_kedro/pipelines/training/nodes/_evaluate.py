
from typing import Any, Dict, List, Union
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error
import time_series_kedro.extras.models as models
import pandas as pd
from tqdm import tqdm
import warnings
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

def test_models(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    best_estimators: pd.DataFrame,
    serie_id: Union[str, List],
    serie_target: str,
    date_col: str,
    score: str,
    ) -> Dict[str, Any]:
    """
    This node evaluate the best estimators in test.
    Args:
        train_data: DataFrame with train time series.
        test_data: DataFrame with test time series.
        best_estimators: DataFrame with best estimators for each serie.
        serie_id: Column or list of columns that identify series.
        serie_data: DataFrame with time series.
        serie_target: Target column name.
        date_col: Period column name.
        score: used metric
    Returns:
        Metrics in test
    """
    train_data = pd.merge(train_data, best_estimators, on=serie_id)
    tqdm.pandas()
    metrics_df = train_data.groupby(serie_id).progress_apply(lambda serie_data: _test_model(serie_data, 
                                                                                test_data,
                                                                                serie_id,
                                                                                serie_target, 
                                                                                date_col,
                                                                                score))
    metrics = {
        "metric": {"value": metrics_df["metric"].mean(), "step":1}
    }
    for group in metrics_df.group.unique():
        metrics[f"metric_{group}"] = {"value": metrics_df[metrics_df.group== group]["metric"].mean(), "step":1}
    logger.info(f"metrics:{metrics}")
    return metrics

def _test_model(
    data: pd.DataFrame,
    test_data: pd.DataFrame,
    serie_id: Union[str, List],
    serie_target: str,
    date_col: str,
    score: str
    ):
    """
    This node evaluate the best estimator in test.
    Args:
        data: DataFrame with train time series.
        test_data: DataFrame with test time series.
        best_estimators: DataFrame with best estimators for each serie.
        serie_id: Column or list of columns that identify series.
        serie_data: DataFrame with time series.
        serie_target: Target column name.
        date_col: Period column name.
        score: used metric
    Returns:
        Metrics in test
    """
    group = data.group.values[0]
     #pd.Series([True for _ in range(test_data.shape[0])], index=test_data.index)
    if isinstance(serie_id, list):
        serie_points = True
        for id_col in serie_id:
            idx = data[id_col].values[0]
            serie_points = (serie_points) & (test_data[id_col] == idx)
    else:
        idx = data[serie_id].values[0]
        serie_points = test_data[serie_id] == idx
    y_true = test_data[serie_points].set_index(date_col)[serie_target]
    ts = data.set_index(date_col)[serie_target]

    estimator = eval(f"models.{data.best_estimator.iloc[0]}")
    estimator.fit(ts)
    y_pred = estimator.predict(y_true.shape[0])
    y_pred[y_pred < 0] = 0
    if score == "rmse":
        metric = np.sqrt(mean_squared_error(y_true, y_pred))
    if score == "mape":
        metric = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
    if score == "rmsle":
        metric = np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))
    return pd.Series({"group": group,  "metric": metric})