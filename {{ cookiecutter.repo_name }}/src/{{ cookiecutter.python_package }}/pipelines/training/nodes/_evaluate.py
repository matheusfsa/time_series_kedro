
from typing import Any, Dict, List, Optional, Union
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_log_error
import {{ cookiecutter.python_package }}.extras.models as models
import pandas as pd
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
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
    train_start: Optional[Dict],
    use_exog: bool,
    exog_info: Optional[Dict],
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
    if train_start is not None:
        if train_start["by"] ==  "offset":
            train_start_date = train_data.date.max() - DateOffset(**train_start["date"])
        else:
            train_start_date = train_start["date"]
        train_data = train_data[train_data.date >= train_start_date]

    exog_list = []
    if exog_info is not None:
        for exog_name in exog_info:
            exog_list += exog_info[exog_name]["target_columns"]

    train_data = pd.merge(train_data, best_estimators, on="serie_id")
    tqdm.pandas()
    metrics_df = train_data.groupby("serie_id").progress_apply(lambda serie_data: _test_model(serie_data,
                                                                                test_data,
                                                                                "serie_id",
                                                                                serie_target,
                                                                                date_col,
                                                                                score,
                                                                                use_exog,
                                                                                exog_list))
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
    score: str,
    use_exog: bool,
    exog_columns: Optional[List[str]]
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
    idx = data["serie_id"].values[0]
    serie_points = test_data["serie_id"] == idx
    y_true = test_data[serie_points].set_index(date_col)[serie_target]
    ts = data.set_index(date_col)[serie_target]

    X_train = data[exog_columns + [date_col]].set_index(date_col) if len(exog_columns) and use_exog else None
    X_test = test_data[exog_columns + [date_col]].set_index(date_col) if len(exog_columns) and use_exog else None

    estimator = eval(f"models.{data.best_estimator.iloc[0]}")
    estimator.fit(np.log1p(ts), X=X_train)
    y_pred = np.expm1(estimator.predict(y_true.shape[0], X=X_test))
    y_pred[y_pred < 0] = 0
    if score == "rmse":
        metric = np.sqrt(mean_squared_error(y_true, y_pred))
    if score == "mape":
        metric = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
    if score == "rmsle":
        metric = np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))
    return pd.Series({"group": group,  "metric": metric})