import imp
from itertools import groupby
from typing import Any, Dict, List, Union
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import time_series_kedro.extras.models as models
import pandas as pd
import warnings

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

    train_data = pd.merge(train_data, best_estimators, on=serie_id)
    
    metrics_df = train_data.groupby(serie_id).apply(lambda serie_data: _test_model(serie_data, 
                                                                                test_data,
                                                                                serie_id,
                                                                                serie_target, 
                                                                                date_col,
                                                                                score))
    metrics = {
        "rmse": {"value": metrics_df["metric"].mean(), "step":1}
    }
    for group in metrics_df.group.unique():
        metrics[f"rmse_{group}"] = {"value": metrics_df[metrics_df.group== group]["metric"].mean(), "step":1}

    return metrics

def _test_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    serie_id: Union[str, List],
    serie_target: str,
    date_col: str,
    score: str
    ):

    group = train_data.group.values[0]
    idx = train_data[serie_id].values[0]
    y_true = test_data[test_data[serie_id].values.reshape(-1) == idx].set_index(date_col)[serie_target]
    ts =train_data.set_index(date_col)[serie_target]

    estimator = eval(f"models.{train_data.best_estimator.iloc[0]}")
    estimator.fit(ts)
    y_pred = estimator.predict(y_true.shape[0])
    if score == "rmse":
        metric = np.sqrt(mean_squared_error(y_true, y_pred))
    if score == "mape":
        metric = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
    return pd.Series({"group": group,  "metric": metric})