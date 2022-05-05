"""
Segmentation functions.
"""
from typing import Any, Dict, Optional
from itertools import product
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def compute_seg_metrics(
    data: pd.DataFrame,
    serie_target: str,
    date_col: str,
    n_last_points: int
) -> pd.Series:
    """
    This node calculates metrics to assess the quality of the series.

    Args:
        data: Dataframe with time series.
        serie_id: Column or list of columns that identify series.
        serie_target: Target column name.
        serie_freq: Serie frequency.
    Returns:
        Dataframe with metrics computed to each serie.
    """
    series_data = data.pivot_table(columns="serie_id", values=serie_target, index=date_col)
    series = series_data.values
    metrics = pd.DataFrame(index=series_data.columns)
    metrics["mean_serie"] = series.mean(axis=0)
    metrics["std_serie"] = series.std(axis=0)
    metrics["cv"] = metrics.std_serie/metrics.mean_serie

    metrics["acc"] = series[-n_last_points:,:].sum(axis=0)
    return metrics.reset_index()


def time_series_segmentation(
    data: pd.DataFrame,
    seg_metrics: pd.DataFrame, 
    group_divisions: Dict[str, Any],
    sampling: Optional[int] = None,
    random_state: int = 42):
    """
    This node segments the series based on a set of conditions that
    have been defined for the metrics.

    Args:
        data: Dataframe with time series.
        seg_metrics: Dataframe with metrics computed to each serie.
        serie_id: Column or list of columns that identify series.
        group_division: Conditions that have been defined for the metrics
    Returns:
        Dataframe with segmentation groups in column ``group``.
    """

    metrics = list(group_divisions)
    seg_metrics["group"] = 0

    for i, group in enumerate(product(["gt", "le"], repeat=len(metrics))):
        series_filter = True 
        for comp, metric in zip(group, metrics):
            method = group_divisions[metric]["method"]
            args = group_divisions[metric]["args"]
            value = getattr(seg_metrics[metric], method)(*args)
            comp_filter = getattr(seg_metrics[metric], comp)(value)
            series_filter = series_filter & comp_filter
        seg_metrics.loc[series_filter, "group"] = i + 1
    seg_metrics = seg_metrics[["serie_id", "group"]]
    data = pd.merge(data, seg_metrics, on="serie_id")
    if sampling:
        np.random.seed(random_state)
        sample = pd.DataFrame()
        for group in data.group.unique():
            data_sample = data[data.group == group]
            series = np.random.choice(data_sample["serie_id"].unique(), min(sampling, data_sample["serie_id"].nunique()), replace=False)
            data_sample = data_sample[data_sample["serie_id"].isin(series)]
            sample = pd.concat((sample, data_sample), ignore_index=True)
        data = sample
        logger.info(f"# Series after sampling: {data['serie_id'].nunique()}")
    return data

