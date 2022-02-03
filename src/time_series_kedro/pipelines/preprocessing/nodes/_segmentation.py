from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd
from itertools import product
from statsmodels.tsa.stattools import adfuller

def compute_seg_metrics(
    data: pd.DataFrame,
    serie_id: Union[List[str], str],
    serie_target: str,
    serie_freq: str
):
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

    seg_data = data.groupby(serie_id).apply(lambda serie_data: _seg_metrics(serie_data, serie_target, serie_freq)) 
    return seg_data.reset_index()

def time_series_segmentation(
    seg_metrics: pd.DataFrame, 
    serie_id: Union[List[str], str],
    group_divisions: Dict[str, Any]):
    """
    This node segments the series based on a set of conditions that 
    have been defined for the metrics.

    Args:
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

    return seg_metrics[serie_id + ["group"]]

def _seg_metrics(
    serie_data: pd.DataFrame,
    serie_target: str,
    serie_freq: str
):
    """
    This function compute metrics (Sample Entropy, Coefficient of variation, 
    Serie size, Amount accumulated in the last cycle).

    Args:
        serie_data: Dataframe with time serie.
        serie_target: Target column name.
        serie_freq: Serie frequency.
    Returns:
        Serie metrics.
    """

    ts = serie_data[serie_target].values
    nonzeros = np.nonzero(ts)
    first_point = nonzeros[0][0]
    last_point = nonzeros[0][-1]
    len_ts = (last_point - first_point) + 1
    ts = ts[first_point:]
    sample_entropy = _sample_entropy(ts, m=2, r=0.2*np.std(ts)) 
    mean = ts.mean()
    if mean:
        cv = ts.std()/mean
    else:
        cv = np.nan

    if serie_freq == "D":
        last = 30
    elif serie_freq == "M" or serie_freq == "MS":
        last = 12
    elif serie_freq == "Y":
        last = 1
    elif serie_freq == "h":
        last = 24
    acc_12m = ts[-last:].sum()

    adf = adfuller(ts)[0]

    
    return pd.Series({
            "sample_entropy": sample_entropy, 
            "cv": cv, 
            "len_ts": len_ts, 
            "acc_12m": acc_12m,
            "adf":adf})



def _sample_entropy(
    L: np.array,
    m: int,
    r: int
) -> int:
    """ 
    Calculates Sample Entropy for a given time series. Sample entropy (SampEn)
    is a modification of approximate entropy (ApEn), used for assessing the 
    complexity of time-series signals. For more details please refer to 
    https://www.mdpi.com/1099-4300/21/6/541.
    
    Args:
        L: array_like, time-series signal.
        m: int, embedding dimension.
        r: int, tolerance
    Returns:
        Sample entropy.
    """
    # Initialize parameters
    N = len(L)
    B = 0.0
    A = 0.0
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)