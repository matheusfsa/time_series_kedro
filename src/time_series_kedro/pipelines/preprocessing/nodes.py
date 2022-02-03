"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.17.6
"""

from typing import List, Union
import pandas as pd

from time_series_kedro.extras.utils import rolling_fill

def _build_series(serie_data, serie_target, date_col):
    serie = serie_data.set_index(date_col)[[serie_target]]
    full_serie = serie.reindex(pd.date_range(serie.index.min(), serie.index.max()))
    full_serie[serie_target] = rolling_fill(full_serie[serie_target], n=2)
    return full_serie

def prepare_time_series(data: pd.DataFrame, 
                        date_col: str, 
                        serie_target: str, 
                        serie_id: Union[str, List[str]]) -> pd.DataFrame:
    
    data = data.groupby([date_col] + serie_id).sum()[serie_target].reset_index()

    data = data.groupby(serie_id).apply(lambda serie_data: _build_series(serie_data, serie_target, date_col))
    return data.reset_index()


def _build_series(serie_data, serie_target, date_col):
    serie = serie_data.set_index(date_col)[[serie_target]]
    full_serie = serie.reindex(pd.date_range(serie.index.min(), serie.index.max()))
    full_serie[serie_target] = rolling_fill(full_serie[serie_target], n=2)
    return full_serie