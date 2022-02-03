"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import prepare_time_series


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=prepare_time_series,
            inputs={
                "data": "train_data",
                "date_col": "params:serie_period",
                "serie_target": "params:serie_target",
                "serie_id": "params:series_level.columns"},
            outputs="prepared_data",
            name="prepare_time_series"
            )
    ])
