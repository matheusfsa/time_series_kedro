"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import prepare_time_series, compute_seg_metrics, time_series_segmentation


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
        ),
        node(
            func=compute_seg_metrics,
            inputs={
                "data": "prepared_data",
                "serie_target": "params:serie_target",
                "serie_id": "params:series_level.columns",
                "serie_freq": "params:serie_freq"},
            outputs="seg_metrics",
            name="compute_seg_metrics"
        ),
        node(
            func=time_series_segmentation,
            inputs={
                "data": "prepared_data",
                "seg_metrics": "seg_metrics",
                "serie_id": "params:series_level.columns",
                "group_divisions": "params:group_divisions"},
            outputs="seg_data",
            name="time_series_segmentation"
        ),
    ])
