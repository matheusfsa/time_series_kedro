"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import (prepare_time_series, 
                    compute_seg_metrics, 
                    time_series_segmentation, 
                    train_test_split)


def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=prepare_time_series,
            inputs={
                "data": "master_table",
                "date_col": "params:serie_period",
                "serie_target": "params:serie_target",
                "serie_id": "params:series_level.columns",
                "sampling": "params:sampling",
                "random_state": "params:random_state"},
            outputs="prepared_data",
            name="prepare_time_series"
        ),
        node(
            func=compute_seg_metrics,
            inputs={
                "data": "prepared_data",
                "serie_target": "params:serie_target",
                "serie_id": "params:series_level.columns",
                "serie_freq": "params:serie_freq",
                "n_jobs": "params:n_jobs"},
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
        node(
            func=train_test_split,
            inputs={
                "data": "seg_data",
                "serie_id": "params:series_level.columns",
                "test_size": "params:test_size"},
            outputs=["train_data", "eval_data"],
            name="train_test_split"
        ),
    ])
