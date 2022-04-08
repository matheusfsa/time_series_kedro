"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from kedro.framework.session.session import get_current_session
from .nodes import (prepare_time_series, 
                    compute_seg_metrics, 
                    time_series_segmentation, 
                    train_test_split,
                    add_exog)


def create_pipeline():
    """
    This function create a pipeline that preprocessing data.
    """
    
    try:
        session = get_current_session()
        context = session.load_context()
        catalog = context.catalog

        exog = catalog.load("params:exog")
    except RuntimeError:
        exog = ["oil",]

    return Pipeline([
        node(
            func=prepare_time_series,
            inputs={
                "data": "master_table",
                "date_col": "params:serie_period",
                "serie_target": "params:serie_target",
                "serie_id": "params:series_level.columns"},
            outputs="prepared_data_wo_exog",
            name="prepare_time_series"
        ),
        node(
            func=add_exog,
            inputs=["prepared_data_wo_exog","params:exog"] + [data_ref for data_ref in exog],
            outputs=["prepared_data", "exog_test_data"],
            name="add_exog"
        ),
        node(
            func=compute_seg_metrics,
            inputs={
                "data": "prepared_data",
                "serie_target": "params:serie_target",
                "date_col": "params:serie_period",
                "n_last_points": "params:n_last_points"},
            outputs="seg_metrics",
            name="compute_seg_metrics"
        ),
        node(
            func=time_series_segmentation,
            inputs={
                "data": "prepared_data",
                "seg_metrics": "seg_metrics",
                "group_divisions": "params:group_divisions",
                "sampling": "params:sampling",
                "random_state": "params:random_state"},
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
