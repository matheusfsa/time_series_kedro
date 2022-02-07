"""
This is a boilerplate pipeline 'forecast'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from .nodes import forecast, get_submission_file

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=forecast,
            inputs={"series_data": "seg_data",
                    "test_data": "test_data",
                    "best_estimators": "best_estimators",
                    "serie_id":"params:series_level.columns",
                    "serie_target":"params:serie_target",
                    "date_col":"params:serie_period",
                    "fr_horizon":"params:fr_horizon"},
            outputs="forecast_results",
            name="forecast"
        ),
        node(
            func=get_submission_file,
            inputs="forecast_results",
            outputs="submission_file",
            name="get_submission_file"
        )
    ])
