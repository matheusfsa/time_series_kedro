"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.17.6
"""

from cgi import test
from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from functools import reduce
from operator import add

from kedro.framework.session.session import get_current_session
from sklearn.model_selection import ParameterSampler
from .nodes import train_model, model_selection, test_models


def search_template(name: str) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_model,
                inputs={"series_data":"train_data", 
                        "serie_id":"params:series_level.columns",
                        "serie_target":"params:serie_target" ,
                        "date_col":"params:serie_period" ,
                        "model":"params:model" ,
                        "stride":"params:stride" ,
                        "fr_horizon":"params:fr_horizon" ,
                        "initial":"params:initial",
                        "n_jobs": "params:n_jobs",
                        "score": "params:score",
                        "train_start": "params:train_start",
                        "exog_info": "params:exog",
                        "use_exog": "params:use_exog"},
                outputs="best_estimators",
                name=name
            ),
        ]
    )

def create_pipeline(**kwargs):
    try:
        session = get_current_session()
        context = session.load_context()
        catalog = context.catalog

        models = catalog.load("params:models")
    except:
        models = ["exponential_smoothing", "arima", "svr"]

    search_pipelines = [
        pipeline(
            pipe=search_template(f"train_{model}"),
            parameters={"params:model": f"params:models.{model}"},
            outputs={"best_estimators": f"best_estimators_{model}"}
        )
        for model in models
    ]
    search_pipeline = reduce(add, search_pipelines)

    evaluation_pipeline = Pipeline([
        node(
            func=model_selection,
            inputs=["params:series_level.columns"] + [f"best_estimators_{model}" for model in models],
            outputs="best_estimators",
            name="model_selection"
        ),
        node(
            func=test_models,
            inputs={"train_data":"train_data", 
                    "test_data": "eval_data",
                    "best_estimators": "best_estimators",
                    "serie_id":"params:series_level.columns",
                    "serie_target":"params:serie_target" ,
                    "date_col":"params:serie_period",
                    "score": "params:score",
                    "train_start": "params:train_start",
                    "exog_info": "params:exog",
                    "use_exog": "params:use_exog"},
            outputs="metrics",
            name="evaluation"
        )
    ])
    return search_pipeline + evaluation_pipeline
