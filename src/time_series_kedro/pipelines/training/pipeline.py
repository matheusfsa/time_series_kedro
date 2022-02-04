"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.17.6
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from functools import reduce
from operator import add

from kedro.framework.session.session import get_current_session
from .nodes import train_model


def search_template(name: str) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_model,
                inputs={"serie_data":"train_data", 
                        "serie_id":"params:serie_id" ,
                        "serie_target":"params:serie_target" ,
                        "date_col":"params:date_col" ,
                        "model":"params:model" ,
                        "stride":"params:stride" ,
                        "fr_horizon":"params:fr_horizon" ,
                        "initial":"params:initial" },
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
            pipe=search_template(f"train_data_{model}"),
            inputs={"model": f"params:models.{model}"},
            outputs={"best_estimators": f"best_estimators_{model}"}
        )
        for model in models
    ]
    search_pipeline = reduce(add, search_pipelines)
    return search_pipeline
