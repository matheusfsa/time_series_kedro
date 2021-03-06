"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from {{ cookiecutter.python_package }}.pipelines import preprocessing as pp
from {{ cookiecutter.python_package }}.pipelines import training
from {{ cookiecutter.python_package }}.pipelines import forecast as fr

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    pp_pipeline = pp.create_pipeline()
    training_pipeline = training.create_pipeline()
    forecast_pipeline = fr.create_pipeline()
    return {
        "__default__": pp_pipeline + training_pipeline + forecast_pipeline,
        "pp": pp_pipeline,
        "training": training_pipeline,
        "forecast": forecast_pipeline}
