"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline
from time_series_kedro.pipelines import preprocessing as pp

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    pp_pipeline = pp.create_pipeline()
    return {
        "__default__": pp_pipeline,
        "pp": pp_pipeline}
