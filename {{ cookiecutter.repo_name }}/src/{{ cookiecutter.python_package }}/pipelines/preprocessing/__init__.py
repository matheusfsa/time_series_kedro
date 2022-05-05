""" ``{{ cookiecutter.python_package }}.pipelines.preprocessing`` is the pipeline responsible for pre-processing the data.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
