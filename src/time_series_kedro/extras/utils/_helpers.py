from typing import Any, Dict, List, Optional
from importlib import import_module

def ld2dl(ld: List[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    """
    This function converts a list of dicts to a dict of lists
    Args:
        ld: List of dicts.
    Returns:
        Dict of lists
    """
    return {k: [dic[k] for dic in ld] for k in ld[0]}



def model_from_string(
    model_name: str, 
    default_args: Optional[Dict[str, Any]] =None
    ) -> Any:
    """
    This function load model from string
    Args:
        model_name: Path to model.
        default_args: Default model args.
    Return:
        Model instance. 
    """
    model_class = getattr(
        import_module((".").join(model_name.split(".")[:-1])),
        model_name.rsplit(".")[-1],
    )
    if default_args is None:
        return model_class()
    else:
        return model_class(**default_args)