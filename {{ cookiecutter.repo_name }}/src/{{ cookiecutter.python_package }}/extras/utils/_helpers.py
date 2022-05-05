from typing import Any, Dict, List, Optional
from importlib import import_module
from joblib import Parallel, delayed, cpu_count
from typing import List, Callable
import pandas as pd
import numpy as np

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

def parallel_groupby(
    data: pd.DataFrame,
    group_func: Callable,
    group_cols: List[str],
    n_jobs: int = -1,
    **kwargs
) -> pd.DataFrame:
    
    """Parallel groupby function"""
    
    data['agrup_col'] = list(map(str, zip(*[data[c] for c in group_cols])))
    agrups = data['agrup_col'].unique()
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = min(n_jobs, agrups.shape[0])
    parallel = Parallel(n_jobs=n_jobs, pre_dispatch="2*n_jobs", verbose=0)
    
    out = parallel(
                delayed(group_func)(
                    data=data[data['agrup_col'].isin(group)].copy(),
                    group_cols=group_cols,
                    **kwargs
                )
                for group in np.array_split(data['agrup_col'].unique(), n_jobs)
            )
    out = pd.concat(out)
    
    if isinstance(out, pd.DataFrame):
        if 'agrup_col' in out.columns:
            out = out.drop(column='agrup_col')
            
    return out