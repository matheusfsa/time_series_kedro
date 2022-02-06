from joblib import Parallel, delayed, cpu_count
from typing import List, Callable
import pandas as pd
import numpy as np
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