from sklearn.model_selection import ParameterGrid, ParameterSampler
import scipy.stats as stats
import numpy as np
from typing import Dict

def build_params_search(params_dict: Dict):
    
    params = params_dict["params"]
    for p in params:
        if isinstance(params[p], str):
            params[p] = eval(params[p])
        elif isinstance(params[p], list):
            params[p] = list(map(lambda x: tuple(x) if isinstance(x, list) else x, params[p]))

    search = params_dict["search"]
    if search == "random":
        n_iter = params_dict.get("n_iter", 5)
        return ParameterSampler(params, n_iter)
    return ParameterGrid(params)