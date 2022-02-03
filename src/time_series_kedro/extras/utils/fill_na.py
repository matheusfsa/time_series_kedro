import pandas as pd
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
# Fill function
def rolling_fill(
    data: pd.Series,
    n: int
) -> pd.Series:

    """
    Fills Na values with the mean of the nearest values.
        Params:
            data: Original Series.
            n: Window size.
        Returns:
            pd.Series -> Series with missing values filled. 
    """

    data[data < 0] = 0

    out = np.copy(data)
    w_size = n//2

    # Create sliding window view -> [[x[i]-1, x[i], x[i+1]] for i in range(x.shape)]
    rolling_mean = np.hstack((np.full(w_size, np.nan), out, np.full(w_size, np.nan)))
    axis = 0 if len(rolling_mean.shape) == 1 else 1
    rolling_mean = np.nanmean(sliding_window_view(rolling_mean, (n+1,), axis=axis), axis=1)
    # Get Mean e filling nan values
    out[np.isnan(out)] = rolling_mean[np.isnan(out)]
    out[np.isnan(out)] = 0

    return out