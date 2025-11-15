"""CAKE Residuals Calculator"""

import math
import numpy as np


# Calculate residuals
def residuals(data, fit, init_param):
    n = len(data)
    k = len(init_param)
    res = data - fit
    rss = np.sum(res ** 2)
    r2 = 1 - (rss / np.sum((data - np.mean(data)) ** 2))
    r2_adj = 1 - (((1 - r2) * (n - 1)) / (n - k - 1))
    rmse = math.sqrt((rss / n))
    mae = np.sum(abs(res)) / n
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + np.log(n) * k
    return rss, r2, r2_adj, rmse, mae, aic, bic
