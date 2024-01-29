from typing import Union
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import numpy.typing as npt
import numpy as np


def check_stationarity(series: Union[npt.NDArray[np.float32], pd.Series]) -> None:
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/
    if isinstance(series, pd.Series):
        series = series.to_numpy()
    result = adfuller(series)

    print("ADF Statistic: %f" % result[0])
    print("p-value: %f" % result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print("\t%s: %.3f" % (key, value))

    if (result[1] <= 0.05) & (result[4]["5%"] > result[0]):
        print("\u001b[32mStationary\u001b[0m")
    else:
        print("\x1b[31mNon-stationary\x1b[0m")
