from typing import Literal
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge, TheilSenRegressor
from tqdm import tqdm


def compute_trend_target_on_dataframe(
    dataframe: pd.DataFrame,
    original_feature: str = "prices",
    forward_window: int = 60,
    target_name: str = "target",
    drop_na: bool = True,
    regression_type: Literal["linear", "theilsen","lasso","ridge"] = "linear",
) -> pd.DataFrame:
    """Compute a forward looking linear regression and extract the slope as the target corresponding the future of the trend.

    Args:
    ----
        dataframe (pd.DataFrame): The dataframe containin the prices.
        original_feature (str): The name of the feature containing the prices to run the regression on. Defaults to "prices".
        forward_window (int): The number of days to look forward.
        target_name (str, optional): The name of the feature created in the dataframe's columns. Defaults to "target".
        drop_na (bool, optional): Whether to keep NaN or not.. Defaults to True.
        regression_type (Literal[&quot;linear&quot;, &quot;theilsen&quot;, &quot;lasso&quot;, &quot;ridge&quot;], optional): The regressor to use, be carefull it could be long using the theilsen. Defaults to "linear".

    Returns:
    ----
        pd.DataFrame: The dataframe with the target column.
    """
    __REGRESSORS = {
        "linear": LinearRegression,
        "theilsen": TheilSenRegressor,
        "lasso": Lasso,
        "ridge": Ridge,
    }
    assert (
        regression_type in __REGRESSORS.keys()
    ), f"{regression_type} not in {__REGRESSORS.keys()}"
    assert isinstance(dataframe, pd.DataFrame), "dataframe must be a pandas DataFrame"
    assert original_feature in dataframe.columns, f"{original_feature} not in dataframe"

    dataframe[target_name] = np.nan
    forward_window = int(forward_window)
    X = np.array([i for i in range(0, forward_window)]).reshape(
        -1, 1
    )  # fake x values, it wont change in the loop.
    for index, _ in tqdm(
        dataframe.iloc[:-forward_window].iterrows(),
        total=len(dataframe) - forward_window,
        desc="Calculating targets",
    ):
        y = (
            dataframe["prices"].iloc[index : index + forward_window].to_numpy()
        )  # y values
        dataframe.loc[index, target_name] = (
            __REGRESSORS[regression_type]().fit(X, y).coef_[0]
        )  # fit linear regression and get slope

    if drop_na:
        return dataframe.dropna().reset_index(drop=True)
    return dataframe.reset_index(drop=True)
