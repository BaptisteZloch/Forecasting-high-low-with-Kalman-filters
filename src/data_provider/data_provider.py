from typing import Callable, Dict
import numpy as np
import pandas as pd

from simulations.stochastic_models import StochasticModels
from utility.types import StochasticModelEnum


class DataProvider:
    __MODEL_STO_DICT: Dict[StochasticModelEnum, Callable] = {
        StochasticModelEnum.HESTON_WITH_JUMPS: StochasticModels.simulate_heston_process_with_jump,
        StochasticModelEnum.ARITHMETIC_BROWNIAN_MOTION: StochasticModels.simulate_arithmetic_brownian_motion_process,
        StochasticModelEnum.GEOMETRIC_BROWNIAN_MOTION: StochasticModels.simulate_geometric_brownian_motion_process,
        StochasticModelEnum.RANDOM_WALK: StochasticModels.simulate_random_walk_process,
    }

    __FILE_PATH = "../data/Appl_data.xlsx"

    @staticmethod
    def get_stochastic_fake_data(
        total_points: int = 1000,
        compute_returns: bool = True,
        compute_volatility: bool = True,
        compute_vol_of_vol: bool = True,
        model_type: StochasticModelEnum = StochasticModelEnum.HESTON_WITH_JUMPS,
    ):
        t, prices, _ = DataProvider.__MODEL_STO_DICT.get(
            model_type, StochasticModels.simulate_heston_process_with_jump
        )(total_points / 252, 100)
        df = pd.DataFrame(
            {
                "Date": t,
                "prices": prices,
            }
        )
        df["returns"] = df.prices.pct_change().fillna(0)
        df["vol"] = df.returns.rolling(30).std()
        df["volvol"] = df.vol.rolling(30).std()
        if compute_vol_of_vol is False:
            df = df.drop(columns=["volvol"])
        if compute_returns is False:
            df = df.drop(columns=["returns"])
        if compute_volatility is False:
            df = df.drop(columns=["vol"]).apply(lambda x: x * np.sqrt(252))
        return df.dropna()

    @staticmethod
    def get_tick_data(
        keep_tick_nb: bool = False,
        load_full_history: bool = False,
        compute_returns: bool = True,
        compute_volatility: bool = True,
        compute_vol_of_vol: bool = True,
    ) -> pd.DataFrame:
        if load_full_history is False:
            df = pd.read_excel(
                DataProvider.__FILE_PATH,
            )[["Date", "price", "nb tick"] if keep_tick_nb else ["Date", "price"]]
        else:
            df = pd.read_excel(DataProvider.__FILE_PATH, sheet_name="Feuil3")[
                ["Date", "price", "nb tick"] if keep_tick_nb else ["Date", "price"]
            ]
        df = df.rename(columns={"price": "prices"})

        df["returns"] = df.prices.pct_change().fillna(0)
        df["vol"] = df.returns.rolling(30).std()
        df["volvol"] = df.vol.rolling(30).std()
        if compute_vol_of_vol is False:
            df = df.drop(columns=["volvol"])
        if compute_returns is False:
            df = df.drop(columns=["returns"])
        if compute_volatility is False:
            df = df.drop(columns=["vol"])
        return df.dropna()

    # df = df.rename(columns={"Date": "t", "price": "prices"})

    # fig, ax = plt.subplots(3, 1, figsize=(20, 10))

    # ax[0].plot(df.prices, label="Stock Price")
    # ax[0].set_xlabel("Time")
    # ax[0].set_ylabel("Stock Price")
    # ax[0].legend()
    # ax[0].grid()

    # ax[1].plot(df.vol, color="blue", label="Volatility")
    # ax[1].set_xlabel("Time")
    # ax[1].set_ylabel("Volatility")
    # ax[1].legend()
    # ax[1].grid()
    # ax[2].plot(df.volvol, color="blue", label="Volatility of Volatility")
    # ax[2].set_xlabel("Time")
    # ax[2].set_ylabel("Volatility of Volatility")
    # ax[2].legend()
    # ax[2].grid()
