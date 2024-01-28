from typing import Tuple

import pandas as pd
from tqdm import tqdm

from backtest.metrics import drawdown


def backtest_startegy(
    df: pd.DataFrame,
    buying_threshold: float,
    selling_threshold: float,
    price_column: str = "prices",
    returns_column: str = "returns",
    date_column: str = "date",
    signal_column: str = "signal_filtered",
    verbose: bool = False,
    trading_fee: float = 0.0,
    slippage_effect: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest a strategy based on a threshold and a signal

    Args:
    ----
        df (pd.DataFrame): The dataframe containing the data to backtest
        buying_threshold (float): The threshold to buy
        selling_threshold (float): The threshold to sell
        price_column (str, optional): _description_. Defaults to "prices".
        returns_column (str, optional): _description_. Defaults to "returns".
        date_column (str, optional): _description_. Defaults to "date".
        signal_column (str, optional): _description_. Defaults to "signal_filtered".
        verbose (bool, optional): Print information during the backtesting loop. Defaults to False.
        trading_fee (float, optional): The trading fees to apply. Defaults to 0.0.
        slippage_effect (float, optional): The market slippage (liquidity) effect. Defaults to 0.0.

    Returns:
    ----
        Tuple[pd.DataFrame, pd.DataFrame]: THe backtested dataframe (same as input df but with new columns) and the trades dataframe storing all the trades
    """
    position_opened = False
    # Preparing the return for the backtest
    df["strategy_returns"] = df[returns_column]
    # Preparing the trades dataframe
    trades_df = pd.DataFrame(
        columns=[
            "entry_date",
            "entry_price",
            "entry_reason",
            "exit_date",
            "exit_price",
            "exit_reason",
            "trade_return",
        ]
    )
    current_trade = {}
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Backtesting..."):
        if position_opened is True and row[signal_column] >= selling_threshold:
            if verbose:
                print(f"SELLING at {row[price_column]}")
            position_opened = False
            df.loc[index, "strategy_returns"] = df.loc[index, "strategy_returns"] - (
                trading_fee + slippage_effect
            )
            # Fill the current trade
            current_trade["exit_date"] = row[date_column]
            current_trade["exit_price"] = row[price_column]
            current_trade["exit_reason"] = "SELLING"
            current_trade["trade_return"] = (
                (current_trade["exit_price"]) / current_trade["entry_price"]
            ) - 1
            # Add the trade to the trades dataframe
            trades_df = pd.concat(
                [trades_df, pd.DataFrame([current_trade])], ignore_index=True
            )
            # Reset the current trade
            current_trade = {}
        elif position_opened is False and row[signal_column] <= buying_threshold:
            if verbose:
                print(f"BUYING at {row[price_column]}")
            position_opened = True
            df.loc[index, "strategy_returns"] = df.loc[index, "strategy_returns"] - (
                trading_fee + slippage_effect
            )
            # Fill the current trade
            current_trade["entry_date"] = row[date_column]
            current_trade["entry_price"] = row[price_column]
            current_trade["entry_reason"] = "BUYING"
        else:
            df.loc[index, "strategy_returns"] = 0.0
    trades_df["trade_duration"] = trades_df["exit_date"] - trades_df["entry_date"]
    df["strategy_cum_returns"] = (df["strategy_returns"] + 1).cumprod()
    df["cum_returns"] = (df[returns_column] + 1).cumprod()
    df["drawdown"] = drawdown(df[returns_column])
    df["strategy_drawdown"] = drawdown(df["strategy_returns"])
    df["date_index"] = df[date_column]
    df = df.set_index("date_index")

    return df, trades_df
