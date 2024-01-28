# ALL THE CODE BELOW COMES FROM THE LIBRARY CREATED BY BAPTISTE ZLOCH :
# https://pypi.org/project/quant-invest-lab/
from typing import Optional, Union
import pandas as pd
import numpy as np

from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.io import show, output_notebook
from bokeh.models import (
    Model,
    ColumnDataSource,
)
from bokeh.transform import dodge


from quant_invest_lab.constants import (
    DT_FORMATTER,
    RETURNS_FORMATTER,
    SALMON_COLOR,
    VIOLET_COLOR,
    BACKGROUND_COLOR,
    GRID_COLOR,
    UNIT_PLOT_HEIGHT,
    UNIT_PLOT_WIDTH,
    TIMEFRAMES,
)


from backtest.metrics import (
    expectancy,
    profit_factor,
    omega_ratio,
    payoff_ratio,
    construct_report_dataframe,
)

from quant_invest_lab.types import Timeframe
from quant_invest_lab.utils import from_returns_to_bins_count

output_notebook()


def print_portfolio_strategy_report(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    timeframe: Timeframe = "1hour",
) -> pd.DataFrame:
    assert timeframe in TIMEFRAMES, f"Timeframe {timeframe} not supported"
    report_df = construct_report_dataframe(
        portfolio_returns, benchmark_returns, timeframe
    )
    if benchmark_returns is not None:
        print(f"\n{'  Returns statistical information  ':-^50}")

        print(
            f"Expected return annualized: {100*report_df.loc['Expected return', 'Portfolio']:.2f} % vs {100*report_df.loc['Expected return', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"CAGR: {100*report_df.loc['CAGR', 'Portfolio']:.2f} % vs {100*report_df.loc['CAGR', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Expected volatility annualized: {100*report_df.loc['Expected volatility', 'Portfolio']:.2f} % vs {100*report_df.loc['Expected volatility', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Specific volatility (diversifiable) annualized: {100*report_df.loc['Specific risk', 'Portfolio'] :.2f} %"
        )
        print(
            f"Systematic volatility annualized: {100*report_df.loc['Systematic risk', 'Portfolio'] :.2f} %"
        )
        print(
            f"Skewness: {report_df.loc['Skewness', 'Portfolio']:.2f} vs {report_df.loc['Skewness', 'Benchmark']:.2f} (buy and hold), <0 = left tail, >0 = right tail"
        )
        print(
            f"Kurtosis: {report_df.loc['Kurtosis', 'Portfolio']:.2f} vs {report_df.loc['Skewness', 'Benchmark']:.2f} (buy and hold)",
            ", >3 = fat tails, <3 = thin tails",
        )
        print(
            f"{timeframe}-95%-VaR: {100*report_df.loc['VaR', 'Portfolio']:.2f} % vs {100*report_df.loc['VaR', 'Benchmark']:.2f} % (buy and hold) -> the lower the better"
        )
        print(
            f"{timeframe}-95%-CVaR: {100*report_df.loc['CVaR', 'Portfolio']:.2f} % vs {100*report_df.loc['CVaR', 'Benchmark']:.2f} % (buy and hold) -> the lower the better"
        )

        print(f"\n{'  Strategy statistical information  ':-^50}")
        print(
            f"Max drawdown: {100*report_df.loc['Max drawdown', 'Portfolio']:.2f} % vs {100*report_df.loc['Max drawdown', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Kelly criterion: {100*report_df.loc['Kelly criterion', 'Portfolio']:.2f} % vs {100*report_df.loc['Kelly criterion', 'Benchmark']:.2f} % (buy and hold)"
        )
        print(
            f"Benchmark sensitivity (beta): {report_df.loc['Portfolio beta', 'Portfolio']:.2f} vs 1 (buy and hold)"
        )
        print(
            f"Excess return (alpha): {report_df.loc['Portfolio alpha', 'Portfolio']:.4f} vs 0 (buy and hold)"
        )
        print(f"Jensen alpha: {report_df.loc['Jensen alpha', 'Portfolio']:.4f}")
        print(f"Determination coefficient R²: {report_df.loc['R2', 'Portfolio']:.2f}")
        print(
            f"Tracking error annualized: {100*report_df.loc['Tracking error', 'Portfolio']:.2f} %"
        )
        print(f"\n{'  Strategy ratios  ':-^50}")
        print("No risk free rate considered for the following ratios.\n")
        print(
            f"Sharpe ratio annualized: {report_df.loc['Sharpe ratio', 'Portfolio']:.2f} vs {report_df.loc['Sharpe ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Sortino ratio annualized: {report_df.loc['Sortino ratio', 'Portfolio']:.2f} vs {report_df.loc['Sortino ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Burke ratio annualized: {report_df.loc['Burke ratio', 'Portfolio']:.2f} vs {report_df.loc['Burke ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Calmar ratio annualized: {report_df.loc['Calmar ratio', 'Portfolio']:.2f} vs {report_df.loc['Calmar ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Tail ratio annualized: {report_df.loc['Tail ratio', 'Portfolio']:.2f} vs {report_df.loc['Tail ratio', 'Benchmark']:.2f} (buy and hold)"
        )
        print(
            f"Treynor ratio annualized: {report_df.loc['Treynor ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Information ratio annualized: {report_df.loc['Information ratio', 'Portfolio']:.2f}"
        )
    else:
        print(f"\n{'  Returns statistical information  ':-^50}")

        print(
            f"Expected return annualized: {100*report_df.loc['Expected return', 'Portfolio']:.2f} %"
        )
        print(
            f"Expected volatility annualized: {100*report_df.loc['Expected volatility', 'Portfolio']:.2f} %"
        )
        print(
            f"Skewness: {report_df.loc['Skewness', 'Portfolio'] :.2f}, <0 = left tail, >0 = right tail"
        )
        print(
            f"Kurtosis: {report_df.loc['Kurtosis', 'Portfolio']:.2f}",
            ", >3 = fat tails, <3 = thin tails",
        )
        print(
            f"{timeframe}-95%-VaR: {100*report_df.loc['VaR', 'Portfolio']:.2f} % -> the lower the better"
        )
        print(
            f"{timeframe}-95%-CVaR: {100*report_df.loc['CVaR', 'Portfolio']:.2f} % -> the lower the better"
        )

        print(f"\n{'  Strategy statistical information  ':-^50}")
        print(f"Max drawdown: {100*report_df.loc['Max drawdown', 'Portfolio']:.2f} %")
        print(
            f"Kelly criterion: {100*report_df.loc['Kelly criterion', 'Portfolio']:.2f} %"
        )
        print(f"\n{'  Strategy ratios  ':-^50}")
        print("No risk free rate considered for the following ratios.\n")
        print(
            f"Sharpe ratio annualized: {report_df.loc['Sharpe ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Sortino ratio annualized: {report_df.loc['Sortino ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Burke ratio annualized: {report_df.loc['Burke ratio', 'Portfolio']:.2f}"
        )
        print(
            f"Calmar ratio annualized: {report_df.loc['Calmar ratio', 'Portfolio']:.2f}"
        )
        print(f"Tail ratio annualized: {report_df.loc['Tail ratio', 'Portfolio']:.2f}")
    return report_df


def print_backtest_report(
    trades_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    timeframe: Timeframe,
    initial_equity: Union[int, float] = 1000,
) -> None:
    good_trades = trades_df.loc[trades_df["trade_return"] > 0]
    bad_trades = trades_df.loc[trades_df["trade_return"] < 0]
    total_trades = trades_df.shape[0]

    print(f"\n{'  Strategy performances  ':-^50}")

    print(
        f'Strategy final net balance: {ohlcv_df["strategy_cum_returns"].iloc[-1]*initial_equity:.2f} $, return: {(ohlcv_df["strategy_cum_returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(
        f'Buy & Hold final net balance: {ohlcv_df["cum_returns"].iloc[-1]*initial_equity:.2f} $, returns: {(ohlcv_df["cum_returns"].iloc[-1]-1)*100:.2f} %'
    )
    print(f"Strategy winrate ratio: {100 * good_trades.shape[0] / total_trades:.2f} %")
    print(f"Strategy payoff ratio: {payoff_ratio(trades_df['trade_return']):.2f}")
    print(
        f"Strategy profit factor ratio: {profit_factor(trades_df['trade_return']):.2f}"
    )
    print(f"Strategy expectancy: {100*expectancy(trades_df['trade_return']):.2f} %")

    print_portfolio_strategy_report(
        ohlcv_df["strategy_returns"], ohlcv_df["returns"], timeframe
    )

    print(f"\n{'  Trades informations  ':-^50}")
    print(f"Mean trade return : {100*trades_df['trade_return'].mean():.2f} %")
    print(f"Median trade return : {100*trades_df['trade_return'].median():.2f} %")
    print(f'Mean trade volatility: {100*trades_df["trade_return"].std():.2f} %')
    print(
        f"Mean trade duration: {str((trades_df['trade_duration']).mean()).split('.')[0]}"
    )

    print(f"Total trades: {total_trades}")

    print(f"\n  Total good trades: {len(good_trades)}")
    print(f"  Mean good trades return: {100*good_trades['trade_return'].mean():.2f} %")
    print(
        f"  Median good trades return: {100*good_trades['trade_return'].median():.2f} %"
    )
    print(
        f"  Best trades return: {100*trades_df['trade_return'].max():.2f} % | Date: {trades_df.iloc[trades_df['trade_return'].idxmax()]['exit_date']} | Duration: {trades_df.iloc[trades_df['trade_return'].idxmax()]['trade_duration']}"  # type: ignore
    )
    print(
        f"  Mean good trade duration: {str((good_trades['trade_duration']).mean()).split('.')[0]}"
    )
    print(f"\n  Total bad trades: {len(bad_trades)}")
    print(f"  Mean bad trades return: {100*bad_trades['trade_return'].mean():.2f} %")
    print(
        f"  Median bad trades return: {100*bad_trades['trade_return'].median():.2f} %"
    )
    print(
        f"  Worst trades return: {100*trades_df['trade_return'].min():.2f} % | Date: {trades_df.iloc[trades_df['trade_return'].idxmin()]['exit_date']} | Duration: {trades_df.iloc[trades_df['trade_return'].idxmin()]['trade_duration']}"  # type: ignore
    )
    print(
        f"  Mean bad trade duration: {str((bad_trades['trade_duration']).mean()).split('.')[0]}"
    )

    print(f"\nExit reasons repartition: ")
    for reason, val in zip(
        trades_df.exit_reason.value_counts().index, trades_df.exit_reason.value_counts()
    ):
        print(f"- {reason}: {val}")


def plot_cumulative_performances(
    portfolio_cumulative_returns: pd.Series,
    benchmark_cumulative_returns: Optional[pd.Series] = None,
) -> Model:
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title="Cumulative performance",
        x_axis_label="Datetime",
        x_axis_type="datetime",
        y_axis_label="Cumulative performance (Log-scale)",
        y_axis_type="log",
        background_fill_color=BACKGROUND_COLOR,
    )
    if benchmark_cumulative_returns is not None:
        p.line(
            x=benchmark_cumulative_returns.index,
            y=benchmark_cumulative_returns,
            color=VIOLET_COLOR,
            line_width=2,
            legend_label="Benchmark cumulative return",
        )
    p.line(
        x=portfolio_cumulative_returns.index,
        y=portfolio_cumulative_returns,
        color=SALMON_COLOR,
        line_width=2,
        legend_label="Portfolio cumulative return",
    )
    p.xaxis.formatter = DT_FORMATTER

    p.legend.location = "center"

    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.grid.grid_line_alpha = 1
    return p


def plot_returns_distribution(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Model:
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title="Returns distribution",
        x_axis_label="returns",
        y_axis_label="Count",
        background_fill_color=BACKGROUND_COLOR,
    )
    hist, edges = np.histogram(
        portfolio_returns,
        density=True,
        bins=from_returns_to_bins_count(portfolio_returns),
    )

    p.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color=SALMON_COLOR,
        line_color="#FFFFFF00",
        alpha=0.45,
        legend_label="Strategy returns distribution",
    )
    if benchmark_returns is not None:
        hist, edges = np.histogram(
            benchmark_returns,
            density=True,
            bins=from_returns_to_bins_count(benchmark_returns),
        )
        p.quad(
            top=hist,
            bottom=0,
            left=edges[:-1],
            right=edges[1:],
            fill_color=VIOLET_COLOR,
            line_color="#FFFFFF00",
            alpha=0.45,
            legend_label="Benchmark returns distribution",
        )
    p.legend.location = "center"
    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.xaxis.formatter = RETURNS_FORMATTER
    p.grid.grid_line_alpha = 1
    return p


def plot_drawdown(
    portfolio_drawdown: pd.Series,
    benchmark_drawdown: Optional[pd.Series] = None,
) -> Model:
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title="Underwater area (drawdown)",
        x_axis_label="Datetime",
        x_axis_type="datetime",
        y_axis_label="Drawdown",
        background_fill_color=BACKGROUND_COLOR,
    )
    if benchmark_drawdown is not None:
        p.line(
            x=benchmark_drawdown.index,
            y=benchmark_drawdown,
            color=VIOLET_COLOR,
            line_width=1.5,
        )
        p.varea(
            x=benchmark_drawdown.index,
            y1=0,
            y2=benchmark_drawdown,
            color=VIOLET_COLOR,
            alpha=0.55,
            legend_label="Benchmark drawdown",
        )

    p.line(
        x=portfolio_drawdown.index,
        y=portfolio_drawdown,
        color=SALMON_COLOR,
        line_width=1.5,
    )
    p.varea(
        x=portfolio_drawdown.index,
        y1=0,
        y2=portfolio_drawdown,
        color=SALMON_COLOR,
        alpha=0.55,
        legend_label="Portfolio drawdown",
    )
    p.xaxis.formatter = DT_FORMATTER
    p.yaxis.formatter = RETURNS_FORMATTER
    p.legend.location = "center"

    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.grid.grid_line_alpha = 1
    return p


def plot_decile_performance(
    portfolio_returns: pd.Series, benchmark_returns: pd.Series
) -> Model:
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title="Conditional performance per decile",
        x_axis_label="Decile",
        y_axis_label="returns",
        background_fill_color=BACKGROUND_COLOR,
    )

    df_rets = pd.DataFrame(
        {"returns": benchmark_returns, "strategy_returns": portfolio_returns}
    )

    deciles = np.array(
        [
            (chunks["returns"].mean(), chunks["strategy_returns"].mean())
            for chunks in np.array_split(
                df_rets.sort_values(by="returns", ascending=True), 10
            )
        ]
    )
    source = ColumnDataSource(
        data={
            "decile": np.arange(1, deciles[:, 0].shape[0] + 1),
            "benchmark": deciles[:, 0],
            "portfolio": deciles[:, 1],
        }
    )

    p.vbar(
        x=dodge("decile", -0.2, range=p.x_range),
        top="benchmark",
        width=0.4,
        alpha=0.7,
        source=source,
        line_color="#FFFFFF00",
        fill_color=VIOLET_COLOR,
        legend_label="Benchmark decile",
    )
    p.vbar(
        x=dodge("decile", 0.2, range=p.x_range),
        top="portfolio",
        width=0.4,
        alpha=0.7,
        source=source,
        line_color="#FFFFFF00",
        fill_color=SALMON_COLOR,
        legend_label="Portfolio decile",
    )

    # p.vbar(x=x, top=deciles[:,-1], width=0.9, line_color="#FFFFFF00",fill_color=SALMON_COLOR, legend_label="Portfolio decile",)
    p.legend.location = "center"

    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.yaxis.formatter = RETURNS_FORMATTER
    p.grid.grid_line_alpha = 1
    p.x_range.start = 0
    p.x_range.end = 11
    return p


def plot_expected_return_profile(
    portfolio_cumulative_returns: pd.Series,
    benchmark_cumulative_returns: Optional[pd.Series] = None,
) -> Model:
    windows_bh = [
        day for day in range(5, portfolio_cumulative_returns.shape[0] // 3, 30)
    ]
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title="Expected return profile",
        x_axis_label="Horizon (in candles)",
        y_axis_label="Expected return",
        background_fill_color=BACKGROUND_COLOR,
    )
    if benchmark_cumulative_returns is not None:
        bench_expected_return_profile = [
            benchmark_cumulative_returns.rolling(window)
            .apply(lambda prices: (prices.iloc[-1] / prices.iloc[0]) - 1)
            .mean()
            for window in windows_bh
        ]
        p.circle(
            windows_bh,
            bench_expected_return_profile,
            size=5,
            color=VIOLET_COLOR,
        )
        p.line(
            x=windows_bh,
            y=bench_expected_return_profile,
            color=VIOLET_COLOR,
            line_width=2,
            legend_label="Benchmark expected return profile",
        )
    ptf_expected_return_profile = [
        portfolio_cumulative_returns.rolling(window)
        .apply(lambda prices: (prices.iloc[-1] / prices.iloc[0]) - 1)
        .mean()
        for window in windows_bh
    ]
    p.circle(
        windows_bh,
        ptf_expected_return_profile,
        size=5,
        color=SALMON_COLOR,
    )
    p.line(
        x=windows_bh,
        y=ptf_expected_return_profile,
        color=SALMON_COLOR,
        line_width=2,
        legend_label="Portfolio expected return profile",
    )
    p.legend.location = "center"

    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.yaxis.formatter = RETURNS_FORMATTER
    p.grid.grid_line_alpha = 1
    return p


def plot_rolling_sharpe_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Model:
    n_rolling = portfolio_returns.shape[0] // 10
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title=f"{n_rolling}-candles rolling Sharpe ratio",
        x_axis_label="Datetime",
        x_axis_type="datetime",
        y_axis_label="Sharpe ratio",
        background_fill_color=BACKGROUND_COLOR,
    )
    if benchmark_returns is not None:
        p.line(
            x=benchmark_returns.index,
            y=benchmark_returns.rolling(n_rolling)
            .apply(
                lambda rets: (365 * rets.mean()) / (rets.std() * (365**0.5)),
            )
            .fillna(0),
            color=VIOLET_COLOR,
            line_width=2,
            legend_label="Benchmark rolling sharpe ratio",
        )
    p.line(
        x=portfolio_returns.index,
        y=portfolio_returns.rolling(n_rolling)
        .apply(
            lambda rets: (365 * rets.mean()) / (rets.std() * (365**0.5)),
        )
        .fillna(0),
        color=SALMON_COLOR,
        line_width=2,
        legend_label="Portfolio rolling sharpe ratio",
    )
    p.xaxis.formatter = DT_FORMATTER
    p.legend.location = "center"

    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.grid.grid_line_alpha = 1
    return p


def plot_omega_curve(
    portfolio_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
) -> Model:
    thresholds = np.linspace(0.01, 0.75, 100)
    omega_bench = []
    omega_ptf = []
    for threshold in thresholds:
        omega_ptf.append(omega_ratio(portfolio_returns, threshold))
        if benchmark_returns is not None:
            omega_bench.append(omega_ratio(benchmark_returns, threshold))
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title=f"Omega curve",
        x_axis_label="Return threshold",
        y_axis_label="Omega ratio",
        background_fill_color=BACKGROUND_COLOR,
    )
    if benchmark_returns is not None:
        p.line(
            x=thresholds,
            y=omega_bench,
            color=VIOLET_COLOR,
            line_width=2,
            legend_label="Benchmark omega curve",
        )
    p.line(
        x=thresholds,
        y=omega_ptf,
        color=SALMON_COLOR,
        line_width=2,
        legend_label="Portfolio omega curve",
    )
    p.legend.location = "center"

    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.xaxis.formatter = RETURNS_FORMATTER
    p.grid.grid_line_alpha = 1
    return p


def plot_price_evolution(
    price: pd.Series,
) -> Model:
    p = figure(
        tools="pan,wheel_zoom,box_zoom,reset,save",
        width=UNIT_PLOT_WIDTH,
        height=UNIT_PLOT_HEIGHT,
        title="Price evolution",
        x_axis_label="Datetime",
        x_axis_type="datetime",
        y_axis_label="Price evolution",
        background_fill_color=BACKGROUND_COLOR,
    )

    p.line(
        x=price.index,
        y=price,
        color=VIOLET_COLOR,
        line_width=2,
        legend_label="Price",
    )
    p.xaxis.formatter = DT_FORMATTER

    p.legend.location = "center"

    p.add_layout(p.legend[0], "below")
    p.grid.grid_line_color = GRID_COLOR
    p.grid.grid_line_alpha = 1
    return p


def plot_from_trade_df(price_df: pd.DataFrame) -> None:
    """Plot historical price, equity progression, drawdown evolution and return distribution.

    Args:
    ----
        price_df (pd.DataFrame): The historical price dataframe.

    """
    output_notebook()
    grid = gridplot(
        [
            [
                plot_price_evolution(price_df["prices"]),
                plot_returns_distribution(
                    price_df["strategy_returns"], price_df["returns"]
                ),
            ],
            [
                plot_cumulative_performances(
                    price_df["strategy_cum_returns"], price_df["cum_returns"]
                ),
                plot_expected_return_profile(
                    price_df["strategy_cum_returns"], price_df["cum_returns"]
                ),
            ],
            [
                plot_drawdown(price_df["strategy_drawdown"], price_df["drawdown"]),
                plot_decile_performance(
                    price_df["strategy_returns"], price_df["returns"]
                ),
            ],
            [
                plot_rolling_sharpe_ratio(
                    price_df["strategy_returns"], price_df["returns"]
                ),
                plot_omega_curve(price_df["strategy_returns"], price_df["returns"]),
            ],
        ]  # type: ignore
    )

    show(grid)
