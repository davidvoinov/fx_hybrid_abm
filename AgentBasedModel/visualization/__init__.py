from AgentBasedModel.visualization.paper_style import (
    use_paper_style,
    PAPER_PALETTE,
    COLOR_WITH_AMM,
    COLOR_NO_AMM,
    COLOR_GOOD,
    COLOR_BAD,
    COLOR_NEUTRAL,
)

# Apply the paper style as soon as any plotting code from this package is
# imported. Importing a submodule (e.g. `from ...visualization.dashboards
# import ...`) executes this __init__, so main.py and downstream scripts pick
# up the style without each call site having to opt in.
use_paper_style()
from AgentBasedModel.visualization.market import plot_price, plot_price_fundamental, plot_arbitrage, plot_dividend,\
    plot_orders, plot_volatility_price, plot_volatility_return, plot_liquidity
from AgentBasedModel.visualization.trader import plot_equity, plot_cash, plot_assets, plot_returns,\
    plot_strategies, plot_strategies2, plot_sentiments, plot_sentiments2
from AgentBasedModel.visualization.venue_plots import (
    # H1
    plot_execution_cost_curves, plot_cost_decomposition, plot_total_market_depth,
    # H2
    plot_cost_timeseries, plot_flow_allocation, plot_clob_spread_vs_amm_cost,
    plot_commonality, plot_amm_liquidity, plot_stress_flow_migration,
    # Context / auxiliary
    plot_environment, plot_fx_price, plot_clob_spread,
    plot_amm_reserves, plot_volume_slippage_profile,
)
