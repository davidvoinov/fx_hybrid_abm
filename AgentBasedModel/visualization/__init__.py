from AgentBasedModel.visualization.paper_style import (
    use_paper_style,
    PAPER_PALETTE,
    COLOR_TREATMENT,
    COLOR_CONTROL,
    COLOR_GOOD,
    COLOR_BAD,
    COLOR_NEUTRAL,
)

# Apply the paper style as soon as any plotting code from this package is
# imported. Importing a submodule (e.g. `from ...visualization.dashboards
# import ...`) executes this __init__, so main.py and downstream scripts pick
# up the style without each call site having to opt in.
use_paper_style()
# The per agent plots of the general purpose model are gone with the
# populations they drew: dividends, sentiments and strategy switching have
# no counterpart in a dealer intermediated FX market, and the collector
# behind them recorded a per agent snapshot every period that nothing read.
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
