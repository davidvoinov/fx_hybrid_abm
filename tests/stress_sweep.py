#!/usr/bin/env python3
"""
Stress Sweep — Simulation + Analysis (unified)
===============================================

Функционал
----------
Этот файл объединяет генерацию данных и их анализ в единый пайплайн.

**Генерация (Block A + Block B)**

  Block A:  Полный перебор CLOB-конфигураций
            (MarketMaker, Random, Fundamentalist, Chartist, Universalist).
            MM 0-5, каждый другой тип 0-5, ≥1 не-MM.  Без AMM-пулов.

  Block B:  Полный перебор AMM-конфигураций из 1-5 пулов
            (CPMM + HFMM).  Всего 20 конфигураций.  Без CLOB-агентов.
            AMM полностью автономен: TCA по internal benchmark (mid-price
            пула), reference price берётся из ShadowCLOB (env.fair_price).

  Block C:  Гетерогенные HFMM-пулы.  3 варианта:
            tight (fee=5bp, A=50, reserves=500),
            balanced (fee=10bp, A=10, reserves=1000),
            deep (fee=20bp, A=5, reserves=2000).
            Полный перебор 1-5 пулов из 3 вариантов.  55 конфигураций.

  Block D:  Мультивенью CLOB+AMM с FX-тейкерами.
            CLOB: лучший состав из Block A (MM=2, Rnd=1, Fund=3, Chart=1, Univ=1).
            AMM:  лучший пул из Block C (1 tight HFMM, fee=5bp, A=50, R=500).
            Перебор amm_share_pct = 0, 10, 20, ..., 100.  11 конфигураций.
            FX-тейкеры маршрутизируют ордера между CLOB и AMM.

  Каждая конфигурация:
    * 1 000 итераций, фиксированный seed=42
    * Шок -20% fair price на t=350
    * Глубина DEPTH=1000: одинаковая для CLOB volume и AMM reserves
    * Без внешних FX-тейкеров (замкнутая экосистема)
    * Результат: CSV с посделочными записями (config_id, t, venue,
      side, quantity, fair_mid, exec_price, cost_bps)

**Анализ**

  После генерации автоматически запускается анализ:
    1. Обзор данных и качество (fill rate, inf-записи)
    2. Block A: влияние состава агентов на fv_cost
    3. Block B: влияние типа/числа пулов на cost_bps
    3c. Block C: гетерогенные HFMM-пулы (homo vs hetero, marginal effects)
    3d. Block D: мультивенью CLOB+AMM (sweep amm_share_pct 0-100%)
    4. Кросс-площадочное сравнение CLOB vs AMM
    5. Шоковая устойчивость (pre/post t=350)
    6. Лучшие конфигурации для CLOB, AMM, HFMM и мультивенью
    7. Выводы

  Recovery time (во всех блоках):
    Метрика: кол-во периодов после шока, пока rolling(20) median fv_cost
    не вернётся в пределы ±20 bps от pre-shock baseline.
    Блоки A-C: recovery per config. Блок D: combined + per-venue (CLOB/AMM).

  Метрика fv_cost (fair-value execution cost, bps):
    buy  -> 10000 * (exec_price - fair_mid) / fair_mid
    sell -> 10000 * (fair_mid - exec_price) / fair_mid
    Положительная = переплата, отрицательная = выгодная цена.

Выходные файлы
--------------
  output/stress/CLOB_config_mm*_rnd*_fund*_chart*_univ*.csv
  output/stress/AMM_config_cpmm{C}_hfmm{H}.csv
  output/stress/AMM_C_config_t{T}_b{B}_d{D}.csv
  output/stress/MV_config_amm{P}pct.csv

Запуск
------
    python tests/stress_sweep.py            # генерация + анализ
    python tests/stress_sweep.py --analyze  # только анализ (без перегенерации)
"""

import sys, os, random, math, re, glob, warnings, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional
from itertools import chain as _chain

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from AgentBasedModel.agents.agents import (
    ExchangeAgent, Random, Fundamentalist, Chartist, Universalist,
    MarketMaker, AMMProvider, AMMArbitrageur, Trader,
)
from AgentBasedModel.simulator.simulator import Simulator, SimulatorInfo
from AgentBasedModel.venues.clob import CLOBVenue, ShadowCLOB
from AgentBasedModel.venues.amm import CPMMPool, HFMMPool
from AgentBasedModel.environment.processes import MarketEnvironment
from AgentBasedModel.metrics.logger import MetricsLogger
from main import (
    build_parser as main_build_parser,
    _apply_preset_defaults as main_apply_preset_defaults,
    build_sim as main_build_sim,
    _seed_all as main_seed_all,
    _shock_metric_snapshot as main_shock_metric_snapshot,
)

# ======================================================================
#  Constants
# ======================================================================
N_ITER      = 1000
SHOCK_ITER  = 350
SHOCK_PCT   = -20.0       # -20 % crash
SEED        = 42
PRICE       = 100.0
DEPTH       = 1000        # comparable CLOB volume & AMM reserves per pool

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'stress')
os.makedirs(OUT_DIR, exist_ok=True)

CONSISTENT_OUT_DIR = os.path.join(OUT_DIR, 'consistent')
os.makedirs(CONSISTENT_OUT_DIR, exist_ok=True)

CONSISTENT_Q = 10.0
CONSISTENT_MAX_AGENT = 5
CONSISTENT_DEFAULT_SHORTLIST = 12
CONSISTENT_DEFAULT_HYBRID_PRESETS = (
    'mm_withdrawal',
    'flash_crash',
)
CONSISTENT_DEFAULT_HFMM_A_GRID = (5.0, 10.0, 25.0, 50.0)
CONSISTENT_DEFAULT_HFMM_FEE_GRID = (0.0005, 0.0010, 0.0015)
CONSISTENT_DEFAULT_HFMM_RESERVE_GRID = (500.0, 1000.0, 1500.0)

CONSISTENT_STAGE1_WEIGHTS = {
    'realized_cost_med': 2.0,
    'quoted_cost_q_med': 2.0,
    'spread_tail_mean': 1.5,
    'fill_rate': 2.0,
    'depth_tail_mean': 2.0,
    'systemic_liquidity_tail_mean': 1.0,
}
CONSISTENT_STAGE2_WEIGHTS = {
    'realized_cost_med': 1.0,
    'post_cost_med': 3.0,
    'spread_ratio': 2.0,
    'system_recovery_ticks': 3.0,
    'fill_rate': 1.0,
    'depth_tail_mean': 1.0,
    'depth_ratio': 2.0,
}
CONSISTENT_HYBRID_WEIGHTS = {
    'realized_cost_med': 1.0,
    'post_cost_med': 3.0,
    'spread_ratio': 2.5,
    'system_recovery_ticks': 3.0,
    'avg_basis_bps_0_20': 2.0,
    'fill_rate': 1.0,
    'depth_ratio': 2.0,
    'shock_systemic_liquidity': 1.5,
}

_CONSISTENT_MAIN_PARSER = main_build_parser(default_venue_choice_rule='liquidity_aware')


def _apply_research_preset(args, preset: str):
    """Keep research sweeps pinned to the isolated MM-withdrawal scenario."""
    if preset == 'mm_withdrawal':
        args.clob_amm_interaction = 'none'


# ######################################################################
#  PART 1 -- SIMULATION  (data generation)
# ######################################################################


# -- Monkey-patch: include exec_price in trade records -----------------
_orig_make_record = Trader._make_trade_record

def _patched_make_record(self, venue, side, Q, result, cls_info):
    rec = _orig_make_record(self, venue, side, Q, result, cls_info)
    rec['exec_price'] = result.get('exec_price', None)
    return rec

Trader._make_trade_record = _patched_make_record

# -- Monkey-patch: capture CLOB agent market orders --------------------
_captured_clob_trades: list = []

_orig_buy_market = Trader._buy_market
_orig_sell_market = Trader._sell_market


def _capturing_buy_market(self, quantity):
    q = round(quantity)
    if q <= 0 or not self.market.order_book['ask']:
        return _orig_buy_market(self, quantity)
    mid = self.market.price()
    total_cost, remaining = 0.0, q
    for o in self.market.order_book['ask']:
        if remaining <= 0:
            break
        fill = min(remaining, o.qty)
        total_cost += fill * o.price
        remaining -= fill
    filled = q - remaining
    unfilled = _orig_buy_market(self, quantity)
    if filled > 0 and mid and mid > 0:
        vwap = total_cost / filled
        _captured_clob_trades.append({
            'venue': 'clob', 'side': 'buy', 'quantity': filled,
            'exec_price': vwap,
            'cost_bps': 10_000.0 * (vwap - mid) / mid,
        })
    return unfilled


def _capturing_sell_market(self, quantity):
    q = round(quantity)
    if q <= 0 or not self.market.order_book['bid']:
        return _orig_sell_market(self, quantity)
    mid = self.market.price()
    total_cost, remaining = 0.0, q
    for o in self.market.order_book['bid']:
        if remaining <= 0:
            break
        fill = min(remaining, o.qty)
        total_cost += fill * o.price
        remaining -= fill
    filled = q - remaining
    unfilled = _orig_sell_market(self, quantity)
    if filled > 0 and mid and mid > 0:
        vwap = total_cost / filled
        _captured_clob_trades.append({
            'venue': 'clob', 'side': 'sell', 'quantity': filled,
            'exec_price': vwap,
            'cost_bps': 10_000.0 * (mid - vwap) / mid,
        })
    return unfilled


Trader._buy_market = _capturing_buy_market
Trader._sell_market = _capturing_sell_market

# -- Monkey-patch: capture AMM pool executions (arbitrageur trades) ----
_captured_amm_trades: list = []

_orig_cpmm_exec_buy = CPMMPool.execute_buy
_orig_cpmm_exec_sell = CPMMPool.execute_sell
_orig_hfmm_exec_buy = HFMMPool.execute_buy
_orig_hfmm_exec_sell = HFMMPool.execute_sell


def _make_amm_capture(orig_fn, side):
    def wrapper(self, Q):
        result = orig_fn(self, Q)
        if result.get('cost_bps', float('inf')) != float('inf'):
            name = getattr(self, 'pool_name', 'amm')
            _captured_amm_trades.append({
                'venue': name,
                'side': side,
                'quantity': round(Q, 6),
                'exec_price': result.get('exec_price', 0),
                'cost_bps': result.get('cost_bps', 0),
            })
        return result
    return wrapper


CPMMPool.execute_buy = _make_amm_capture(_orig_cpmm_exec_buy, 'buy')
CPMMPool.execute_sell = _make_amm_capture(_orig_cpmm_exec_sell, 'sell')
HFMMPool.execute_buy = _make_amm_capture(_orig_hfmm_exec_buy, 'buy')
HFMMPool.execute_sell = _make_amm_capture(_orig_hfmm_exec_sell, 'sell')


# -- Simulation helpers ------------------------------------------------

def _seed_all():
    random.seed(SEED)
    np.random.seed(SEED)


def _make_env():
    return MarketEnvironment(
        sigma_low=0.01, sigma_high=0.05,
        c_low=0.001, c_high=0.02,
        stress_start=-1, stress_end=-1,
        mode='piecewise',
        price=PRICE,
    )


# -- Custom simulation loop -------------------------------------------

def run_simulation(config_id, exchange, env, clob, amm_pools,
                   clob_agents, fx_traders,
                   lp_providers, arbitrageur,
                   sim_info=None):
    records = []

    for t in range(N_ITER):

        # 0. Price shock
        if t == SHOCK_ITER:
            env.apply_shock(SHOCK_PCT)
            try:
                mid = exchange.price()
            except Exception:
                mid = PRICE
            dp = mid * (SHOCK_PCT / 100.0)
            for order in _chain(*exchange.order_book.values()):
                order.price = round(order.price + dp, 2)
            for agent in clob_agents[:min(3, len(clob_agents))]:
                try:
                    agent.call()
                except Exception:
                    pass

        # 1. Environment step
        env.step()

        # 1b. Pre-trade arbitrage
        _captured_amm_trades.clear()
        if arbitrageur is not None:
            arbitrageur.arbitrage()

        # 2. SimulatorInfo capture + behaviour changes
        if sim_info is not None:
            try:
                sim_info.capture()
                for agent in clob_agents:
                    if type(agent) is Universalist:
                        agent.change_strategy(sim_info)
                    elif type(agent) is Chartist:
                        agent.change_sentiment(sim_info)
            except Exception:
                pass

        # 4. Shuffle & call all CLOB agents
        _captured_clob_trades.clear()
        random.shuffle(clob_agents)
        for agent in clob_agents:
            try:
                agent.call()
            except Exception:
                pass

        # 5. Dividends
        exchange.generate_dividend()

        # 6. Reset AMM period fees
        for pool in amm_pools.values():
            pool.reset_period_fees()

        # 6b. Drain captured CLOB-agent market orders + pre-trade arb
        period_trades = list(_captured_clob_trades)
        _captured_clob_trades.clear()
        period_trades.extend(_captured_amm_trades)
        _captured_amm_trades.clear()

        # 7. FX liquidity takers
        random.shuffle(fx_traders)
        for tr in fx_traders:
            result = tr.call()
            if result is not None:
                period_trades.append(result)

        # 8. Post-trade arbitrage
        if arbitrageur is not None:
            arbitrageur.arbitrage()

        # 8b. Drain captured AMM arb trades
        period_trades.extend(_captured_amm_trades)
        _captured_amm_trades.clear()

        # 9. LP adjust liquidity
        for lp in lp_providers:
            lp.update_liquidity()

        # 10. Record AMM state
        for pool in amm_pools.values():
            pool.record_state()

        # 11. Collect trade records
        fair_mid = env.fair_price if env.fair_price is not None else PRICE
        for tr in period_trades:
            exec_price = tr.get('exec_price')
            cost_bps = tr.get('cost_bps', 0.0)
            if exec_price is None or exec_price == float('inf'):
                sign = 1.0 if tr.get('side') == 'buy' else -1.0
                exec_price = fair_mid * (1.0 + sign * cost_bps / 10_000.0)
            records.append({
                'config_id': config_id,
                't':          t,
                'venue':      tr.get('venue', 'unknown'),
                'side':       tr.get('side', 'unknown'),
                'quantity':   tr.get('quantity', 0),
                'fair_mid':   round(fair_mid, 6),
                'exec_price': round(exec_price, 6),
                'cost_bps':   round(cost_bps, 4),
            })

    return pd.DataFrame(records)


# -- Block A helpers ---------------------------------------------------

MAX_AGENT = 5   # каждый тип 0..5


def _enumerate_clob_configs():
    """Yield (n_mm, n_rnd, n_fund, n_chart, n_univ).

    Каждый тип 0-5, хотя бы 1 не-MM агент.
    """
    for n_mm in range(MAX_AGENT + 1):
        for n_rnd in range(MAX_AGENT + 1):
            for n_fund in range(MAX_AGENT + 1):
                for n_chart in range(MAX_AGENT + 1):
                    for n_univ in range(MAX_AGENT + 1):
                        if n_rnd + n_fund + n_chart + n_univ == 0:
                            continue  # нужен хотя бы 1 тейкер
                        yield (n_mm, n_rnd, n_fund, n_chart, n_univ)


def _build_clob_agents(exchange, env, n_mm, n_rnd, n_fund, n_chart, n_univ):
    agents = []
    for i in range(n_mm):
        agents.append(MarketMaker(
            exchange, cash=1e4, env=env,
            alpha0=3.0 + i * 0.5, alpha1=300.0, alpha2=200.0,
            d0=50.0 - i * 5.0, d1=500.0, d2=250.0, d_min=5.0,
        ))
    for _ in range(n_rnd):
        agents.append(Random(exchange, cash=1e4))
    for _ in range(n_fund):
        agents.append(Fundamentalist(exchange, cash=1e4, access=1))
    for _ in range(n_chart):
        agents.append(Chartist(exchange, cash=1e4))
    for _ in range(n_univ):
        agents.append(Universalist(exchange, cash=1e4, access=1))
    return agents


def _config_id_a(n_mm, n_rnd, n_fund, n_chart, n_univ):
    return (f'CLOB_config_mm{n_mm}_rnd{n_rnd}'
            f'_fund{n_fund}_chart{n_chart}_univ{n_univ}')


# -- Block A runner ----------------------------------------------------

def run_block_a():
    configs = list(_enumerate_clob_configs())
    print('=' * 70)
    print(f'  BLOCK A -- CLOB agent compositions  ({len(configs)} configs, no AMM)')
    print(f'  MM 0-{MAX_AGENT}, others 0-{MAX_AGENT} each.  Types: MM, Random, Fund, Chartist, Univ')
    print('=' * 70)

    for idx, (n_mm, n_rnd, n_fund, n_chart, n_univ) in enumerate(configs, 1):
        config_id = _config_id_a(n_mm, n_rnd, n_fund, n_chart, n_univ)
        total = n_mm + n_rnd + n_fund + n_chart + n_univ
        print(f'\n  [{idx}/{len(configs)}] {config_id} (total={total}) ...',
              end=' ', flush=True)

        _seed_all()

        env       = _make_env()
        exchange  = ExchangeAgent(price=PRICE, std=2.0, volume=DEPTH)
        clob      = CLOBVenue(exchange)
        amm_pools = {}

        clob_agents = _build_clob_agents(
            exchange, env, n_mm, n_rnd, n_fund, n_chart, n_univ)

        sim_info = None
        if n_chart > 0 or n_univ > 0:
            sim_info = SimulatorInfo(exchange, clob_agents)

        df = run_simulation(
            config_id, exchange, env, clob, amm_pools,
            clob_agents=clob_agents, fx_traders=[],
            lp_providers=[], arbitrageur=None,
            sim_info=sim_info,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# -- Block B runner ----------------------------------------------------

def run_block_b():
    print('\n' + '=' * 70)
    print('  BLOCK B -- AMM configurations  (CPMM + HFMM <= 5, fully autonomous)')
    print('=' * 70)

    configs = []
    for total in range(1, 6):
        for n_cpmm in range(total + 1):
            configs.append((n_cpmm, total - n_cpmm))

    for n_cpmm, n_hfmm in configs:
        config_id = f'AMM_config_cpmm{n_cpmm}_hfmm{n_hfmm}'
        print(f'\n  [{config_id}] ...', end=' ', flush=True)

        _seed_all()

        env      = _make_env()
        exchange = ExchangeAgent(price=PRICE, std=2.0, volume=DEPTH)
        clob     = ShadowCLOB(env)

        amm_pools  = {}
        lp_providers = []

        for i in range(n_cpmm):
            name = f'cpmm_{i}' if n_cpmm > 1 else 'cpmm'
            pool = CPMMPool(x=float(DEPTH), y=float(DEPTH) * PRICE, fee=0.003)
            pool.pool_name = name
            amm_pools[name] = pool
            lp_providers.append(AMMProvider(pool, env))

        for i in range(n_hfmm):
            name = f'hfmm_{i}' if n_hfmm > 1 else 'hfmm'
            pool = HFMMPool(x=float(DEPTH), y=float(DEPTH) * PRICE,
                            A=10.0, fee=0.001, rate=PRICE)
            pool.pool_name = name
            amm_pools[name] = pool
            lp_providers.append(AMMProvider(pool, env))

        arb = AMMArbitrageur(clob, amm_pools, env=env) if amm_pools else None
        fx_traders = []

        df = run_simulation(
            config_id, exchange, env, clob, amm_pools,
            clob_agents=[],
            fx_traders=fx_traders,
            lp_providers=lp_providers, arbitrageur=arb,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# -- Block C: heterogeneous HFMM pools ---------------------------------

HFMM_VARIANTS = {
    'tight':    {'fee': 0.0005, 'A': 50.0, 'reserves': 500},
    'balanced': {'fee': 0.001,  'A': 10.0, 'reserves': 1000},
    'deep':     {'fee': 0.002,  'A': 5.0,  'reserves': 2000},
}


def _enumerate_amm_c_configs():
    """Yield (n_tight, n_balanced, n_deep) with 1 <= total <= 5."""
    for total in range(1, 6):
        for n_tight in range(total + 1):
            for n_balanced in range(total - n_tight + 1):
                n_deep = total - n_tight - n_balanced
                yield (n_tight, n_balanced, n_deep)


def _config_id_c(n_tight, n_balanced, n_deep):
    return f'AMM_C_config_t{n_tight}_b{n_balanced}_d{n_deep}'


def run_block_c():
    configs = list(_enumerate_amm_c_configs())
    print('\n' + '=' * 70)
    print(f'  BLOCK C -- Heterogeneous HFMM  ({len(configs)} configs, fully autonomous)')
    print(f'  Variants: tight (fee=5bp A=50 R=500), balanced (fee=10bp A=10 R=1000),')
    print(f'            deep (fee=20bp A=5 R=2000)')
    print('=' * 70)

    for n_tight, n_balanced, n_deep in configs:
        config_id = _config_id_c(n_tight, n_balanced, n_deep)
        print(f'\n  [{config_id}] ...', end=' ', flush=True)

        _seed_all()

        env      = _make_env()
        exchange = ExchangeAgent(price=PRICE, std=2.0, volume=DEPTH)
        clob     = ShadowCLOB(env)

        amm_pools    = {}
        lp_providers = []

        pool_idx = 0
        for variant_name, count in [('tight', n_tight),
                                     ('balanced', n_balanced),
                                     ('deep', n_deep)]:
            v = HFMM_VARIANTS[variant_name]
            for i in range(count):
                name = f'hfmm_{variant_name}_{pool_idx}'
                pool = HFMMPool(
                    x=float(v['reserves']),
                    y=float(v['reserves']) * PRICE,
                    A=v['A'], fee=v['fee'], rate=PRICE,
                )
                pool.pool_name = name
                amm_pools[name] = pool
                lp_providers.append(AMMProvider(pool, env))
                pool_idx += 1

        arb = AMMArbitrageur(clob, amm_pools, env=env,
                             routing='best') if amm_pools else None

        df = run_simulation(
            config_id, exchange, env, clob, amm_pools,
            clob_agents=[],
            fx_traders=[],
            lp_providers=lp_providers, arbitrageur=arb,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# -- Block D: Multi-venue with best Block A CLOB ----------------------

AMM_SHARE_GRID = list(range(0, 101, 10))  # 0%, 10%, ..., 100%
BLOCK_D_BEST_CLOB = dict(n_mm=2, n_rnd=1, n_fund=3, n_chart=1, n_univ=1)


def _build_block_d_fx_traders(exchange, clob, amm_pools, env, amm_pct):
    fx_traders = []

    for _ in range(15):
        fx_traders.append(Random(
            exchange, cash=1e4,
            clob=clob, amm_pools=amm_pools, env=env,
            amm_share_pct=amm_pct,
            beta_amm=0.05, cpmm_bias_bps=5.0, cost_noise_std=1.5,
            trade_prob=0.3, q_min=1, q_max=5, label='Noise',
        ))
    for _ in range(5):
        fx_traders.append(Fundamentalist(
            exchange, cash=1e4,
            clob=clob, amm_pools=amm_pools, env=env,
            amm_share_pct=amm_pct,
            beta_amm=0.05, cpmm_bias_bps=5.0, cost_noise_std=1.5,
            fundamental_rate=PRICE, fx_gamma=5e-3, fx_q_max=10,
        ))
    for _ in range(10):
        fx_traders.append(Random(
            exchange, cash=1e4,
            clob=clob, amm_pools=amm_pools, env=env,
            amm_share_pct=amm_pct,
            beta_amm=0.05, cpmm_bias_bps=5.0, cost_noise_std=1.5,
            trade_prob=0.4, q_min=1, q_max=2, label='Retail',
        ))
    for _ in range(3):
        fx_traders.append(Random(
            exchange, cash=5e4,
            clob=clob, amm_pools=amm_pools, env=env,
            amm_share_pct=amm_pct,
            beta_amm=0.05, cpmm_bias_bps=5.0, cost_noise_std=1.5,
            trade_prob=0.15, q_min=15, q_max=50, label='Institutional',
        ))

    return fx_traders


def run_block_d():
    """Sweep amm_share_pct using best Block A CLOB + best Block C AMM.

    CLOB: MM=2, Random=1, Fundamentalist=3, Chartist=1, Univ=1.
    AMM: 1 tight HFMM (fee=5bp, A=50, reserves=500).
    FX takers preserved from default architecture: 33 agents.
    """
    print('\n' + '=' * 70)
    print(f'  BLOCK D -- Multi-venue with best Block A CLOB  ({len(AMM_SHARE_GRID)} configs)')
    print(f'  CLOB: MM=2 Random=1 Fund=3 Chart=1 Univ=1')
    print(f'  FX takers: 15 Noise + 5 Fund + 10 Retail + 3 Institutional')
    print(f'  AMM: 1 tight HFMM (fee=5bp, A=50, R=500)')
    print(f'  Sweep: amm_share_pct = {AMM_SHARE_GRID}')
    print('=' * 70)

    for amm_pct in AMM_SHARE_GRID:
        config_id = f'MV_config_amm{amm_pct}pct'
        print(f'\n  [{config_id}] ...', end=' ', flush=True)

        _seed_all()

        env = _make_env()
        exchange = ExchangeAgent(price=PRICE, std=2.0, volume=DEPTH)
        clob = CLOBVenue(exchange)

        clob_agents = _build_clob_agents(
            exchange, env,
            BLOCK_D_BEST_CLOB['n_mm'],
            BLOCK_D_BEST_CLOB['n_rnd'],
            BLOCK_D_BEST_CLOB['n_fund'],
            BLOCK_D_BEST_CLOB['n_chart'],
            BLOCK_D_BEST_CLOB['n_univ'],
        )
        sim_info = SimulatorInfo(exchange, clob_agents)

        hfmm = HFMMPool(x=500.0, y=500.0 * PRICE, A=50.0, fee=0.0005, rate=PRICE)
        hfmm.pool_name = 'hfmm_tight'
        amm_pools = {'hfmm_tight': hfmm}
        lp_providers = [AMMProvider(hfmm, env)]
        arbitrageur = AMMArbitrageur(clob, amm_pools, env=env, routing='best')

        fx_traders = _build_block_d_fx_traders(exchange, clob, amm_pools, env, amm_pct)

        df = run_simulation(
            config_id, exchange, env, clob, amm_pools,
            clob_agents=clob_agents,
            fx_traders=fx_traders,
            lp_providers=lp_providers, arbitrageur=arbitrageur,
            sim_info=sim_info,
        )

        path = os.path.join(OUT_DIR, f'{config_id}.csv')
        df.to_csv(path, index=False)
        print(f'{len(df)} trades  ->  {path}')


# ######################################################################
#  PART 2 -- ANALYSIS  (reads CSVs, prints report)
# ######################################################################

pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', '{:.2f}'.format)


def _parse_clob_name(fname):
    m = re.match(r'CLOB_config_mm(\d+)_rnd(\d+)_fund(\d+)_chart(\d+)_univ(\d+)', fname)
    if not m:
        return {}
    return dict(n_mm=int(m[1]), n_rnd=int(m[2]), n_fund=int(m[3]),
                n_chart=int(m[4]), n_univ=int(m[5]))


def _parse_amm_name(fname):
    m = re.match(r'AMM_config_cpmm(\d+)_hfmm(\d+)', fname)
    if not m:
        return {}
    return dict(n_cpmm=int(m[1]), n_hfmm=int(m[2]))


def _parse_amm_c_name(fname):
    m = re.match(r'AMM_C_config_t(\d+)_b(\d+)_d(\d+)', fname)
    if not m:
        return {}
    return dict(n_tight=int(m[1]), n_balanced=int(m[2]), n_deep=int(m[3]))


def _parse_mv_name(fname):
    m = re.match(r'MV_config_amm(\d+)pct', fname)
    if not m:
        return {}
    return dict(amm_pct=int(m[1]))


def load_block(prefix):
    files = sorted(glob.glob(os.path.join(OUT_DIR, f'{prefix}_*.csv')))
    if prefix == 'AMM':
        files = [f for f in files if not os.path.basename(f).startswith('AMM_C_')]
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        fname = os.path.basename(f).replace('.csv', '')
        df['config_id'] = fname
        if prefix == 'CLOB':
            for k, v in _parse_clob_name(fname).items():
                df[k] = v
        elif prefix == 'AMM_C':
            for k, v in _parse_amm_c_name(fname).items():
                df[k] = v
        elif prefix == 'MV':
            for k, v in _parse_mv_name(fname).items():
                df[k] = v
        else:
            for k, v in _parse_amm_name(fname).items():
                df[k] = v
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_fv_cost(df):
    """
    Fair-value execution cost (bps):
      buy  -> 10000 * (exec_price - fair_mid) / fair_mid
      sell -> 10000 * (fair_mid - exec_price) / fair_mid
    Positive = trader paid more than fair value (real cost).
    """
    df = df.copy()
    df['fv_cost'] = np.where(
        df['side'] == 'buy',
        10000 * (df['exec_price'] - df['fair_mid']) / df['fair_mid'],
        10000 * (df['fair_mid'] - df['exec_price']) / df['fair_mid'],
    )
    return df


def _section(title):
    print('\n' + '=' * 80)
    print(f'  {title}')
    print('=' * 80)


# -- Recovery time computation -----------------------------------------

RECOVERY_WINDOW    = 20     # rolling window (periods)
RECOVERY_THRESHOLD = 20.0   # bps: recovered when within this of baseline
WARMUP_PERIODS     = 50     # skip initial periods for baseline


def compute_recovery_time(df, shock_iter=SHOCK_ITER, window=RECOVERY_WINDOW,
                          threshold=RECOVERY_THRESHOLD, warmup=WARMUP_PERIODS):
    """Periods after shock until rolling(window) median fv_cost returns
    to within ±threshold of pre-shock baseline.

    Returns (recovery_time, baseline_fv).
    recovery_time: int (periods), inf (never recovered), or nan (no data).
    """
    fin = df[np.isfinite(df['fv_cost'])].copy()
    pre = fin[(fin['t'] >= warmup) & (fin['t'] < shock_iter)]
    if len(pre) < window:
        return np.nan, np.nan
    baseline = pre['fv_cost'].median()

    post = fin[fin['t'] >= shock_iter]
    if len(post) == 0:
        return np.nan, baseline

    per_t = post.groupby('t')['fv_cost'].median().sort_index()
    roll = per_t.rolling(window, min_periods=window).median()
    within = (roll - baseline).abs() <= threshold
    recovered = within[within]
    if len(recovered) == 0:
        return float('inf'), baseline
    return max(0, recovered.index[0] - shock_iter), baseline


def _recovery_summary(rec_series, label):
    """Print one-line recovery summary for a series of recovery times."""
    finite = rec_series[np.isfinite(rec_series)]
    n_inf = (rec_series == float('inf')).sum()
    n_nan = rec_series.isna().sum()
    if len(finite):
        print(f'    {label}:  median={finite.median():.0f}  '
              f'mean={finite.mean():.0f}  min={finite.min():.0f}  '
              f'max={finite.max():.0f}  never={n_inf}  nodata={n_nan}')
    elif n_inf:
        print(f'    {label}:  never recovered ({n_inf} configs)')
    else:
        print(f'    {label}:  no data')


# -- Consistent sweep on the main simulator ----------------------------

def _finite_values(values):
    return [float(v) for v in values if math.isfinite(v)]


def _median_from(values):
    clean = _finite_values(values)
    return float(np.median(clean)) if clean else float('nan')


def _mean_from(values):
    clean = _finite_values(values)
    return float(np.mean(clean)) if clean else float('nan')


def _tail_slice(values, tail=100):
    return values[-tail:] if len(values) > tail else values


def _trade_frame_from_logger(sim):
    df = pd.DataFrame(sim.logger.trade_log)
    if df.empty:
        return pd.DataFrame(columns=['t', 'venue', 'cost_bps', 'quantity'])
    for col in ['t', 'venue', 'cost_bps', 'quantity']:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _median_trade_cost(df, start_t=None, end_t=None, venue=None):
    if df.empty:
        return float('nan')
    sub = df
    if start_t is not None:
        sub = sub[sub['t'] >= start_t]
    if end_t is not None:
        sub = sub[sub['t'] < end_t]
    if venue == 'amm':
        sub = sub[sub['venue'] != 'clob']
    elif venue is not None:
        sub = sub[sub['venue'] == venue]
    sub = sub[np.isfinite(sub['cost_bps'])]
    return float(sub['cost_bps'].median()) if len(sub) else float('nan')


def _fill_rate(df, venue=None):
    if df.empty:
        return float('nan')
    sub = df
    if venue == 'amm':
        sub = sub[sub['venue'] != 'clob']
    elif venue is not None:
        sub = sub[sub['venue'] == venue]
    if len(sub) == 0:
        return float('nan')
    return 100.0 * float(np.isfinite(sub['cost_bps']).mean())


def _average_actual_amm_share(sim):
    logger = sim.logger
    if not sim.amm_pools or not logger.iterations:
        return 0.0
    amm_share = []
    for idx in range(len(logger.iterations)):
        total = 0.0
        amm = 0.0
        for venue, volume in logger.flow_volume.items():
            if idx >= len(volume):
                continue
            val = float(volume[idx])
            total += val
            if venue != 'clob':
                amm += val
        amm_share.append(100.0 * amm / total if total > 0 else 0.0)
    return _mean_from(amm_share)


def _composite_rank(df, *, lower_better, higher_better, weights=None):
    ranked = df.copy()
    rank_cols = []
    weight_map = weights or {}
    for col in lower_better:
        rank_col = f'rank_{col}'
        ranked[rank_col] = ranked[col].rank(method='average', ascending=True, na_option='bottom')
        rank_cols.append(rank_col)
    for col in higher_better:
        rank_col = f'rank_{col}'
        ranked[rank_col] = ranked[col].rank(method='average', ascending=False, na_option='bottom')
        rank_cols.append(rank_col)
    total_weight = 0.0
    weighted_sum = 0.0
    for rank_col in rank_cols:
        metric = rank_col.replace('rank_', '', 1)
        weight = float(weight_map.get(metric, 1.0))
        weighted_sum = weighted_sum + ranked[rank_col] * weight
        total_weight += weight
        ranked[f'weight_{metric}'] = weight
    ranked['composite_rank'] = weighted_sum / total_weight if total_weight > 0 else ranked[rank_cols].mean(axis=1)
    ranked['composite_weight_total'] = total_weight
    return ranked.sort_values(['composite_rank', lower_better[0]])


def _amm_overrides(hfmm_A, hfmm_fee, hfmm_reserves):
    return {
        'hfmm_A': float(hfmm_A),
        'hfmm_fee': float(hfmm_fee),
        'hfmm_reserves': float(hfmm_reserves),
    }


def _consistent_clob_configs(limit=0):
    configs = []
    for n_mm in range(CONSISTENT_MAX_AGENT + 1):
        for n_noise in range(CONSISTENT_MAX_AGENT + 1):
            for n_fund in range(CONSISTENT_MAX_AGENT + 1):
                for n_chart in range(CONSISTENT_MAX_AGENT + 1):
                    for n_univ in range(CONSISTENT_MAX_AGENT + 1):
                        configs.append({
                            'n_mm': n_mm,
                            'n_noise': n_noise,
                            'n_clob_fund': n_fund,
                            'n_clob_chart': n_chart,
                            'n_clob_univ': n_univ,
                            'config_id': (
                                f'clob_mm{n_mm}_noise{n_noise}_fund{n_fund}'
                                f'_chart{n_chart}_univ{n_univ}'
                            ),
                        })
    return configs[:limit] if limit and limit > 0 else configs


def _build_consistent_args(seed, *, preset, n_iter, overrides):
    args = _CONSISTENT_MAIN_PARSER.parse_args([])
    args.preset = preset
    main_apply_preset_defaults(_CONSISTENT_MAIN_PARSER, args)
    _apply_research_preset(args, preset)
    args.seed = seed
    args.n_iter = n_iter
    args.silent = True
    args.no_plots = True
    args.no_summary = True
    args.comparison = False
    args.venue_choice_rule = 'liquidity_aware'
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _run_consistent_sim(seed, *, preset, n_iter, overrides):
    args = _build_consistent_args(seed, preset=preset, n_iter=n_iter, overrides=overrides)
    main_seed_all(seed)
    sim = main_build_sim(args)
    sim.simulate(args.n_iter, silent=True)
    return args, sim


def _extract_consistent_baseline_metrics(sim):
    logger = sim.logger
    trades = _trade_frame_from_logger(sim)
    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    clob_cost_q = logger.cost_series('clob', CONSISTENT_Q)
    return {
        'fill_rate': _fill_rate(trades, venue='clob'),
        'realized_cost_med': _median_trade_cost(trades, venue='clob'),
        'quoted_cost_q_med': _median_from(_tail_slice(clob_cost_q)),
        'spread_tail_mean': _mean_from(_tail_slice(logger.clob_qspr)),
        'depth_tail_mean': _mean_from(_tail_slice(depth_series)),
        'systemic_liquidity_tail_mean': _mean_from(_tail_slice(logger.systemic_liquidity_series)),
    }


def _safe_normalization_time(series, *, shock_iter, baseline,
                             direction, rel_tol, abs_tol=0.0,
                             window=5, horizon=100):
    if not math.isfinite(baseline):
        return float('nan')
    end = min(len(series), shock_iter + horizon)
    post = pd.Series(series[shock_iter:end], dtype='float64')
    rolled = post.rolling(window, min_periods=window).median()
    if direction == 'upper':
        target = max(baseline * (1.0 + rel_tol), baseline + abs_tol)
        hits = rolled[rolled <= target]
    else:
        target = baseline * (1.0 - rel_tol)
        hits = rolled[rolled >= target]
    if len(hits) == 0:
        return float('inf')
    return float(max(0, int(hits.index[0])))


def _local_shock_snapshot(sim):
    logger = sim.logger
    shock_iter = getattr(sim, 'shock_iter', None)
    if shock_iter is None:
        return {}

    n = len(logger.iterations)
    pre_start = max(0, shock_iter - 30)
    pre_end = min(n, shock_iter)
    shock_end = min(n, shock_iter + 5)
    post_start = min(n, shock_iter + 5)
    post_end = min(n, shock_iter + 20)

    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    basis_series = logger.max_venue_basis_series()

    pre_spread = _median_from(logger.clob_qspr[pre_start:pre_end])
    shock_spread = _median_from(logger.clob_qspr[shock_iter:shock_end])
    pre_depth = _median_from(depth_series[pre_start:pre_end])
    shock_depth = _median_from(depth_series[shock_iter:shock_end])
    shock_liq = _median_from(logger.systemic_liquidity_series[shock_iter:post_end])
    max_basis = _mean_from(basis_series[shock_iter:post_end])

    spread_ratio = float('nan')
    if math.isfinite(pre_spread) and pre_spread > 0 and math.isfinite(shock_spread):
        spread_ratio = shock_spread / pre_spread

    depth_ratio = float('nan')
    if math.isfinite(pre_depth) and pre_depth > 0 and math.isfinite(shock_depth):
        depth_ratio = shock_depth / pre_depth

    recovery_candidates = []
    for series, direction, rel_tol, abs_tol in [
        (logger.clob_qspr, 'upper', 0.50, 10.0),
        (depth_series, 'lower', 0.25, 0.0),
        (logger.systemic_liquidity_series, 'lower', 0.15, 0.0),
        (basis_series, 'upper', 0.25, 5.0),
    ]:
        baseline = _median_from(series[pre_start:pre_end])
        recovery_candidates.append(
            _safe_normalization_time(
                series,
                shock_iter=shock_iter,
                baseline=baseline,
                direction=direction,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        )

    if any(val == float('inf') for val in recovery_candidates):
        system_recovery = float('inf')
    else:
        system_recovery = max(_finite_values(recovery_candidates)) if _finite_values(recovery_candidates) else float('nan')

    return {
        'spread_ratio': spread_ratio,
        'depth_ratio': depth_ratio,
        'avg_basis_bps_0_20': max_basis,
        'system_recovery_ticks': system_recovery,
        'shock_spread_bps': shock_spread,
        'shock_depth': shock_depth,
        'shock_systemic_liquidity': shock_liq,
    }


def _extract_consistent_shock_metrics(sim):
    logger = sim.logger
    trades = _trade_frame_from_logger(sim)
    shock_iter = getattr(sim, 'shock_iter', None)
    try:
        snapshot = main_shock_metric_snapshot(sim) or {}
    except TypeError:
        snapshot = _local_shock_snapshot(sim)
    depth_series = [d.get('total', float('nan')) for d in logger.clob_depth]
    return {
        'fill_rate': _fill_rate(trades),
        'clob_fill_rate': _fill_rate(trades, venue='clob'),
        'amm_fill_rate': _fill_rate(trades, venue='amm'),
        'realized_cost_med': _median_trade_cost(trades),
        'post_cost_med': _median_trade_cost(trades, start_t=shock_iter),
        'post_clob_cost_med': _median_trade_cost(trades, start_t=shock_iter, venue='clob'),
        'post_amm_cost_med': _median_trade_cost(trades, start_t=shock_iter, venue='amm'),
        'spread_tail_mean': _mean_from(_tail_slice(logger.clob_qspr)),
        'depth_tail_mean': _mean_from(_tail_slice(depth_series)),
        'actual_amm_share_pct': _average_actual_amm_share(sim),
        'spread_ratio': snapshot.get('spread_ratio', float('nan')),
        'depth_ratio': snapshot.get('depth_ratio', float('nan')),
        'avg_basis_bps_0_20': snapshot.get('avg_basis_bps_0_20', float('nan')),
        'system_recovery_ticks': snapshot.get('system_recovery_ticks', float('nan')),
        'shock_spread_bps': snapshot.get('shock_spread_bps', float('nan')),
        'shock_depth': snapshot.get('shock_depth', float('nan')),
        'shock_systemic_liquidity': snapshot.get('shock_systemic_liquidity', float('nan')),
    }


def _aggregate_consistent_rows(rows, group_cols):
    df = pd.DataFrame(rows)
    value_cols = [col for col in df.columns if col not in set(group_cols) | {'seed'}]
    agg = df.groupby(group_cols, dropna=False)[value_cols].mean().reset_index()
    agg['n_seeds'] = df.groupby(group_cols, dropna=False).size().values
    return agg


def _clob_overrides_from_config(config, *, enable_amm, amm_share_pct):
    return {
        'enable_amm': 1 if enable_amm else 0,
        'amm_share_pct': float(amm_share_pct),
        'n_mm': int(config['n_mm']),
        'enable_clob_mm': 1,
        'n_noise': int(config['n_noise']),
        'n_clob_fund': int(config['n_clob_fund']),
        'n_clob_chart': int(config['n_clob_chart']),
        'n_clob_univ': int(config['n_clob_univ']),
    }


def run_consistent_clob_selection(*, seeds, base_seed, n_iter,
                                  shock_preset, shortlist,
                                  stage1_limit=0):
    _section('CONSISTENT BLOCK A -- Main-model CLOB selection')
    configs = _consistent_clob_configs(limit=stage1_limit)
    print(f'  Stage 1 configs: {len(configs)}  |  Stage 2 shortlist: {shortlist}')
    print(f'  Shock preset for final ranking: {shock_preset}')

    stage1_rows = []
    for idx, config in enumerate(configs, 1):
        if idx % 25 == 0 or idx == 1 or idx == len(configs):
            print(f'    Stage 1 progress: {idx}/{len(configs)}', flush=True)
        _args, sim = _run_consistent_sim(
            base_seed,
            preset='baseline',
            n_iter=n_iter,
            overrides=_clob_overrides_from_config(config, enable_amm=False, amm_share_pct=0),
        )
        metrics = _extract_consistent_baseline_metrics(sim)
        stage1_rows.append({**config, **metrics})

    stage1_df = _composite_rank(
        pd.DataFrame(stage1_rows),
        lower_better=['realized_cost_med', 'quoted_cost_q_med', 'spread_tail_mean'],
        higher_better=['fill_rate', 'depth_tail_mean', 'systemic_liquidity_tail_mean'],
        weights=CONSISTENT_STAGE1_WEIGHTS,
    )
    stage1_path = os.path.join(CONSISTENT_OUT_DIR, 'clob_stage1_baseline.csv')
    stage1_df.to_csv(stage1_path, index=False)

    shortlisted = stage1_df.head(shortlist).to_dict('records')
    stage2_rows = []
    for idx, config in enumerate(shortlisted, 1):
        print(f'    Stage 2 config {idx}/{len(shortlisted)}: {config["config_id"]}', flush=True)
        for seed in range(base_seed, base_seed + seeds):
            _args, sim = _run_consistent_sim(
                seed,
                preset=shock_preset,
                n_iter=n_iter,
                overrides=_clob_overrides_from_config(config, enable_amm=False, amm_share_pct=0),
            )
            metrics = _extract_consistent_shock_metrics(sim)
            stage2_rows.append({**config, 'seed': seed, **metrics})

    stage2_df = _aggregate_consistent_rows(
        stage2_rows,
        ['config_id', 'n_mm', 'n_noise', 'n_clob_fund', 'n_clob_chart', 'n_clob_univ'],
    )
    stage2_df = _composite_rank(
        stage2_df,
        lower_better=['realized_cost_med', 'post_cost_med', 'spread_ratio', 'system_recovery_ticks'],
        higher_better=['fill_rate', 'depth_tail_mean', 'depth_ratio'],
        weights=CONSISTENT_STAGE2_WEIGHTS,
    )
    stage2_path = os.path.join(CONSISTENT_OUT_DIR, 'clob_stage2_shock.csv')
    stage2_df.to_csv(stage2_path, index=False)

    winner = stage2_df.iloc[0].to_dict()
    winner_path = os.path.join(CONSISTENT_OUT_DIR, 'best_clob_config.csv')
    pd.DataFrame([winner]).to_csv(winner_path, index=False)

    print('\n  Best CLOB config for extrapolation into the main model:')
    print(f'    {winner["config_id"]}')
    print(f'    MM={int(winner["n_mm"])} Noise={int(winner["n_noise"])} '
          f'Fund={int(winner["n_clob_fund"])} Chart={int(winner["n_clob_chart"])} '
          f'Univ={int(winner["n_clob_univ"])}')
    print(f'    composite_rank={winner["composite_rank"]:.2f}  '
          f'post_cost_med={winner["post_cost_med"]:.2f}  '
          f'recovery={winner["system_recovery_ticks"]:.1f}')
    print(f'  Saved: {stage1_path}')
    print(f'  Saved: {stage2_path}')
    print(f'  Saved: {winner_path}')
    return winner, stage1_df, stage2_df


def run_consistent_hybrid_shock_sweep(best_clob, *, seeds, base_seed, n_iter,
                                      shock_presets, amm_share_grid,
                                      hfmm_A_grid, hfmm_fee_grid,
                                      hfmm_reserve_grid):
    _section('CONSISTENT BLOCK D -- Hybrid shock sweep on main model')
    print(f'  Using best CLOB: {best_clob["config_id"]}')
    print(f'  Shock presets: {", ".join(shock_presets)}')
    print(f'  AMM share grid: {amm_share_grid}')
    print(f'  HFMM A grid: {list(hfmm_A_grid)}')
    print(f'  HFMM fee grid: {list(hfmm_fee_grid)}')
    print(f'  HFMM reserve grid: {list(hfmm_reserve_grid)}')

    rows = []
    base_config = {
        'n_mm': int(best_clob['n_mm']),
        'n_noise': int(best_clob['n_noise']),
        'n_clob_fund': int(best_clob['n_clob_fund']),
        'n_clob_chart': int(best_clob['n_clob_chart']),
        'n_clob_univ': int(best_clob['n_clob_univ']),
        'config_id': best_clob['config_id'],
    }

    total = len(shock_presets) * len(amm_share_grid) * len(hfmm_A_grid) * len(hfmm_fee_grid) * len(hfmm_reserve_grid)
    done = 0
    for preset in shock_presets:
        for amm_pct in amm_share_grid:
            for hfmm_A in hfmm_A_grid:
                for hfmm_fee in hfmm_fee_grid:
                    for hfmm_reserves in hfmm_reserve_grid:
                        done += 1
                        print(
                            f'    Hybrid progress: {done}/{total}  preset={preset} amm={amm_pct}% '
                            f'A={hfmm_A:g} fee={hfmm_fee:.4f} R={hfmm_reserves:.0f}',
                            flush=True,
                        )
                        overrides = _clob_overrides_from_config(base_config, enable_amm=True, amm_share_pct=amm_pct)
                        overrides.update(_amm_overrides(hfmm_A, hfmm_fee, hfmm_reserves))
                        for seed in range(base_seed, base_seed + seeds):
                            _args, sim = _run_consistent_sim(
                                seed,
                                preset=preset,
                                n_iter=n_iter,
                                overrides=overrides,
                            )
                            metrics = _extract_consistent_shock_metrics(sim)
                            rows.append({
                                'preset': preset,
                                'amm_share_pct': amm_pct,
                                'hfmm_A': float(hfmm_A),
                                'hfmm_fee': float(hfmm_fee),
                                'hfmm_reserves': float(hfmm_reserves),
                                'seed': seed,
                                **base_config,
                                **metrics,
                            })

    detail_df = pd.DataFrame(rows)
    detail_path = os.path.join(CONSISTENT_OUT_DIR, 'hybrid_shock_sweep_detail.csv')
    detail_df.to_csv(detail_path, index=False)

    summary_df = _aggregate_consistent_rows(
        rows,
        ['preset', 'amm_share_pct', 'hfmm_A', 'hfmm_fee', 'hfmm_reserves',
         'config_id', 'n_mm', 'n_noise', 'n_clob_fund', 'n_clob_chart', 'n_clob_univ'],
    )
    ranked_parts = []
    for preset, sub in summary_df.groupby('preset', sort=False):
        ranked = _composite_rank(
            sub,
            lower_better=['realized_cost_med', 'post_cost_med', 'spread_ratio', 'system_recovery_ticks', 'avg_basis_bps_0_20'],
            higher_better=['fill_rate', 'depth_ratio', 'shock_systemic_liquidity'],
            weights=CONSISTENT_HYBRID_WEIGHTS,
        )
        ranked_parts.append(ranked)
    ranked_df = pd.concat(ranked_parts, ignore_index=True)
    summary_path = os.path.join(CONSISTENT_OUT_DIR, 'hybrid_shock_sweep_summary.csv')
    ranked_df.to_csv(summary_path, index=False)

    print('\n  Best hybrid configs by shock preset:')
    for preset, sub in ranked_df.groupby('preset', sort=False):
        best = sub.iloc[0]
        print(
            f'    {preset}: amm_share={int(best["amm_share_pct"])}%  '
            f'A={best["hfmm_A"]:.0f}  fee={best["hfmm_fee"] * 10_000:.0f}bps  '
            f'R={best["hfmm_reserves"]:.0f}  '
            f'cost={best["post_cost_med"]:.2f}  '
            f'spread_ratio={best["spread_ratio"]:.2f}  '
            f'recovery={best["system_recovery_ticks"]:.1f}  '
            f'score={best["composite_rank"]:.2f}'
        )
    print(f'  Saved: {detail_path}')
    print(f'  Saved: {summary_path}')
    return detail_df, ranked_df


def run_consistent_sweep(args):
    shock_presets = tuple(args.consistent_hybrid_presets) if args.consistent_hybrid_presets else CONSISTENT_DEFAULT_HYBRID_PRESETS
    active_shock_presets = {args.consistent_clob_shock_preset, *shock_presets}
    if args.consistent_n_iter <= SHOCK_ITER + 20 and any(preset != 'baseline' for preset in active_shock_presets):
        print(
            f'Warning: consistent_n_iter={args.consistent_n_iter} ends before the shock window is fully observed; '
            'stability and recovery rankings may contain NaN values.',
            flush=True,
        )
    winner, _stage1, _stage2 = run_consistent_clob_selection(
        seeds=args.consistent_seeds,
        base_seed=args.consistent_base_seed,
        n_iter=args.consistent_n_iter,
        shock_preset=args.consistent_clob_shock_preset,
        shortlist=args.consistent_shortlist,
        stage1_limit=args.consistent_stage1_limit,
    )
    run_consistent_hybrid_shock_sweep(
        winner,
        seeds=args.consistent_seeds,
        base_seed=args.consistent_base_seed,
        n_iter=args.consistent_n_iter,
        shock_presets=shock_presets,
        amm_share_grid=AMM_SHARE_GRID,
        hfmm_A_grid=args.consistent_hfmm_A_grid,
        hfmm_fee_grid=args.consistent_hfmm_fee_grid,
        hfmm_reserve_grid=args.consistent_hfmm_reserve_grid,
    )


def _safe_series(values, fallback=0.0):
    out = []
    for v in values:
        out.append(float(v) if (v is not None and math.isfinite(v)) else fallback)
    return out


def _rolling_corr(x, y, window):
    sx = pd.Series(x, dtype='float64')
    sy = pd.Series(y, dtype='float64')
    return sx.rolling(window=window, min_periods=window).corr(sy).tolist()


def _load_best_clob_for_spillover() -> Optional[dict]:
    path = os.path.join(CONSISTENT_OUT_DIR, 'best_clob_config.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    row = df.iloc[0]
    try:
        return {
            'n_mm': int(row.get('n_mm', 2)),
            'n_noise': int(row.get('n_noise', 12)),
            'n_clob_fund': int(row.get('n_clob_fund', 2)),
            'n_clob_chart': int(row.get('n_clob_chart', 0)),
            'n_clob_univ': int(row.get('n_clob_univ', 1)),
            'config_id': str(row.get('config_id', 'unknown')),
        }
    except Exception:
        return None


def run_spillover_visualization(args):
    _section('CONSISTENT SPILLOVER VISUALIZATION')

    overrides = {
        'enable_amm': 1,
        'amm_share_pct': float(args.spillover_amm_share),
        'hfmm_A': float(args.spillover_hfmm_A),
        'hfmm_fee': float(args.spillover_hfmm_fee),
        'hfmm_reserves': float(args.spillover_hfmm_reserves),
    }

    picked = 'defaults'
    if args.spillover_use_best_clob:
        best = _load_best_clob_for_spillover()
        if best is not None:
            overrides.update({
                'n_mm': best['n_mm'],
                'n_noise': best['n_noise'],
                'n_clob_fund': best['n_clob_fund'],
                'n_clob_chart': best['n_clob_chart'],
                'n_clob_univ': best['n_clob_univ'],
                'enable_clob_mm': 1,
            })
            picked = best.get('config_id', 'best_clob_config.csv')

    _cfg_args, sim = _run_consistent_sim(
        args.spillover_seed,
        preset=args.spillover_preset,
        n_iter=args.spillover_n_iter,
        overrides=overrides,
    )

    logger = sim.logger
    lag = max(1, int(args.spillover_lag))
    window = max(5, int(args.spillover_roll_window))
    split_at = getattr(sim, 'shock_iter', None)

    clob_depth = logger.clob_total_depth_series()
    amm_depth = logger.amm_total_depth_series()
    d_clob = logger._log_diff(clob_depth)
    d_amm = logger._log_diff(amm_depth)
    roll_corr = _rolling_corr(d_clob, d_amm, window)

    spill = logger.liquidity_spillover_metrics(lag=lag, split_at=split_at)

    pre_end = min(len(clob_depth), split_at) if split_at is not None else len(clob_depth)
    pre_clob = [x for x in clob_depth[:pre_end] if math.isfinite(x)]
    pre_amm = [x for x in amm_depth[:pre_end] if math.isfinite(x)]
    base_clob = (sum(pre_clob) / len(pre_clob)) if pre_clob else 1.0
    base_amm = (sum(pre_amm) / len(pre_amm)) if pre_amm else 1.0
    clob_idx = [x / base_clob if math.isfinite(x) and base_clob > 0 else float('nan') for x in clob_depth]
    amm_idx = [x / base_amm if math.isfinite(x) and base_amm > 0 else float('nan') for x in amm_depth]

    df = pd.DataFrame({
        't': list(range(len(clob_depth))),
        'clob_depth_total': _safe_series(clob_depth, fallback=float('nan')),
        'amm_depth_total': _safe_series(amm_depth, fallback=float('nan')),
        'clob_depth_index': _safe_series(clob_idx, fallback=float('nan')),
        'amm_depth_index': _safe_series(amm_idx, fallback=float('nan')),
    })
    if len(d_clob):
        df_d = pd.DataFrame({
            't': list(range(1, len(clob_depth))),
            'dlog_clob_depth': _safe_series(d_clob, fallback=float('nan')),
            'dlog_amm_depth': _safe_series(d_amm, fallback=float('nan')),
            f'rolling_corr_w{window}': _safe_series(roll_corr, fallback=float('nan')),
        })
        df = df.merge(df_d, on='t', how='left')

    csv_path = os.path.join(CONSISTENT_OUT_DIR, f'spillover_{args.spillover_preset}_seed{args.spillover_seed}.csv')
    df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)

    x0 = np.arange(len(clob_idx))
    axes[0].plot(x0, clob_idx, label='CLOB depth index', color='#1f77b4', lw=2)
    axes[0].plot(x0, amm_idx, label='AMM depth index', color='#2ca02c', lw=2)
    if split_at is not None:
        axes[0].axvline(split_at, color='#d62728', ls='--', lw=1.5, label='shock')
    axes[0].set_title('Liquidity Levels: CLOB vs AMM (indexed to pre-shock mean)')
    axes[0].set_ylabel('Index')
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=9)

    x1 = np.arange(1, len(clob_depth))
    axes[1].plot(x1, _safe_series(roll_corr, fallback=float('nan')), color='#9467bd', lw=2)
    axes[1].axhline(0.0, color='#333333', ls=':', lw=1.0)
    if split_at is not None:
        axes[1].axvline(split_at, color='#d62728', ls='--', lw=1.5)
    axes[1].set_title(f'Rolling correlation of liquidity changes (window={window})')
    axes[1].set_ylabel('corr')
    axes[1].grid(alpha=0.3)

    labels = ['AMM->CLOB', 'CLOB->AMM']
    full_betas = [spill['amm_to_clob'].get('beta', float('nan')), spill['clob_to_amm'].get('beta', float('nan'))]
    before = spill.get('before_after', {}).get('before', {})
    after = spill.get('before_after', {}).get('after', {})
    before_betas = [before.get('amm_to_clob', {}).get('beta', float('nan')),
                    before.get('clob_to_amm', {}).get('beta', float('nan'))]
    after_betas = [after.get('amm_to_clob', {}).get('beta', float('nan')),
                   after.get('clob_to_amm', {}).get('beta', float('nan'))]

    pos = np.arange(len(labels))
    wbar = 0.25
    axes[2].bar(pos - wbar, full_betas, width=wbar, color='#1f77b4', label='full sample')
    axes[2].bar(pos, before_betas, width=wbar, color='#ff7f0e', label='before')
    axes[2].bar(pos + wbar, after_betas, width=wbar, color='#2ca02c', label='after')
    axes[2].axhline(0.0, color='#333333', ls=':', lw=1.0)
    axes[2].set_xticks(pos)
    axes[2].set_xticklabels(labels)
    axes[2].set_ylabel('beta')
    axes[2].set_title(f'Directional spillovers (lag={lag})')
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=9)

    fig.suptitle(
        f'Spillover diagnostics | preset={args.spillover_preset} seed={args.spillover_seed} '
        f'| clob_cfg={picked}',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    png_path = os.path.join(CONSISTENT_OUT_DIR, f'spillover_{args.spillover_preset}_seed{args.spillover_seed}.png')
    fig.savefig(png_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(f'  spillover csv -> {csv_path}')
    print(f'  spillover png -> {png_path}')
    print(f'  avg flow shares: clob={spill.get("avg_flow_share_clob", float("nan")):.3f} '
          f'amm={spill.get("avg_flow_share_amm", float("nan")):.3f}')


# -- 1. Data overview --------------------------------------------------

def report_overview(clob, amm):
    _section('1. DATA OVERVIEW')
    n_clob_cfg = clob['config_id'].nunique() if len(clob) else 0
    n_amm_cfg  = amm['config_id'].nunique() if len(amm) else 0
    print(f'  CLOB configs : {n_clob_cfg}   ({len(clob):,} trade records)')
    print(f'  AMM  configs : {n_amm_cfg}   ({len(amm):,} trade records)')
    print(f'  Iterations   : {N_ITER} per config,  shock at t={SHOCK_ITER}')

    if len(clob):
        n_inf = (~np.isfinite(clob['cost_bps'])).sum()
        print(f'\n  CLOB data quality:')
        print(f'    Total trades           : {len(clob):,}')
        print(f'    Inf cost_bps (unfilled): {n_inf:,}  ({100*n_inf/len(clob):.1f}%)')
        print(f'    Fillable trades        : {len(clob)-n_inf:,}')

    if len(amm):
        n_inf_a = (~np.isfinite(amm['cost_bps'])).sum()
        print(f'\n  AMM data quality:')
        print(f'    Total trades           : {len(amm):,}')
        print(f'    Inf cost_bps           : {n_inf_a:,}  ({100*n_inf_a/len(amm):.1f}%)')
        print(f'    -> AMMs always fill orders (inf should be 0)')


# -- 2. Block A -- CLOB analysis ---------------------------------------

def report_block_a(clob):
    _section('2. BLOCK A -- CLOB: Effect of Agent Composition')

    if not len(clob):
        print('  (no data)')
        return None

    grp = clob.groupby('config_id').agg(
        n_mm   = ('n_mm', 'first'),
        n_rnd  = ('n_rnd', 'first'),
        n_fund = ('n_fund', 'first'),
        n_chart= ('n_chart', 'first'),
        n_univ = ('n_univ', 'first'),
        total  = ('cost_bps', 'size'),
        inf    = ('cost_bps', lambda s: (~np.isfinite(s)).sum()),
    ).reset_index()
    grp['fill_rate'] = 100 * (1 - grp['inf'] / grp['total'])

    filled = clob[np.isfinite(clob['cost_bps']) & np.isfinite(clob['fv_cost'])].copy()
    filled = filled[filled['fv_cost'].between(-5000, 5000)]

    med_fv = filled.groupby('config_id')['fv_cost'].agg(
        fv_cost_med='median', fv_cost_mean='mean',
    ).reset_index()

    cost_agg = filled.groupby('config_id')['cost_bps'].agg(
        cost_med='median', cost_mean='mean',
    ).reset_index()

    grp = grp.merge(med_fv, on='config_id', how='left')
    grp = grp.merge(cost_agg, on='config_id', how='left')

    # MM presence
    print('\n  2a. Effect of Market Maker count (MM=0..5):')
    print('  ' + '-' * 60)
    for mm_val in sorted(grp['n_mm'].unique()):
        sub = grp[grp['n_mm'] == mm_val]
        print(f'    MM={mm_val}  ({len(sub)} configs):')
        print(f'      Fill rate       : {sub["fill_rate"].mean():.1f}% '
              f'(min {sub["fill_rate"].min():.1f}%, max {sub["fill_rate"].max():.1f}%)')
        print(f'      Median fv_cost  : {sub["fv_cost_med"].median():.1f} bps')
        print(f'      Mean   fv_cost  : {sub["fv_cost_mean"].mean():.1f} bps')

    # Top/Bottom by fill rate
    print('\n  2b. Top 10 configs by fill rate:')
    print('  ' + '-' * 60)
    top = grp.nlargest(10, 'fill_rate')
    for _, r in top.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.0f} bps')

    print('\n  2c. Bottom 10 configs by fill rate:')
    print('  ' + '-' * 60)
    bot = grp.nsmallest(10, 'fill_rate')
    for _, r in bot.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.0f} bps')

    # Top/Bottom by fv_cost
    print('\n  2d. Top 10 configs by LOWEST median fv_cost (cheapest execution):')
    print('  ' + '-' * 60)
    cheap = grp.dropna(subset=['fv_cost_med']).nsmallest(10, 'fv_cost_med')
    for _, r in cheap.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.1f} bps')

    print('\n  2e. Top 10 configs by HIGHEST median fv_cost (most expensive):')
    print('  ' + '-' * 60)
    expen = grp.dropna(subset=['fv_cost_med']).nlargest(10, 'fv_cost_med')
    for _, r in expen.iterrows():
        print(f'    mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
              f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
              f'fill={r.fill_rate:.1f}%  fv_med={r.fv_cost_med:.1f} bps')

    # Per-agent-type marginal effect
    print('\n  2f. Marginal effect of each agent type (average across configs):')
    print('  ' + '-' * 60)
    for col, label in [('n_mm','MarketMaker'), ('n_rnd','Random'),
                       ('n_fund','Fundamentalist'), ('n_chart','Chartist'),
                       ('n_univ','Universalist')]:
        agg = grp.groupby(col).agg(
            fill_rate=('fill_rate', 'mean'),
            fv_cost_med=('fv_cost_med', 'median'),
            count=('config_id', 'count'),
        )
        print(f'\n    {label}:')
        for val, row in agg.iterrows():
            print(f'      n={int(val):d}  ({int(row["count"]):2d} cfgs)  '
                  f'fill={row["fill_rate"]:.1f}%  fv_med={row["fv_cost_med"]:.1f} bps')

    # 2g. Recovery time
    print('\n  2g. Post-shock recovery time (periods after t=%d):' % SHOCK_ITER)
    print('  ' + '-' * 60)

    rec_rows = []
    for cid in clob['config_id'].unique():
        sub = clob[clob['config_id'] == cid]
        rt, bl = compute_recovery_time(sub)
        meta = _parse_clob_name(cid)
        rec_rows.append({**meta, 'config_id': cid, 'recovery_t': rt})
    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['config_id', 'recovery_t']], on='config_id', how='left')

    for mm_val in sorted(rec_df['n_mm'].unique()):
        sub = rec_df[rec_df['n_mm'] == mm_val]
        _recovery_summary(sub['recovery_t'], f'MM={mm_val} ({len(sub)} cfgs)')

    print('\n    Per-agent-type marginal effect on recovery:')
    for col, label in [('n_mm','MarketMaker'), ('n_rnd','Random'),
                       ('n_fund','Fundamentalist'), ('n_chart','Chartist'),
                       ('n_univ','Universalist')]:
        for val in sorted(rec_df[col].unique()):
            sub = rec_df[rec_df[col] == val]
            fin = sub['recovery_t'][np.isfinite(sub['recovery_t'])]
            med_str = f'{fin.median():.0f}' if len(fin) else 'n/a'
            print(f'      {label} n={int(val):d}  ({len(sub):2d} cfgs)  '
                  f'recovery_med={med_str} periods')

    # Top 5 fastest / slowest recovery
    ranked = rec_df[np.isfinite(rec_df['recovery_t'])].sort_values('recovery_t')
    if len(ranked):
        print('\n    Top 5 fastest recovery:')
        for _, r in ranked.head(5).iterrows():
            print(f'      mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
                  f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
                  f'recovery={r.recovery_t:.0f} periods')
        print('\n    Top 5 slowest recovery:')
        for _, r in ranked.tail(5).iterrows():
            print(f'      mm={int(r.n_mm)} rnd={int(r.n_rnd)} fund={int(r.n_fund)} '
                  f'chart={int(r.n_chart)} univ={int(r.n_univ)}  '
                  f'recovery={r.recovery_t:.0f} periods')

    return grp


# -- 3. Block B -- AMM analysis ----------------------------------------

def report_block_b(amm):
    _section('3. BLOCK B -- AMM: Effect of Pool Composition')

    if not len(amm):
        print('  (no data)')
        return None

    grp = amm.groupby('config_id').agg(
        n_cpmm  = ('n_cpmm', 'first'),
        n_hfmm  = ('n_hfmm', 'first'),
        total   = ('cost_bps', 'size'),
        cost_med= ('cost_bps', 'median'),
        cost_mean=('cost_bps', 'mean'),
        fv_med  = ('fv_cost', 'median'),
        fv_mean = ('fv_cost', 'mean'),
    ).reset_index()
    grp['n_pools'] = grp['n_cpmm'] + grp['n_hfmm']

    print('\n  3a. All AMM configs (sorted by median cost_bps):')
    print('  ' + '-' * 60)
    for _, r in grp.sort_values('cost_med').iterrows():
        print(f'    cpmm={int(r.n_cpmm)} hfmm={int(r.n_hfmm)}  '
              f'trades={int(r.total):5d}  cost_med={r.cost_med:.1f}  '
              f'cost_mean={r.cost_mean:.1f}  fv_med={r.fv_med:.1f} bps')

    # Effect of pool type
    print('\n  3b. CPMM vs HFMM (marginal effect):')
    print('  ' + '-' * 60)
    for col, label in [('n_cpmm','CPMM'), ('n_hfmm','HFMM')]:
        agg = grp.groupby(col).agg(
            cost_med=('cost_med', 'mean'),
            fv_med=('fv_med', 'mean'),
            count=('config_id', 'count'),
        )
        print(f'\n    {label}:')
        for val, row in agg.iterrows():
            print(f'      n={int(val):d}  ({int(row["count"]):2d} cfgs)  '
                  f'cost_med={row["cost_med"]:.1f}  fv_med={row["fv_med"]:.1f} bps')

    # Effect of total pool count
    print('\n  3c. Effect of total pool count:')
    print('  ' + '-' * 60)
    for n, sub in grp.groupby('n_pools'):
        print(f'    {int(n)} pools  ({len(sub)} cfgs):  '
              f'cost_med={sub["cost_med"].mean():.1f}  fv_med={sub["fv_med"].mean():.1f} bps')

    # 3d. Recovery time
    print('\n  3d. Post-shock recovery time (periods after t=%d):' % SHOCK_ITER)
    print('  ' + '-' * 60)

    rec_rows = []
    for cid in amm['config_id'].unique():
        sub = amm[amm['config_id'] == cid]
        rt, bl = compute_recovery_time(sub)
        meta = _parse_amm_name(cid)
        rec_rows.append({**meta, 'config_id': cid, 'recovery_t': rt})
    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['config_id', 'recovery_t']], on='config_id', how='left')

    for _, r in rec_df.sort_values('recovery_t').iterrows():
        rt_str = f'{r.recovery_t:.0f}' if np.isfinite(r.recovery_t) else 'never'
        print(f'    cpmm={int(r.n_cpmm)} hfmm={int(r.n_hfmm)}  recovery={rt_str} periods')

    # CPMM vs HFMM effect
    for col, label in [('n_cpmm', 'CPMM'), ('n_hfmm', 'HFMM')]:
        print(f'\n    {label} marginal effect on recovery:')
        for val in sorted(rec_df[col].unique()):
            sub = rec_df[rec_df[col] == val]
            _recovery_summary(sub['recovery_t'], f'n={int(val)} ({len(sub)} cfgs)')

    return grp


# -- 3c. Block C -- heterogeneous HFMM --------------------------------

def report_block_c(amm_c):
    _section('3c. BLOCK C -- Heterogeneous HFMM Pools')

    if not len(amm_c):
        print('  (no data)')
        return None

    grp = amm_c.groupby('config_id').agg(
        n_tight   = ('n_tight', 'first'),
        n_balanced= ('n_balanced', 'first'),
        n_deep    = ('n_deep', 'first'),
        total     = ('cost_bps', 'size'),
        cost_med  = ('cost_bps', 'median'),
        cost_mean = ('cost_bps', 'mean'),
        fv_med    = ('fv_cost', 'median'),
        fv_mean   = ('fv_cost', 'mean'),
    ).reset_index()
    grp['n_pools'] = grp['n_tight'] + grp['n_balanced'] + grp['n_deep']
    grp['is_homo'] = (((grp['n_tight'] > 0).astype(int) +
                       (grp['n_balanced'] > 0).astype(int) +
                       (grp['n_deep'] > 0).astype(int)) == 1)

    print('\n  3c-1. All configs (sorted by median cost_bps):')
    print('  ' + '-' * 75)
    for _, r in grp.sort_values('cost_med').iterrows():
        tag = 'HOMO' if r.is_homo else 'MIX '
        print(f'    t={int(r.n_tight)} b={int(r.n_balanced)} d={int(r.n_deep)}'
              f'  [{tag}]  trades={int(r.total):5d}'
              f'  cost_med={r.cost_med:.1f}  cost_mean={r.cost_mean:.1f}'
              f'  fv_med={r.fv_med:.1f} bps')

    # Homogeneous vs heterogeneous
    homo = grp[grp['is_homo']]
    hetero = grp[~grp['is_homo']]
    print(f'\n  3c-2. Homogeneous vs Heterogeneous:')
    print('  ' + '-' * 60)
    if len(homo):
        print(f'    Homogeneous  ({len(homo):2d} cfgs):  '
              f'cost_med avg = {homo["cost_med"].mean():.1f} bps')
    if len(hetero):
        print(f'    Heterogeneous ({len(hetero):2d} cfgs):  '
              f'cost_med avg = {hetero["cost_med"].mean():.1f} bps')

    # Per-variant marginal effect
    print(f'\n  3c-3. Marginal effect of each variant:')
    print('  ' + '-' * 60)
    for col, label in [('n_tight', 'Tight (fee=5bp A=50 R=500)'),
                       ('n_balanced', 'Balanced (fee=10bp A=10 R=1000)'),
                       ('n_deep', 'Deep (fee=20bp A=5 R=2000)')]:
        agg = grp.groupby(col).agg(
            cost_med=('cost_med', 'mean'),
            fv_med=('fv_med', 'mean'),
            count=('config_id', 'count'),
        )
        print(f'\n    {label}:')
        for val, row in agg.iterrows():
            print(f'      n={int(val):d}  ({int(row["count"]):2d} cfgs)  '
                  f'cost_med={row["cost_med"]:.1f}  fv_med={row["fv_med"]:.1f} bps')

    # Best vs Block B reference
    best_c = grp.loc[grp['cost_med'].idxmin()]
    print(f'\n  3c-4. Best Block C vs Block B reference (HFMM=1, 63.0 bps):')
    print('  ' + '-' * 60)
    print(f'    Block B best : HFMM=1 (fee=10bp A=10 R=1000)  cost_med = 63.0 bps')
    print(f'    Block C best : t={int(best_c.n_tight)} b={int(best_c.n_balanced)} '
          f'd={int(best_c.n_deep)}  cost_med = {best_c.cost_med:.1f} bps')

    # 3c-5. Recovery time
    print(f'\n  3c-5. Post-shock recovery time (periods after t={SHOCK_ITER}):')
    print('  ' + '-' * 60)

    rec_rows = []
    for cid in amm_c['config_id'].unique():
        sub = amm_c[amm_c['config_id'] == cid]
        rt, bl = compute_recovery_time(sub)
        meta = _parse_amm_c_name(cid)
        rec_rows.append({**meta, 'config_id': cid, 'recovery_t': rt})
    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['config_id', 'recovery_t']], on='config_id', how='left')

    # Homo vs hetero recovery
    rec_df['n_pools'] = rec_df['n_tight'] + rec_df['n_balanced'] + rec_df['n_deep']
    rec_df['is_homo'] = (((rec_df['n_tight'] > 0).astype(int) +
                          (rec_df['n_balanced'] > 0).astype(int) +
                          (rec_df['n_deep'] > 0).astype(int)) == 1)
    homo = rec_df[rec_df['is_homo']]
    hetero = rec_df[~rec_df['is_homo']]
    _recovery_summary(homo['recovery_t'], f'Homogeneous  ({len(homo)} cfgs)')
    _recovery_summary(hetero['recovery_t'], f'Heterogeneous ({len(hetero)} cfgs)')

    # Per-variant marginal
    print('\n    Per-variant marginal effect on recovery:')
    for col, label in [('n_tight', 'Tight'), ('n_balanced', 'Balanced'),
                       ('n_deep', 'Deep')]:
        for val in sorted(rec_df[col].unique()):
            sub = rec_df[rec_df[col] == val]
            fin = sub['recovery_t'][np.isfinite(sub['recovery_t'])]
            med_str = f'{fin.median():.0f}' if len(fin) else 'n/a'
            print(f'      {label} n={int(val):d}  ({len(sub):2d} cfgs)  '
                  f'recovery_med={med_str} periods')

    return grp


# -- 3d. Block D -- multi-venue CLOB+AMM -------------------------------

def report_block_d(mv):
    _section('3d. BLOCK D -- Multi-venue CLOB+AMM')

    if not len(mv):
        print('  (no data)')
        return None

    # Filter to finite cost_bps for aggregation
    mv_fin = mv[np.isfinite(mv['cost_bps']) & np.isfinite(mv['fv_cost'])].copy()
    n_inf = len(mv) - len(mv_fin)
    if n_inf:
        print(f'\n  Note: {n_inf} trades with inf cost_bps excluded ({100*n_inf/len(mv):.1f}%)')

    # Per-config aggregate
    grp = mv_fin.groupby('config_id').agg(
        amm_pct   = ('amm_pct', 'first'),
        total     = ('cost_bps', 'size'),
        cost_med  = ('cost_bps', 'median'),
        cost_mean = ('cost_bps', 'mean'),
        fv_med    = ('fv_cost', 'median'),
        fv_mean   = ('fv_cost', 'mean'),
    ).reset_index().sort_values('amm_pct')

    # 3d-1. Overview sorted by amm_share_pct
    print('\n  3d-1. All configs (sorted by amm_share_pct):')
    print('  ' + '-' * 75)
    for _, r in grp.iterrows():
        print(f'    amm_pct={int(r.amm_pct):3d}%  trades={int(r.total):5d}'
              f'  cost_med={r.cost_med:.1f}  cost_mean={r.cost_mean:.1f}'
              f'  fv_med={r.fv_med:.1f} bps')

    # 3d-2. Venue split per config
    print('\n  3d-2. Venue split (CLOB vs AMM trades per config):')
    print('  ' + '-' * 75)
    for amm_pct in sorted(mv['amm_pct'].unique()):
        sub = mv[mv['amm_pct'] == amm_pct]
        n_clob = (sub['venue'] == 'clob').sum()
        n_amm  = (sub['venue'] != 'clob').sum()
        total  = len(sub)
        pct_amm = 100 * n_amm / total if total else 0

        sub_f = sub[np.isfinite(sub['fv_cost'])]
        clob_sub = sub_f[sub_f['venue'] == 'clob']
        amm_sub  = sub_f[sub_f['venue'] != 'clob']

        clob_med = clob_sub['fv_cost'].median() if len(clob_sub) else float('nan')
        amm_med  = amm_sub['fv_cost'].median() if len(amm_sub) else float('nan')

        n_inf_cfg = (sub['cost_bps'] == float('inf')).sum()
        inf_tag = f'  [inf={n_inf_cfg}]' if n_inf_cfg else ''

        print(f'    amm_pct={amm_pct:3d}%:  '
              f'CLOB={n_clob:4d} ({100-pct_amm:.0f}%)  '
              f'AMM={n_amm:4d} ({pct_amm:.0f}%)  '
              f'fv_clob={clob_med:.1f}  fv_amm={amm_med:.1f} bps{inf_tag}')

    # 3d-3. Pre/post shock resilience
    print(f'\n  3d-3. Shock resilience (pre/post t={SHOCK_ITER}):')
    print('  ' + '-' * 75)
    for amm_pct in sorted(mv['amm_pct'].unique()):
        sub = mv[mv['amm_pct'] == amm_pct]
        pre  = sub[sub['t'] < SHOCK_ITER]
        post = sub[sub['t'] >= SHOCK_ITER]
        pre_f  = pre[np.isfinite(pre['fv_cost']) & pre['fv_cost'].between(-5000, 5000)]
        post_f = post[np.isfinite(post['fv_cost']) & post['fv_cost'].between(-5000, 5000)]
        pre_m  = pre_f['fv_cost'].median() if len(pre_f) else float('nan')
        post_m = post_f['fv_cost'].median() if len(post_f) else float('nan')
        delta  = post_m - pre_m if np.isfinite(pre_m) and np.isfinite(post_m) else float('nan')
        print(f'    amm_pct={amm_pct:3d}%:  pre_fv={pre_m:.1f}  post_fv={post_m:.1f}'
              f'  delta={delta:+.1f} bps')

    # 3d-4. Best config
    best = grp.loc[grp['fv_med'].abs().idxmin()]
    print(f'\n  3d-4. Best multi-venue config (closest fv_cost to 0):')
    print('  ' + '-' * 60)
    print(f'    amm_share_pct = {int(best.amm_pct)}%')
    print(f'    trades        = {int(best.total):,}')
    print(f'    cost_bps med  = {best.cost_med:.1f} bps')
    print(f'    fv_cost  med  = {best.fv_med:.1f} bps')
    print(f'    |fv_cost|     = {abs(best.fv_med):.1f} bps')

    # 3d-5. Recovery time — combined + per-venue
    print(f'\n  3d-5. Post-shock recovery time (periods after t={SHOCK_ITER}):')
    print('  ' + '-' * 75)
    print(f'    {"amm_pct":>8s}  {"combined":>10s}  {"CLOB":>10s}  {"AMM":>10s}')
    print('    ' + '-' * 46)

    rec_rows = []
    for amm_pct in sorted(mv['amm_pct'].unique()):
        sub = mv[mv['amm_pct'] == amm_pct]
        rt_all, _ = compute_recovery_time(sub)
        clob_sub = sub[sub['venue'] == 'clob']
        amm_sub  = sub[sub['venue'] != 'clob']
        rt_clob, _ = compute_recovery_time(clob_sub) if len(clob_sub) else (np.nan, np.nan)
        rt_amm, _  = compute_recovery_time(amm_sub) if len(amm_sub) else (np.nan, np.nan)

        def _fmt(v):
            if v != v:  return 'n/a'
            if v == float('inf'):  return 'never'
            return f'{v:.0f}'
        print(f'    {amm_pct:7d}%  {_fmt(rt_all):>10s}  {_fmt(rt_clob):>10s}  {_fmt(rt_amm):>10s}')
        rec_rows.append({'amm_pct': amm_pct, 'recovery_all': rt_all,
                         'recovery_clob': rt_clob, 'recovery_amm': rt_amm})

    rec_df = pd.DataFrame(rec_rows)
    grp = grp.merge(rec_df[['amm_pct', 'recovery_all']], on='amm_pct', how='left')

    # Best combined recovery
    fin_rec = rec_df[np.isfinite(rec_df['recovery_all'])]
    if len(fin_rec):
        best_rec = fin_rec.loc[fin_rec['recovery_all'].idxmin()]
        print(f'\n    Fastest combined recovery: amm_pct={int(best_rec.amm_pct)}%  '
              f'{best_rec.recovery_all:.0f} periods')

    return grp


# -- 4. Cross-venue comparison -----------------------------------------

def report_cross_venue(clob, amm):
    _section('4. CROSS-VENUE: CLOB vs AMM')

    def fv_stats(df, label):
        fin = df[np.isfinite(df['cost_bps']) & np.isfinite(df['fv_cost'])].copy()
        fin = fin[fin['fv_cost'].between(-5000, 5000)]
        n_inf = (~np.isfinite(df['cost_bps'])).sum()
        fill = 100 * (1 - n_inf / len(df)) if len(df) else 0
        print(f'  {label}:')
        print(f'    Trades: {len(df):,}   Filled: {len(fin):,}  Fill rate: {fill:.1f}%')
        if len(fin):
            print(f'    fv_cost: med={fin["fv_cost"].median():.1f}  '
                  f'mean={fin["fv_cost"].mean():.1f}  '
                  f'p25={fin["fv_cost"].quantile(0.25):.1f}  '
                  f'p75={fin["fv_cost"].quantile(0.75):.1f} bps')
        print()

    fv_stats(clob[clob['n_mm'] >= 1], 'CLOB with MM (MM>=1)')
    fv_stats(clob[clob['n_mm'] == 0], 'CLOB without MM')
    fv_stats(amm, 'AMM (all configs)')


# -- 5. Shock resilience -----------------------------------------------

def report_shock(clob, amm):
    _section('5. SHOCK RESILIENCE (pre vs post shock)')

    def pre_post(df, label):
        pre  = df[df['t'] < SHOCK_ITER].copy()
        post = df[df['t'] >= SHOCK_ITER].copy()

        pre_f  = pre[np.isfinite(pre['cost_bps']) & np.isfinite(pre['fv_cost'])]
        post_f = post[np.isfinite(post['cost_bps']) & np.isfinite(post['fv_cost'])]
        pre_f  = pre_f[pre_f['fv_cost'].between(-5000, 5000)]
        post_f = post_f[post_f['fv_cost'].between(-5000, 5000)]

        pre_fill  = 100 * len(pre_f) / len(pre) if len(pre) else 0
        post_fill = 100 * len(post_f) / len(post) if len(post) else 0

        print(f'\n  {label}:')
        if len(pre_f):
            print(f'    PRE-shock  (t<{SHOCK_ITER}):  fill={pre_fill:.1f}%  '
                  f'fv_cost med={pre_f["fv_cost"].median():.1f}  '
                  f'mean={pre_f["fv_cost"].mean():.1f} bps')
        else:
            print(f'    PRE-shock: no filled trades')
        if len(post_f):
            print(f'    POST-shock (t>={SHOCK_ITER}):  fill={post_fill:.1f}%  '
                  f'fv_cost med={post_f["fv_cost"].median():.1f}  '
                  f'mean={post_f["fv_cost"].mean():.1f} bps')
        else:
            print(f'    POST-shock: no filled trades')

    pre_post(clob[clob['n_mm'] >= 1], 'CLOB with MM (MM>=1)')
    pre_post(clob[clob['n_mm'] == 0], 'CLOB without MM')
    pre_post(amm, 'AMM (all configs)')


# -- 6. Best configs ---------------------------------------------------

def report_best_configs(clob_grp, amm_grp, amm_c_grp=None, mv_grp=None):
    _section('6. BEST CONFIGURATIONS')

    # Best CLOB — closest to fair value (|fv_cost| → 0) with MM>=1
    if clob_grp is not None and len(clob_grp):
        valid = clob_grp.dropna(subset=['fv_cost_med'])
        if len(valid):
            # Prefer configs WITH market maker (MM>=1) — proper CLOB
            mm_pos = valid[valid['n_mm'] >= 1]
            pool = mm_pos if len(mm_pos) else valid
            pool = pool.copy()
            pool['abs_fv'] = pool['fv_cost_med'].abs()
            best = pool.loc[pool['abs_fv'].idxmin()]
            print(f'\n  >>> BEST CLOB configuration (closest to fair value, MM>=1):')
            print(f'      MM={int(best.n_mm)}  Random={int(best.n_rnd)}  '
                  f'Fund={int(best.n_fund)}  Chartist={int(best.n_chart)}  '
                  f'Univ={int(best.n_univ)}')
            print(f'      Fill rate    : {best.fill_rate:.1f}%')
            print(f'      cost_bps med : {best.cost_med:.1f} bps')
            print(f'      cost_bps mean: {best.cost_mean:.1f} bps')
            print(f'      fv_cost med  : {best.fv_cost_med:.2f} bps')
            print(f'      fv_cost mean : {best.fv_cost_mean:.2f} bps')
            print(f'      |fv_cost|    : {best.abs_fv:.2f} bps  (lower = better)')

    # Best AMM
    if amm_grp is not None and len(amm_grp):
        best_a = amm_grp.loc[amm_grp['cost_med'].idxmin()]
        print(f'\n  >>> BEST AMM configuration (lowest median cost_bps):')
        print(f'      CPMM pools={int(best_a.n_cpmm)}  HFMM pools={int(best_a.n_hfmm)}')
        print(f'      Trades       : {int(best_a.total):,}')
        print(f'      Fill rate    : 100.0%  (AMM always fills)')
        print(f'      cost_bps med : {best_a.cost_med:.1f} bps')
        print(f'      cost_bps mean: {best_a.cost_mean:.1f} bps')
        print(f'      fv_cost  med : {best_a.fv_med:.1f} bps')
        print(f'      fv_cost  mean: {best_a.fv_mean:.1f} bps')
        print(f'      |fv_cost|    : {abs(best_a.fv_med):.1f} bps  (lower = better)')

    # Best AMM_C (heterogeneous HFMM)
    if amm_c_grp is not None and len(amm_c_grp):
        best_c = amm_c_grp.loc[amm_c_grp['cost_med'].idxmin()]
        print(f'\n  >>> BEST heterogeneous HFMM (Block C, lowest median cost_bps):')
        print(f'      Tight={int(best_c.n_tight)}  Balanced={int(best_c.n_balanced)}  '
              f'Deep={int(best_c.n_deep)}')
        print(f'      Trades       : {int(best_c.total):,}')
        print(f'      cost_bps med : {best_c.cost_med:.1f} bps')
        print(f'      cost_bps mean: {best_c.cost_mean:.1f} bps')
        print(f'      fv_cost  med : {best_c.fv_med:.1f} bps')
        print(f'      |fv_cost|    : {abs(best_c.fv_med):.1f} bps')

    # Best multi-venue (Block D)
    if mv_grp is not None and len(mv_grp):
        best_mv = mv_grp.loc[mv_grp['fv_med'].abs().idxmin()]
        print(f'\n  >>> BEST multi-venue CLOB+AMM (Block D, closest fv_cost to 0):')
        print(f'      amm_share_pct = {int(best_mv.amm_pct)}%')
        print(f'      Trades       : {int(best_mv.total):,}')
        print(f'      cost_bps med : {best_mv.cost_med:.1f} bps')
        print(f'      cost_bps mean: {best_mv.cost_mean:.1f} bps')
        print(f'      fv_cost  med : {best_mv.fv_med:.1f} bps')
        print(f'      |fv_cost|    : {abs(best_mv.fv_med):.1f} bps')


# -- 7. Conclusions ----------------------------------------------------

def report_conclusions():
    _section('7. SUMMARY & CONCLUSIONS')
    print("""
    Blocks A-C: closed ecosystems (no external FX takers).
        A: CLOB agents trade among themselves (MM 0-5, others 0-5).
    B: AMM pools with only the arbitrageur trading.
    C: heterogeneous HFMM pools, routing='best'.
    Block D: open system with best Block A CLOB + best Block C AMM.
        CLOB = MM2 + Random1 + Fund3 + Chart1 + Univ1.
        AMM = 1 tight HFMM.  FX takers = 33 routed agents.

  COMPONENT INSIGHTS (Blocks A-C):
  1. MARKET MAKER is critical for CLOB stability.
     MM>=1 improves fill rate and execution quality.
     More MMs → tighter spreads, more depth, lower fv_cost.

  2. HFMM CHEAPER THAN CPMM.
     Pure HFMM ~ 63 bps, pure CPMM ~ 85 bps.
     Tight HFMM (fee=5bp, A=50) further reduces to 52 bps.

  3. POOL HETEROGENEITY does not help without LP migration.
     Smart routing worsens multi-pool configs (untouched pools drift).

  4. AMM IS STRUCTURALLY PREDICTABLE.
     IQR(fv_cost): AMM ~ 41 bps vs CLOB ~ 2200+ bps.
     AMM is 50-60x more predictable in execution cost variance.

  5. AMM IS SHOCK-RESILIENT.
     Post-shock delta: AMM ~ 3 bps, CLOB ~ 200+ bps.
     Arbitrageur realigns pool prices immediately.

  SYSTEM-LEVEL INSIGHTS (Block D — full architecture):
  6. OPTIMAL AMM SHARE ~ 30%.
     |fv_cost| minimized at amm_share=30%.
     CLOB provides deep, cheap execution; AMM stabilizes.

  7. AMM FV_COST GROWS WITH FLOW.
     More AMM traffic → more slippage (fee + price impact).
     AMM fv_cost: ~10 bps at amm=30% → ~15 bps at amm=100%.

  8. CLOB FV_COST IS VOLATILE BUT DEEP.
     30 noise traders + MM absorb institutional flow (q=15-50).
     Only 1 inf trade out of 270k (0.0% rejection rate).

  9. SHOCK RESILIENCE at system level.
     amm_share=50-60%: pre/post delta < 1 bps (most stable).
     CLOB-heavy configs show higher shock sensitivity.

  RECOVERY TIME INSIGHTS:
  10. RECOVERY TIME measures periods after shock until rolling(20)
      median fv_cost returns to within ±20 bps of pre-shock baseline.

  11. AMM RECOVERS FASTER THAN CLOB.
      Arbitrageur realigns AMM pool prices in few periods.
      CLOB needs book refill (new limit orders from MM + noise).

  12. AMM SHARE AFFECTS SYSTEM RECOVERY.
      Higher amm_share_pct → faster combined recovery
      (AMM component stabilizes system execution quality).
""")


# ######################################################################
#  Analysis runner
# ######################################################################

def run_analysis():
    print('\nLoading data for analysis...')
    clob  = load_block('CLOB')
    amm   = load_block('AMM')
    amm_c = load_block('AMM_C')
    mv    = load_block('MV')

    if len(clob):
        clob = add_fv_cost(clob)
    if len(amm):
        amm = add_fv_cost(amm)
    if len(amm_c):
        amm_c = add_fv_cost(amm_c)
    if len(mv):
        mv = add_fv_cost(mv)

    report_overview(clob, amm)
    clob_grp = report_block_a(clob)
    amm_grp  = report_block_b(amm)
    amm_c_grp = report_block_c(amm_c)
    mv_grp    = report_block_d(mv)
    report_cross_venue(clob, amm)
    report_shock(clob, amm)
    report_best_configs(clob_grp, amm_grp, amm_c_grp, mv_grp)
    report_conclusions()


# ######################################################################
#  Entry point
# ######################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stress Sweep -- simulation + analysis')
    parser.add_argument('--spillover-viz', action='store_true',
                        help='Run one simulation and save spillover visual diagnostics (PNG + CSV)')
    parser.add_argument('--spillover-preset', default='mm_withdrawal',
                        choices=['baseline', 'mm_withdrawal', 'mm_withdrawal_feedback', 'flash_crash', 'funding_liquidity_shock'],
                        help='Scenario preset used for spillover visualization (default: isolated mm_withdrawal)')
    parser.add_argument('--spillover-seed', type=int, default=42,
                        help='Seed for spillover visualization run (default: 42)')
    parser.add_argument('--spillover-n-iter', type=int, default=900,
                        help='Simulation length for spillover visualization (default: 900)')
    parser.add_argument('--spillover-amm-share', type=float, default=30.0,
                        help='Target AMM share in spillover visualization run (default: 30)')
    parser.add_argument('--spillover-hfmm-A', type=float, default=25.0,
                        help='HFMM amplification A in spillover visualization run (default: 25)')
    parser.add_argument('--spillover-hfmm-fee', type=float, default=0.001,
                        help='HFMM fee in spillover visualization run (default: 0.001)')
    parser.add_argument('--spillover-hfmm-reserves', type=float, default=1000.0,
                        help='HFMM reserves in spillover visualization run (default: 1000)')
    parser.add_argument('--spillover-lag', type=int, default=1,
                        help='Lag used in directional spillover regressions (default: 1)')
    parser.add_argument('--spillover-roll-window', type=int, default=30,
                        help='Rolling window for liquidity-change correlation (default: 30)')
    parser.add_argument('--spillover-use-best-clob', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Use best_clob_config.csv (if available) for spillover run (default: on)')
    parser.add_argument('--consistent', action='store_true',
                        help='Run the main-model consistent sweep instead of the legacy sandbox blocks')
    parser.add_argument('--consistent-seeds', type=int, default=5,
                        help='Sequential seed count for the consistent sweep (default: 5)')
    parser.add_argument('--consistent-base-seed', type=int, default=42,
                        help='First seed for the consistent sweep (default: 42)')
    parser.add_argument('--consistent-n-iter', type=int, default=900,
                        help='Simulation length for the consistent sweep (default: 900)')
    parser.add_argument('--consistent-shortlist', type=int, default=CONSISTENT_DEFAULT_SHORTLIST,
                        help='Top-K baseline CLOB configs re-evaluated under shock (default: 12)')
    parser.add_argument('--consistent-stage1-limit', type=int, default=0,
                        help='Optional cap on Stage-1 CLOB configs, useful for smoke tests (default: 0 = full grid)')
    parser.add_argument('--consistent-clob-shock-preset', default='mm_withdrawal',
                        choices=['baseline', 'mm_withdrawal', 'mm_withdrawal_feedback', 'flash_crash', 'funding_liquidity_shock'],
                        help='Shock preset used to rank shortlisted CLOB configs (default: isolated mm_withdrawal)')
    parser.add_argument('--consistent-hybrid-presets', nargs='*',
                        default=list(CONSISTENT_DEFAULT_HYBRID_PRESETS),
                        choices=['mm_withdrawal', 'mm_withdrawal_feedback', 'flash_crash', 'funding_liquidity_shock'],
                        help='Shock presets used for the CLOB+AMM sweep (default: isolated mm_withdrawal flash_crash funding_liquidity_shock)')
    parser.add_argument('--consistent-hfmm-A-grid', nargs='*', type=float,
                        default=list(CONSISTENT_DEFAULT_HFMM_A_GRID),
                        dest='consistent_hfmm_A_grid',
                        help='HFMM amplification grid for the hybrid sweep (default: 5 10 25 50)')
    parser.add_argument('--consistent-hfmm-fee-grid', nargs='*', type=float,
                        default=list(CONSISTENT_DEFAULT_HFMM_FEE_GRID),
                        dest='consistent_hfmm_fee_grid',
                        help='HFMM fee grid for the hybrid sweep as fractions (default: 0.0005 0.0010 0.0015)')
    parser.add_argument('--consistent-hfmm-reserve-grid', nargs='*', type=float,
                        default=list(CONSISTENT_DEFAULT_HFMM_RESERVE_GRID),
                        dest='consistent_hfmm_reserve_grid',
                        help='HFMM reserve grid for the hybrid sweep (default: 500 1000 1500)')
    parser.add_argument('--analyze', action='store_true',
                        help='Skip simulation, run analysis only on existing CSVs')
    parser.add_argument('--block-c', action='store_true',
                        help='Run only Block C (heterogeneous HFMM), then analyze all')
    parser.add_argument('--block-d', action='store_true',
                        help='Run only Block D (multi-venue CLOB+AMM), then analyze all')
    args = parser.parse_args()

    if args.spillover_viz:
        run_spillover_visualization(args)
    elif args.consistent:
        run_consistent_sweep(args)
    elif args.block_c:
        run_block_c()
    elif args.block_d:
        run_block_d()
    elif not args.analyze:
        run_block_a()
        run_block_b()
        run_block_c()
        run_block_d()
        print('\n' + '=' * 70)
        print('  Stress sweep complete.  Results -> output/stress/')
        print('=' * 70)

    if not args.consistent and not args.spillover_viz:
        run_analysis()
    print('\nDone.')
