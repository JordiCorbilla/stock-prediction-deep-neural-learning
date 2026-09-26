import numpy as np

from stocklab.backtest import backtest_directional_strategy


def test_transaction_costs_reduce_directional_strategy_return():
    actual = np.array([0.01, -0.01, 0.02, -0.02, 0.01])
    predicted = np.array([0.01, -0.01, 0.01, -0.01, 0.01])

    free = backtest_directional_strategy(actual, predicted, transaction_cost_bps=0)
    costly = backtest_directional_strategy(actual, predicted, transaction_cost_bps=10)

    assert costly.annualized_return < free.annualized_return
    assert costly.average_turnover == free.average_turnover
