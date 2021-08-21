import tensorflow as tf
import numpy as np

@tf.function
def __daily_returns(prices):
    return prices[:, :-1] / prices[:, 1:] - 1.0

@tf.function
def __downside_risk(returns, risk_free=0):
    adj_returns = returns - risk_free
    sqr_downside = tf.square(tf.clip_by_value(adj_returns, np.NINF, 0))
    return tf.sqrt(tf.reduce_mean(sqr_downside, axis=-1) * 252.0)

@tf.function
def portfolio_return_loss(Y_actual, Y_pred):
    start_value = tf.constant(1000.0)

    alloced = tf.multiply(Y_actual, tf.expand_dims(Y_pred, axis=1))
    pos_vals = tf.multiply(start_value, alloced)
    portfolio_values = tf.reduce_sum(pos_vals, axis=-1)

    end_val = tf.squeeze(portfolio_values[:, -1:])
    start_val = tf.squeeze(portfolio_values[:, 0:1])

    ret = (end_val - start_val) / start_val

    return tf.math.negative(ret)



@tf.function
def volatility_loss(Y_actual, Y_pred):
    start_value = tf.constant(1000.0)

    alloced = tf.multiply(Y_actual, tf.expand_dims(Y_pred, axis=1))
    pos_vals = tf.multiply(start_value, alloced)
    portfolio_values = tf.reduce_sum(pos_vals, axis=-1)

    port_rets = __daily_returns(portfolio_values)

    std_return = tf.math.reduce_std(port_rets, axis=-1, keepdims=False)

    return tf.math.abs(std_return)


@tf.function
def sortino_ratio_loss(Y_actual, Y_pred):
    """
    Calculates a sharpe ratio for the portfolio
    :param Y_actual: The price series for time frame T+1. i.e. future prices
    :param Y_pred: The portfolio allocations predicted
    :return: The sharpe ratio of the porfolio at time frame T+1
    """

    start_value = tf.constant(1000.0)

    risk_free_rate = tf.constant(0.0005)
    daily_rfr = tf.subtract(tf.pow(tf.add(1.0, risk_free_rate), tf.divide(1.0, 252.0)), 1.0)
    frequency = tf.sqrt(252.0)

    alloced = tf.multiply(Y_actual, tf.expand_dims(Y_pred, axis=1))
    pos_vals = tf.multiply(start_value, alloced)
    portfolio_values = tf.reduce_sum(pos_vals, axis=2)

    end_val = tf.squeeze(portfolio_values[:, -1:])
    start_val = tf.squeeze(portfolio_values[:, 0:1])

    ret = (end_val - start_val) / start_val

    port_rets = __daily_returns(portfolio_values)

    mean_return = tf.math.reduce_mean(port_rets, axis=-1, keepdims=False)
    std_return = __downside_risk(port_rets, risk_free_rate)
    # print(std_return)

    sharpe = (mean_return - risk_free_rate) / std_return

    sharpe = tf.sqrt(252.0) * sharpe

    return tf.math.negative(sharpe)

@tf.function
def sharpe_ratio_loss(Y_actual, Y_pred):
    """
    Calculates a sharpe ratio for the portfolio
    :param Y_actual: The price series for time frame T+1. i.e. future prices
    :param Y_pred: The portfolio allocations predicted
    :return: The sharpe ratio of the porfolio at time frame T+1
    """

    start_value = tf.constant(1000.0)

    risk_free_rate = tf.constant(0.0005)
    daily_rfr = tf.subtract(tf.pow(tf.add(1.0, risk_free_rate), tf.divide(1.0, 252.0)), 1.0)
    frequency = tf.sqrt(252.0)

    alloced = tf.multiply(Y_actual, tf.expand_dims(Y_pred, axis=1))
    pos_vals = tf.multiply(start_value, alloced)
    portfolio_values = tf.reduce_sum(pos_vals, axis=2)

    end_val = tf.squeeze(portfolio_values[:, -1:])
    start_val = tf.squeeze(portfolio_values[:, 0:1])

    port_rets = __daily_returns(portfolio_values)

    mean_return = tf.math.reduce_mean(port_rets, axis=-1, keepdims=False)
    std_return = tf.math.reduce_std(port_rets, axis=-1, keepdims=False)
    # print(std_return)

    sharpe = (mean_return - risk_free_rate) / std_return

    sharpe = tf.sqrt(252.0) * sharpe

    return tf.math.negative(sharpe)


if __name__ == '__main__':
    prices = tf.constant([[
        [1.0, 1.0, 1.0],
        [1.1, 1.5, 1.3],
        [1.2, 1.0, 1.5],
        [1.3, 0.8, 1.0],
        [1.4, 1.0, 1.7],
        [1.3, 1.2, 2.0],
    ],[
        [1.0, 1.0, 1.0],
        [3.01, -1.5, 1.3],
        [1.02, 1.0, 1.5],
        [0.02, -0.8, 1.0],
        [2.01, 0.5, 1.7],
        [0.001, -0.2, 2.0],
    ]])

    allocations = tf.constant([[0.5, 0.1, 0.4], [0.9, 0.1, 0]])

    sharpe = volatility_loss(prices, allocations)
    print(sharpe)