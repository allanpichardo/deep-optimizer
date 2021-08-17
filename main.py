import tensorflow as tf
from deepoptimizer.tools.portfolio import Portfolio
from deepoptimizer.losses import sharpe_ratio_loss, volatility_loss, portfolio_return_loss, sortino_ratio_loss
import numpy as np
import argparse


def sharpe(p, w):
    return -sharpe_ratio_loss(p, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Portfolio.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--train_start_date', default='2014-01-01')
    parser.add_argument('--train_end_date', default='2020-01-01')
    parser.add_argument('--tickers', nargs="+", default=["FBALX", "FBSOX", "FAGIX", "FIPDX"])
    parser.add_argument('--loss_weights', nargs='+', default=[0.6, 0.1, 0.3], type=float)

    args = parser.parse_args()

    tf.random.set_seed(5)

    tickers = args.tickers.copy()
    number_of_assets = len(tickers)

    portfolio = Portfolio(tickers=tickers, start_date=args.train_start_date, end_date=args.train_end_date)
    dataset = portfolio.create_dataset(step=1).shuffle(50000).batch(32).cache().prefetch(tf.data.experimental.AUTOTUNE)

    input_prices = tf.keras.layers.Input((252, number_of_assets), name="price_input")
    input_indicators = tf.keras.layers.Input((252, 4), name="indicators_input")
    p = tf.keras.layers.LSTM(8, activation='tanh', return_sequences=True)(input_prices)
    p = tf.keras.layers.LSTM(8, activation='tanh', return_sequences=False)(p)
    p = tf.keras.layers.BatchNormalization()(p)

    i = tf.keras.layers.Conv1D(8, 3, activation='tanh')(input_indicators)
    i = tf.keras.layers.Conv1D(8, 3, activation='tanh')(i)
    i = tf.keras.layers.Conv1D(1, 1, activation='tanh')(i)
    i = tf.keras.layers.BatchNormalization()(i)
    i = tf.keras.layers.Flatten()(i)

    x = tf.keras.layers.Concatenate()([p, i])
    allocations = tf.keras.layers.Dense(number_of_assets, activation='softmax', name="allocations")(x)

    x = tf.keras.layers.Concatenate()([allocations, p])
    returns = tf.keras.layers.Dense(1, activation='linear', name='returns')(x)
    volatility = tf.keras.layers.Dense(1, activation='linear', name='volatility')(x)

    model = tf.keras.Model(inputs=[input_prices, input_indicators], outputs=[allocations, returns, volatility])

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=1.0),
        loss=[sortino_ratio_loss, portfolio_return_loss, volatility_loss],
        metrics=[],
        loss_weights=args.loss_weights
    )

    print("----Training Start----")
    print("Tickers:\n{}".format(args.tickers))
    print("--------")
    model.fit(dataset, epochs=args.epochs, verbose=1)

    print("----Optimization Start----")
    pred = Portfolio(tickers=tickers, start_date='2020-01-01', end_date='2021-01-01').create_dataset(skip_y=True).batch(
        1)
    port_pred = model.predict(pred.take(1), verbose=1)
    print("Allocations:\n{}".format(args.tickers))
    print(np.around(port_pred[0], decimals=2))
    print("Returns:")
    print(np.around(port_pred[1], decimals=2))
    print("Stdev:")
    print(np.around(port_pred[2], decimals=2))
