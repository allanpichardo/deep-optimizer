import tensorflow as tf
import tensorflow.keras
from deepoptimizer.tools.portfolio import Portfolio
from deepoptimizer.losses import sharpe_ratio_loss, volatility_loss, portfolio_return_loss, sortino_ratio_loss, __downside_risk
import numpy as np
import argparse


def sharpe(p, w):
    return -sharpe_ratio_loss(p, w)

def deep_conv(input, number_layers=10, channels=8):
    x = input
    for i in range(1, number_layers):
        x = tf.keras.layers.Conv1D(i * 2, 3, padding='same', activation='elu')(x)
    x = tf.keras.layers.Flatten()(x)
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Portfolio.')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_start_date', default='2014-01-01')
    parser.add_argument('--train_end_date', default='2020-01-01')
    parser.add_argument('--tickers', nargs="+", default=["FIPDX", "FOCPX", "FBGRX"])
    parser.add_argument('--window_size', type=int, default=252)
    parser.add_argument('--mode', type=str, default='predict')

    args = parser.parse_args()

    tf.random.set_seed(5)

    tickers = args.tickers.copy()
    number_of_assets = len(tickers)
    window_size = args.window_size
    skip_y = args.mode == 'optimize'

    if skip_y:
        print('Running in optimization mode')
    else:
        print('Running in forecast mode')

    portfolio = Portfolio(tickers=tickers, start_date=args.train_start_date, end_date=args.train_end_date)
    dataset = portfolio.create_dataset(step=1, size=window_size, skip_y=skip_y).shuffle(50000).batch(args.batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)

    input_prices = tf.keras.layers.Input((window_size, number_of_assets), name="price_input")
    input_indicators = tf.keras.layers.Input((window_size, 7), name="indicators_input")
    p = tf.keras.layers.LSTM(256, return_sequences=True)(input_prices)
    # p = tf.keras.layers.Flatten()(p)

    i = tf.keras.layers.LSTM(256, return_sequences=True)(input_indicators)
    # i = tf.keras.layers.Flatten()(i)

    x = tf.keras.layers.Concatenate()([p, i])
    x = tf.keras.layers.Conv1D(256, 3, padding='same', activation='elu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    allocations = tf.keras.layers.Dense(number_of_assets, activation='softmax', name="allocations")(x)
    volatility = tf.keras.layers.Dense(number_of_assets, activation='softmax', name="volatility")(x)
    returns = tf.keras.layers.Dense(number_of_assets, activation='softmax', name="returns")(x)

    model = tf.keras.Model(inputs=[input_prices, input_indicators], outputs=[allocations, volatility, returns])

    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=[sharpe_ratio_loss, volatility_loss, portfolio_return_loss],
        metrics=[],
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="loss")]
    )

    print("----Training Start----")
    print("Tickers:\n{}".format(args.tickers))
    print("--------")
    model.fit(dataset, epochs=args.epochs, verbose=1)

    print("----Optimization Start----")
    pred = Portfolio(tickers=tickers, start_date='2020-01-01', end_date='2021-01-01').create_dataset(skip_y=True, step=window_size, size=window_size).batch(
        1)
    port_pred = model.predict(pred.take(1), verbose=1)
    print("Allocations:\n{}".format(args.tickers))
    print(np.around(port_pred[0], decimals=2))
    print("Volatility:")
    print(np.around(port_pred[1], decimals=2))
    print("Returns:")
    print(np.around(port_pred[2], decimals=2))
