import tensorflow as tf
from deepoptimizer.tools.portfolio import Portfolio
from deepoptimizer.losses import sharpe_ratio_loss, volatility_loss, portfolio_return_loss
import numpy as np
import argparse


def sharpe(p, w):
    return -sharpe_ratio_loss(p, w)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize Portfolio.')
    parser.add_argument('epochs', metavar='N', type=int, nargs='+',
                        help='Number of epochs of training', default=100)

    args = parser.parse_args()

    tf.random.set_seed(5)

    tickers = ["FBALX", "FBSOX"]
    number_of_assets = 2

    portfolio = Portfolio(tickers=tickers, start_date='2010-01-01', end_date='2020-01-01')
    dataset = portfolio.create_dataset(step=1).shuffle(50000).batch(32).cache().prefetch(tf.data.experimental.AUTOTUNE)

    input_prices = tf.keras.layers.Input((252, number_of_assets), name="price_input")
    input_indicators = tf.keras.layers.Input((252, 3), name="indicators_input")
    p = tf.keras.layers.LSTM(126, activation='tanh', return_sequences=True)(input_prices)
    p = tf.keras.layers.BatchNormalization()(p)
    # p = tf.keras.layers.AveragePooling1D()(p)
    p = tf.keras.layers.LSTM(126, activation='tanh')(p)
    p = tf.keras.layers.BatchNormalization()(p)
    # p = tf.keras.layers.AveragePooling1D()(p)
    # p = tf.keras.layers.Conv1D(8, 3, activation='relu')(p)
    # p = tf.keras.layers.BatchNormalization()(p)
    # p = tf.keras.layers.AveragePooling1D()(p)
    # p = tf.keras.layers.Conv1D(8, 3, activation='relu')(p)
    # p = tf.keras.layers.BatchNormalization()(p)
    # p = tf.keras.layers.AveragePooling1D()(p)
    # p = tf.keras.layers.Flatten()(p)

    i = tf.keras.layers.Conv1D(126, 3, activation='tanh')(input_indicators)
    i = tf.keras.layers.BatchNormalization()(i)
    i = tf.keras.layers.MaxPooling1D()(i)
    i = tf.keras.layers.Conv1D(126, 3, activation='tanh')(i)
    i = tf.keras.layers.BatchNormalization()(i)
    # i = tf.keras.layers.AveragePooling1D()(i)
    # i = tf.keras.layers.Conv1D(8, 3, activation='tanh')(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    # i = tf.keras.layers.AveragePooling1D()(i)
    # i = tf.keras.layers.Conv1D(8, 3, activation='tanh')(i)
    # i = tf.keras.layers.BatchNormalization()(i)
    # i = tf.keras.layers.AveragePooling1D()(i)
    # i = tf.keras.layers.Flatten()(i)
    i = tf.keras.layers.GlobalMaxPool1D()(i)

    x = tf.keras.layers.Concatenate()([p, i])
    x = tf.keras.layers.Dense(64, activation='tanh')(x)
    allocations = tf.keras.layers.Dense(number_of_assets, activation='softmax', name="allocations")(x)
    returns = tf.keras.layers.Dense(1, activation='linear', name="returns")(x)
    volatility = tf.keras.layers.Dense(1, activation='linear', name="volatility")(x)

    model = tf.keras.Model(inputs=[input_prices, input_indicators], outputs=[allocations, returns, volatility])

    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.00001, clipnorm=1.0), loss=[sharpe_ratio_loss, portfolio_return_loss, volatility_loss], metrics=[])
    model.fit(dataset, epochs=args.epochs, verbose=1)

    pred = Portfolio(tickers=tickers, start_date='2020-01-01', end_date='2021-01-01').create_dataset(skip_y=True).batch(1)
    port_pred = model.predict(pred.take(1), verbose=1)
    print(np.around(port_pred[0], decimals=2))
