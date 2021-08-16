import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime


def get_data(symbol, cols=None, base_dir=os.path.join(os.path.dirname(__file__), "../../data/prices"), date_parser=None):
    if cols is not None and 'Date' not in cols:
        cols.insert(0, 'Date')

    df = pd.read_csv("{}/{}.csv".format(base_dir, symbol), index_col='Date', parse_dates=True, na_values=['nan', 'N/A'],
                     usecols=cols, date_parser=date_parser, header=0)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def get_bond_yields():
    dateparse = lambda x: datetime.strptime(x, '%m/%d/%y')
    df = get_data('bond_yield', cols=['3 mo', '30 yr'], base_dir=os.path.join(os.path.dirname(__file__),
                                                                              "../../data/treasury"), date_parser=dateparse)
    return df

def get_cpi():
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df = get_data('CPIAUCSL', cols=['CPI'], base_dir=os.path.join(os.path.dirname(__file__), "../../data/treasury"),
                  date_parser=dateparse)
    return df

def get_deflation_probabilities():
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    df = get_data('DeflationProb', cols=['DeflationProbability'], base_dir=os.path.join(os.path.dirname(__file__),
                                                                                        "../../data/treasury"),
                  date_parser=dateparse)
    return df

def normalize_to_one(df):
    return df / df.iloc[0]


def std_normalize(df):
    return (df - df.mean()) / df.std()


def normalize_to_spy(df):
    return df.divide(df['SPY'], axis=0)


def get_combined_data(symbols, column, start_date, end_date):
    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    for symbol in symbols:
        data = get_data(symbol, cols=[column])
        data = data.rename(columns={column: symbol})
        df = df.join(data, how='left')

        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])

    return df


def plot_subset(df, start_date, end_date, symbols, y_label="", x_label="", title="", normalized=False):
    subset = df.loc[start_date:end_date, symbols]

    if normalized:
        subset = normalize_to_one(subset)

    ax = subset.plot(title=title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()


def get_sma(series, window=20):
    return series.rolling(window=window).mean()


def get_rolling_std(series, window=20):
    return series.rolling(window=window).std()


def get_bollinger_bands(series, window=20, sigma=2):
    mean = get_sma(series, window)
    std = get_rolling_std(series, window) * sigma
    return mean + std, mean - std


def get_daily_returns(df):
    return (df.iloc[1:]/df.iloc[:-1].values).fillna(0) - 1


def get_cumulative_returns(df):
    return df.iloc[-1]/df.iloc[0].values - 1


def main():
    start_date = '2005-01-01'
    end_date = '2020-12-31'

    df = get_combined_data(['XLF', 'GLD', 'XLK'], 'Adj Close', start_date, end_date)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    dp = get_deflation_probabilities()
    dp.plot()

    plt.show()


if __name__ == '__main__':
    main()
