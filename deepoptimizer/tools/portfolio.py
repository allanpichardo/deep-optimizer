import math

from deepoptimizer.tools.stock_utils import get_combined_data, get_daily_returns, normalize_to_one, get_bond_yields, get_cpi, \
    std_normalize
import pandas as pd
import tensorflow as tf


class EconomicData:
    def __init__(self, start_date='2010-01-01', end_date='2021-01-01'):
        dates = pd.date_range(start_date, end_date)
        df = pd.DataFrame(index=dates)

        bond_yield = get_bond_yields()
        cpi = get_cpi()

        df = df.join(bond_yield, how='left')
        df = df.join(cpi, how='left')

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        self.economic_data = df

    def create_dataset(self, step=12, size=12, skip_y=False):
        """
        Return a Tensorflow dataset from this portfolio
        :param step: difference in days between X prices and Y prices (offset)
        :param size: length of window
        :return:
        """

        n_examples = len(self.economic_data)
        indicator_collection = []
        k = 0
        while k * step + size < n_examples:
            indicator_collection += [normalize_to_one(self.economic_data.iloc[k * step:k * step + size]).values]
            k += 1

        if skip_y:
            return tf.data.Dataset.from_tensor_slices((indicator_collection, indicator_collection))

        dfa_x = indicator_collection[:-1]
        dfa_y = indicator_collection[1:]

        return tf.data.Dataset.from_tensor_slices((dfa_x, dfa_y))


class Portfolio:

    def __init__(self, tickers=None, start_date='2011-01-01', end_date='2021-01-01'):
        if tickers is None:
            tickers = ['XLF', 'XLK', 'FEZ', 'GXC']

        df = get_combined_data(tickers, 'Adj Close', start_date, end_date)
        self.spy = df['SPY']
        df.drop(columns=['SPY'], inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        self.start_date = start_date
        self.end_date = end_date
        self.prices_raw = df
        self.prices_normalized = normalize_to_one(df)
        self.daily_returns = get_daily_returns(df)[1:]

        dates = pd.date_range(start_date, end_date)
        dfb = pd.DataFrame(index=dates)

        bond_yield = get_bond_yields()
        cpi = get_cpi()

        dfb = dfb.join(bond_yield, how='left')
        dfb = dfb.join(cpi, how='left')
        dfb = dfb.join(self.spy)
        dfb = dfb.dropna(subset=['SPY'])
        # dfb.drop(columns=['SPY'], inplace=True)

        dfb.fillna(method='ffill', inplace=True)
        dfb.fillna(method='bfill', inplace=True)

        self.economic_data = dfb

    def get_portfolio_values(self, allocations, start_value=1000):
        alloced = self.prices_normalized * allocations
        pos_vals = alloced * start_value
        return pos_vals.sum(axis=1)

    def get_cumulative_return(self, allocation, start_value=1000):
        portval = self.get_portfolio_values(allocation, start_value=start_value)
        return (portval[-1] / portval[0]) - 1

    def get_average_daily_return(self):
        return self.daily_returns.mean()

    def get_daily_std_dev(self):
        return self.daily_returns.std()

    def get_sharpe_ratio(self, allocations, start_value=1000, risk_free_rate=0.0005):
        daily_rfr = ((1.0 + risk_free_rate) ** (1 / float(252))) - 1
        frequency = math.sqrt(252)
        port_rets = get_daily_returns(self.get_portfolio_values(allocations, start_value=start_value))
        sharpe = ((port_rets - daily_rfr).mean() / port_rets.std())
        return (frequency ** 0.5) * sharpe

    def create_dataset(self, step=252, size=252, skip_y=False):
        """
        Return a Tensorflow dataset from this portfolio
        :param step: difference in days between X prices and Y prices (offset)
        :param size: length of window
        :return:
        """

        n_examples_prices = len(self.prices_raw)
        price_collection = []
        k = 0
        while k * step + size < n_examples_prices:
            price_collection += [normalize_to_one(self.prices_raw.iloc[k * step:k * step + size]).values]
            k += 1

        n_examples_indic = len(self.economic_data)
        indicator_collection = []
        k = 0
        while k * step + size < n_examples_indic:
            indicator_collection += [std_normalize(self.economic_data.iloc[k * step:k * step + size]).values]
            k += 1

        if skip_y:
            return tf.data.Dataset.from_tensor_slices(({"price_input": price_collection, "indicators_input": indicator_collection}, price_collection))

        df_x = price_collection[:-1]
        df_y = price_collection[1:]
        dfa_x = indicator_collection[:-1]
        dfa_y = indicator_collection[1:]

        return tf.data.Dataset.from_tensor_slices(({"price_input": df_x, "indicators_input": dfa_x}, df_y))


if __name__ == '__main__':
    ec = Portfolio()
    dataset = ec.create_dataset()

    for x, y in dataset.batch(5).take(1):
        print(x)
        print(y)
