import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web


if __name__ == '__main__':
    df = pd.read_csv('AMD.csv', parse_dates=True, index_col=0)
    print(df.head())
    print('-----')
    print(df[['Open', 'High', 'Low', 'Close']].head())

    # plot
    df['High'].plot()
    df['Low'].plot()
    plt.show(20)
