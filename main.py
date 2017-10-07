import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import argparse


# all functions
def get_data(company_name, stock_name, start_log, end_log, savepath='default.csv'):
    df_data = web.DataReader(company_name, stock_name, start_log, end_log)

    df_data.to_csv(savepath)

def load_data(csv_path='default.csv'):
    if not csv_path:
        print('empty stock data is provided. returning empty data...')
    else:
        df_data = pd.read_csv(csv_path)

    print(df_data.head())
    print(df_data.tail())


# main function
if __name__ == '__main__':
    # parsing
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--C', dest='c', metavar='', default='TSLA', type=str, help='company name')
    parser.add_argument('-s', '--S', dest='s', metavar='', type=str, default='yahoo', help='stock market data source')
    parser.add_argument('-ds', '--DS', dest='ds', metavar='', default='2016-01-01', help='date started (YYYY-MM-DD)')
    parser.add_argument('-df', '--DF', dest='df', metavar='', default='2017-10-01', help='date finished (YYYY-MM-DD')
    p = parser.parse_args()

    # the time span
    start_log = dt.datetime.strptime(p.ds, '%Y-%m-%d')
    end_log = dt.datetime.strptime(p.df, '%Y-%m-%d')

    # verbose
    print('---finance python v.0.1---')
    print('company \t: {}'.format(p.c))
    print('stock \t\t: {}'.format(p.s))
    print('start date\t: {}'.format(p.ds))
    print('finish date\t: {}'.format(p.df))

    # start
    df = get_data(p.c, p.s, p.ds, p.df)

    # load
    df_new = load_data()

    # end
    print('\ndone....')
    print('--------------------------')
