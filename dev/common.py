import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def normalize_data(df, labels):

    # copy the output
    df_out = df.copy()

    scaler = preprocessing.MinMaxScaler()

    for label in labels:
        df_out[label] = scaler.fit_transform(df[label].values.reshape(-1, 1))

    return df_out


def load_data(normed_stock, seq_len, size_train=80, size_valid=10, size_test=10):
    # convert to numpy array
    data_raw = normed_stock.as_matrix()
    data = []

    # create all possible sequenes of length 
    for idx in range(len(data_raw) - seq_len):
        data.append(data_raw[idx:idx+seq_len])

    # convert to numpy
    data = np.array(data)

    # split
    test_set_size = int(np.round(size_test/100 * data.shape[0]))
    valid_set_size = int(np.round(size_valid/100 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size + valid_set_size)

    # train
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    # validate
    x_valid = data[train_set_size:train_set_size+valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size+valid_set_size, -1, :]

    # test
    x_test = data[train_set_size+valid_set_size:, :-1, :]
    y_test = data[train_set_size+valid_set_size:, -1, :]

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def plot_stock_vs_volume(fh=None, df=None, stock_name=None):

    if fh == None:
        fh = plt.figure(figsize=(15, 5))

    ax1 = fh.add_subplot(1, 2, 1)
    ax2 = fh.add_subplot(1, 2, 2)

    # stock price
    ax1.plot(df[df.symbol == stock_name].open.values, color='red', label='open')
    ax1.plot(df[df.symbol == stock_name].close.values, color='green', label='close')
    ax1.plot(df[df.symbol == stock_name].low.values, color='blue', label='low')
    ax1.plot(df[df.symbol == stock_name].high.values, color='black', label='high')
    ax1.set_title('stock price')
    ax1.set_xlabel('time[days]')
    ax1.set_ylabel('price')
    ax1.legend(loc='best')

    # volume
    ax2.plot(df[df.symbol == stock_name].volume.values, color='black', label='volume')
    ax2.set_title('stock volume')
    ax2.set_xlabel('time [days]')
    ax2.set_ylabel('volume')
    ax2.legend(loc='best')

    return fh