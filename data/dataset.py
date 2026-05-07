import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from utils.assets import series_to_supervised,prepare_data



import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class SeqDataset:

    def __init__(self, file_path, config):

        self.df = pd.read_csv(
            file_path,
            parse_dates=['DateTime'],
            index_col='DateTime'
        )

        self.config = config

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()


    # =====================================
    # CREATE SEQUENCES
    # =====================================
    def create_sequences(self, df, target_col):

        X, y = [], []

        n_in = self.config["n_in"]
        n_out = self.config["n_out"]
        lag = self.config["lag"]

        for i in range(len(df) - n_in - n_out - lag + 1):

            # INPUT WINDOW
            seq_x = df.iloc[i:i+n_in].drop(columns=[target_col]).values

            # TARGET WINDOW
            seq_y = df.iloc[
                i+n_in+lag-1 : i+n_in+lag-1+n_out
            ][target_col].values

            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        y = np.array(y)

        # single-step compatibility
        if n_out == 1:
            y = y.reshape(-1, 1)

        return X, y


    # =====================================
    # SPLIT DATA
    # =====================================
    def split_data(self, X, y):

        n = len(X)

        train_end = int(n * self.config['tr_per'])

        val_end = int(
            n * (
                self.config['tr_per'] +
                self.config['val_per']
            )
        )

        train_X, train_y = X[:train_end], y[:train_end]

        val_X, val_y = (
            X[train_end:val_end],
            y[train_end:val_end]
        )

        test_X, test_y = (
            X[val_end:],
            y[val_end:]
        )

        return (
            train_X, train_y,
            val_X, val_y,
            test_X, test_y
        )


    # =====================================
    # NORMALIZATION
    # =====================================
    def normalize_df(
        self,
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y
    ):

        # flatten for scaler
        train_X_shape = train_X.shape
        val_X_shape = val_X.shape
        test_X_shape = test_X.shape

        train_X = train_X.reshape(train_X.shape[0], -1)
        val_X = val_X.reshape(val_X.shape[0], -1)
        test_X = test_X.reshape(test_X.shape[0], -1)

        # FIT ONLY ON TRAIN
        train_X = self.scaler_X.fit_transform(train_X)

        val_X = self.scaler_X.transform(val_X)
        test_X = self.scaler_X.transform(test_X)

        train_y = self.scaler_y.fit_transform(train_y)

        val_y = self.scaler_y.transform(val_y)
        test_y = self.scaler_y.transform(test_y)

        # reshape back
        train_X = train_X.reshape(train_X_shape)
        val_X = val_X.reshape(val_X_shape)
        test_X = test_X.reshape(test_X_shape)

        return (
            train_X, train_y,
            val_X, val_y,
            test_X, test_y
        )


    # =====================================
    # FULL PIPELINE
    # =====================================
    def prepare(self):

        # CREATE SEQUENCES
        X, y = self.create_sequences(
            self.df,
            target_col='P'
        )

        # SPLIT
        (
            train_X, train_y,
            val_X, val_y,
            test_X, test_y
        ) = self.split_data(X, y)

        # NORMALIZE
        (
            train_X, train_y,
            val_X, val_y,
            test_X, test_y
        ) = self.normalize_df(
            train_X, train_y,
            val_X, val_y,
            test_X, test_y
        )

        return (
            train_X, train_y,
            val_X, val_y,
            test_X, test_y,
            self.scaler_X,
            self.scaler_y
        )