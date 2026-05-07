import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def evaluate_and_plot(y_true, y_pred, title="Prediction", horizon="last"):
    """
    horizon:
        - "all"  → plot completo (multi-step)
        - "first" → t+1
        - "last"  → t+H (ultimo orizzonte)
    """

    # ======================================================
    # 1. SELEZIONE ORIZZONTE
    # ======================================================

    if len(y_true.shape) == 2:  # multi-step case

        if horizon == "first":
            y_true_plot = y_true[:, 0]
            y_pred_plot = y_pred[:, 0]

        elif horizon == "last":
            y_true_plot = y_true[:, -1]
            y_pred_plot = y_pred[:, -1]

        else:  # "all"
            y_true_plot = y_true
            y_pred_plot = y_pred

    else:
        # single-step case
        y_true_plot = y_true
        y_pred_plot = y_pred

    # ======================================================
    # 2. METRICA
    # ======================================================

    rmse = sqrt(mean_squared_error(y_true_plot, y_pred_plot))
    print(f"RMSE: {rmse:.3f}")

    # ======================================================
    # 3. PLOT
    # ======================================================

    plt.figure(figsize=(12, 6))

    if horizon == "all" and len(y_true.shape) == 2:
        # heatmap-style plot (multi-step completo)
        plt.plot(y_true[:, 0], label="Actual t+1")
        plt.plot(y_pred[:, 0], label="Pred t+1")
        plt.plot(y_true[:, -1], label="Actual t+H", linestyle="--")
        plt.plot(y_pred[:, -1], label="Pred t+H", linestyle="--")

    else:
        plt.plot(y_true_plot, label="Actual")
        plt.plot(y_pred_plot, label="Predicted")

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()



def series_to_supervised(df, out_col, n_in=1, n_out=1, lag=1, dropnan=True,
                         exclude_col=None, include_additional_lags=False):
    n_vars = 1 if type(df) is pd.Series else df.shape[1]
    cols, names = list(), list()
    # Ensure we capture DateTime column if it's explicitly a column, not just in index
    if 'DateTime' in df.columns:
        dt_column = df['DateTime']
    else:
        dt_column = None

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        for j in range(n_vars):
            col_name = df.columns[j]
            if col_name != out_col and col_name != exclude_col:
                cols.append(df.iloc[:, j].shift(i))
                names.append(f'var{j + 1}(t-{i})')

    # Additional lags if required
    if include_additional_lags:
        additional_lags = [56, 392, 1680, 20440]  # yesterday, last week, last month, last year
        for additional_lag in additional_lags:
            for i in range(additional_lag, additional_lag - n_in, -1):
                for j in range(n_vars):
                    col_name = df.columns[j]
                    if col_name != out_col and col_name != exclude_col:
                        cols.append(df.iloc[:, j].shift(i))
                        lag_name = 'year' if i == 20440 else ('month' if i == 1680 else ('week' if i == 392 else 'day'))
                        names.append(f'var{j + 1}(t-{i}_{lag_name})')

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(lag, n_out + lag):
        for j in range(n_vars):
            col_name = df.columns[j]
            if col_name == out_col:
                cols.append(df.iloc[:, j].shift(-i))
                names.append(f'out{j + 1}(t+{i})')

    # Put it all together)
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dt_column is not None:
        agg['DateTime'] = dt_column  # Append DateTime back to the DataFrame

    # Drop rows with NaN values and adjust DateTime accordingly
    if dropnan:
        valid_indices = agg.dropna().index
        agg.dropna(inplace=True)
        if dt_column is not None:
            # Properly filter DateTime by using loc which aligns by index labels
            agg['DateTime'] = dt_column.loc[valid_indices]

        return agg, valid_indices
    return agg, None


# section 2: split the dataset into train/validation/test----------------------

def prepare_data(data_opt, reframed, original_indices):
    # Extract raw values
    input_col = [col for col in reframed.columns if 'var' in col]
    output_col = [col for col in reframed.columns if 'out' in col]

    # Adjust train-validation-test split to accommodate removed NaNs
    n_train_hours = int(
        data_opt['tr_per'] * len(original_indices))  # Adjusted based on original indices post NaN removal
    n_val_hours = int(data_opt['val_per'] * len(original_indices))  # Validation split
    train_indices = original_indices[:n_train_hours]
    val_indices = original_indices[n_train_hours:n_train_hours + n_val_hours]
    test_indices = original_indices[n_train_hours + n_val_hours:]

    train = reframed.loc[train_indices, :]
    val = reframed.loc[val_indices, :]
    test = reframed.loc[test_indices, :]

    # Split into inputs and outputs
    train_X, train_y = train.loc[:, input_col].values, train.loc[:, output_col].values
    val_X, val_y = val.loc[:, input_col].values, val.loc[:, output_col].values
    test_X, test_y = test.loc[:, input_col].values, test.loc[:, output_col].values

    # Normalize features
    feature_range = (0, 1)
    scaler_X = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)

    train_X = scaler_X.fit_transform(train_X)
    val_X = scaler_X.transform(val_X)
    test_X = scaler_X.transform(test_X)

    train_y = scaler_y.fit_transform(train_y)
    val_y = scaler_y.transform(val_y)
    test_y = scaler_y.transform(test_y)

    # Reshape input to be 3D [samples, timesteps, features]
    n_features = len(input_col) // data_opt['n_in']  # Calculate number of features based on the columns considered
    train_X = train_X.reshape((train.shape[0], data_opt['n_in'], n_features))
    val_X = val_X.reshape((val.shape[0], data_opt['n_in'], n_features))
    test_X = test_X.reshape((test.shape[0], data_opt['n_in'], n_features))

    return train_X, train_y, val_X, val_y, test_X, test_y, scaler_X, scaler_y, train_indices, val_indices, test_indices