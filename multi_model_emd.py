import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
from keras.optimizers import Adam, RMSprop
from scipy.optimize import minimize
from numpy import sqrt, isnan, isinf
from sklearn.metrics import mean_squared_error


# section 1: series_to_supervised function-------------------------------------
def series_to_supervised(df, out_col, n_in=1, n_out=1, lag=0, dropnan=True,
                         exclude_col=None, include_additional_lags=False):
    n_vars = 1 if type(df) is pd.Series else df.shape[1]
    cols, names = list(), list()
    # Ensure we capture DateTime column if it's explicitly a column, not just in index
    if 'DateTime' in df.columns:
        dt_column = df['DateTime']
    else:
        dt_column = None

    # Input sequence (t-n+1, ... t-1, t)
    for i in range(n_in - 1, -1, -1):
        for j in range(n_vars):
            col_name = df.columns[j]
            if col_name != out_col and col_name != exclude_col:
                cols.append(df.iloc[:, j].shift(i))
                names.append(f'var{j + 1}(t-{i})' if i > 0 else f'var{j + 1}(t)')

    # Additional lags if required
    if include_additional_lags:
        additional_lags = [392, 1680, 20440]  # yesterday, last week, last month, last year
        for additional_lag in additional_lags:
            for i in range(additional_lag, additional_lag - n_in, -1):
                for j in range(n_vars):
                    col_name = df.columns[j]
                    if col_name != out_col and col_name != exclude_col:
                        cols.append(df.iloc[:, j].shift(i))
                        lag_name = 'year' if i == 20440 else ('month' if i == 1680 else 'week')
                        names.append(f'var{j + 1}(t-{i}_{lag_name})')

    # Forecast sequence (t+1, ... t+n)
    for i in range(lag, n_out + lag):
        for j in range(n_vars):
            col_name = df.columns[j]
            if col_name == out_col:
                cols.append(df.iloc[:, j].shift(-i))
                names.append(f'out{j + 1}(t+{i})')

    # Put it all together
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


# CSV PATH
directory = "data"

data_opt = {
    'tr_per': 0.75,  # Training percentage
    'val_per': 0.1,  # Validation percentage
    'n_in': 56 * 3,  # Original number of input timesteps
    'n_features': 32,  # Number of input features
    'n_out': 1,
    'lag': 4
}

# Load the CSV files containing the four sub-series
sub_series_1 = pd.read_csv(os.path.join(directory, "SS1.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_2 = pd.read_csv(os.path.join(directory, "SS2.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_3 = pd.read_csv(os.path.join(directory, "SS3.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_4 = pd.read_csv(os.path.join(directory, "SS4.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_5 = pd.read_csv(os.path.join(directory, "SS5.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_6 = pd.read_csv(os.path.join(directory, "SS6.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_7 = pd.read_csv(os.path.join(directory, "SS7.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_8 = pd.read_csv(os.path.join(directory, "SS8.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_9 = pd.read_csv(os.path.join(directory, "SS9.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_10 = pd.read_csv(os.path.join(directory, "SS10.csv"), parse_dates=['DateTime'], index_col='DateTime')
sub_series_11 = pd.read_csv(os.path.join(directory, "SS11.csv"), parse_dates=['DateTime'], index_col='DateTime')
Actual_data = pd.read_csv(os.path.join(directory, "PV.csv"), parse_dates=['DateTime'], index_col='DateTime')

# Transform data into supervised learning dataset and capture original indices
supervised_df_1, original_indices_1 = series_to_supervised(sub_series_1, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_2, original_indices_2 = series_to_supervised(sub_series_2, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_3, original_indices_3 = series_to_supervised(sub_series_3, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_4, original_indices_4 = series_to_supervised(sub_series_4, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_5, original_indices_5 = series_to_supervised(sub_series_5, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_6, original_indices_6 = series_to_supervised(sub_series_6, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_7, original_indices_7 = series_to_supervised(sub_series_7, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_8, original_indices_8 = series_to_supervised(sub_series_8, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_9, original_indices_9 = series_to_supervised(sub_series_9, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)
supervised_df_10, original_indices_10 = series_to_supervised(sub_series_10, 'P', data_opt['n_in'], data_opt['n_out'],
                                                             data_opt['lag'], True, 'DateTime', True)
supervised_df_11, original_indices_11 = series_to_supervised(sub_series_11, 'P', data_opt['n_in'], data_opt['n_out'],
                                                             data_opt['lag'], True, 'DateTime', True)
supervised_df_r, original_indices_r = series_to_supervised(Actual_data, 'P', data_opt['n_in'], data_opt['n_out'],
                                                           data_opt['lag'], True, 'DateTime', True)

# Prepare data for training with original indices and adjust to unpack the correct number of return values
train_X_1, train_y_1, val_X_1, val_y_1, test_X_1, test_y_1, scaler_X_1, scaler_y_1, train_indices_1, val_indices_1, \
    test_indices_1 = prepare_data(data_opt, supervised_df_1, original_indices_1)
train_X_2, train_y_2, val_X_2, val_y_2, test_X_2, test_y_2, scaler_X_2, scaler_y_2, train_indices_2, val_indices_2, \
    test_indices_2 = prepare_data(data_opt, supervised_df_2, original_indices_2)
train_X_3, train_y_3, val_X_3, val_y_3, test_X_3, test_y_3, scaler_X_3, scaler_y_3, train_indices_3, val_indices_3, \
    test_indices_3 = prepare_data(data_opt, supervised_df_3, original_indices_3)
train_X_4, train_y_4, val_X_4, val_y_4, test_X_4, test_y_4, scaler_X_4, scaler_y_4, train_indices_4, val_indices_4, \
    test_indices_4 = prepare_data(data_opt, supervised_df_4, original_indices_4)
train_X_5, train_y_5, val_X_5, val_y_5, test_X_5, test_y_5, scaler_X_5, scaler_y_5, train_indices_5, val_indices_5, \
    test_indices_5 = prepare_data(data_opt, supervised_df_5, original_indices_5)
train_X_6, train_y_6, val_X_6, val_y_6, test_X_6, test_y_6, scaler_X_6, scaler_y_6, train_indices_6, val_indices_6, \
    test_indices_6 = prepare_data(data_opt, supervised_df_6, original_indices_6)
train_X_7, train_y_7, val_X_7, val_y_7, test_X_7, test_y_7, scaler_X_7, scaler_y_7, train_indices_7, val_indices_7, \
    test_indices_7 = prepare_data(data_opt, supervised_df_7, original_indices_7)
train_X_8, train_y_8, val_X_8, val_y_8, test_X_8, test_y_8, scaler_X_8, scaler_y_8, train_indices_8, val_indices_8, \
    test_indices_8 = prepare_data(data_opt, supervised_df_8, original_indices_8)
train_X_9, train_y_9, val_X_9, val_y_9, test_X_9, test_y_9, scaler_X_9, scaler_y_9, train_indices_9, val_indices_9, \
    test_indices_9 = prepare_data(data_opt, supervised_df_9, original_indices_9)
train_X_10, train_y_10, val_X_10, val_y_10, test_X_10, test_y_10, scaler_X_10, scaler_y_10, train_indices_10, \
    val_indices_10, test_indices_10 = prepare_data(data_opt, supervised_df_10, original_indices_10)
train_X_11, train_y_11, val_X_11, val_y_11, test_X_11, test_y_11, scaler_X_11, scaler_y_11, train_indices_11, \
    val_indices_11, test_indices_11 = prepare_data(data_opt, supervised_df_11, original_indices_11)
train_X_r, train_y_r, val_X_r, val_y_r, test_X_r, test_y_r, scaler_X_r, scaler_y_r, train_indices_r, val_indices_r, \
    test_indices_r = prepare_data(data_opt, supervised_df_r, original_indices_r)


# section 5: Define and train the LSTM model-----------------------------------
def lstm_model(n_in, n_features, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', return_sequences=False, input_shape=(n_in, n_features)))
    # model.add(Dropout(0.0))
    # model.add(LSTM(50))
    # model.add(Dense(30, activation='tanh'))
    model.add(Dense(data_opt['n_out']))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


epochs = 50
# Train LSTM models for each sub-series
model_1 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_1 = model_1.fit(train_X_1, train_y_1, epochs=epochs, batch_size=32, validation_data=(val_X_1, val_y_1))

model_2 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_2 = model_2.fit(train_X_2, train_y_2, epochs=epochs, batch_size=32, validation_data=(val_X_2, val_y_2))

model_3 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_3 = model_3.fit(train_X_3, train_y_3, epochs=epochs, batch_size=32, validation_data=(val_X_3, val_y_3))

model_4 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_4 = model_4.fit(train_X_4, train_y_4, epochs=epochs, batch_size=32, validation_data=(val_X_4, val_y_4))

model_5 = lstm_model(data_opt['n_in'], data_opt['n_features'])

history_5 = model_1.fit(train_X_5, train_y_5, epochs=epochs, batch_size=32, validation_data=(val_X_5, val_y_5))

model_6 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_6 = model_6.fit(train_X_6, train_y_6, epochs=epochs, batch_size=32, validation_data=(val_X_6, val_y_6))

model_7 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_7 = model_7.fit(train_X_7, train_y_7, epochs=epochs, batch_size=32, validation_data=(val_X_7, val_y_7))

model_8 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_8 = model_8.fit(train_X_8, train_y_8, epochs=epochs, batch_size=32, validation_data=(val_X_8, val_y_8))

model_9 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_9 = model_9.fit(train_X_9, train_y_9, epochs=epochs, batch_size=32, validation_data=(val_X_9, val_y_9))

model_10 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_10 = model_10.fit(train_X_10, train_y_10, epochs=epochs, batch_size=32, validation_data=(val_X_10, val_y_10))

model_11 = lstm_model(data_opt['n_in'], data_opt['n_features'])
history_11 = model_11.fit(train_X_11, train_y_11, epochs=epochs, batch_size=32, validation_data=(val_X_11, val_y_11))

# List of history objects and titles
histories = [history_1, history_2, history_3, history_4, history_5, history_6, history_7, history_8, history_9,
             history_10, history_11]
titles = ['Sub-Series 1', 'Sub-Series 2', 'Sub-Series 3', 'Sub-Series 4', 'Sub-Series 5', 'Sub-Series 6',
          'Sub-Series 7', 'Sub-Series 8', 'Sub-Series 9', 'Sub-Series 10', 'Sub-Series 11']

# Make predictions for each model
yhat_1 = model_1.predict(test_X_1)
yhat_2 = model_2.predict(test_X_2)
yhat_3 = model_3.predict(test_X_3)
yhat_4 = model_4.predict(test_X_4)
yhat_5 = model_5.predict(test_X_5)
yhat_6 = model_6.predict(test_X_6)
yhat_7 = model_7.predict(test_X_7)
yhat_8 = model_8.predict(test_X_8)
yhat_9 = model_9.predict(test_X_9)
yhat_10 = model_10.predict(test_X_10)
yhat_11 = model_11.predict(test_X_11)

# Invert scaling for yhat (predicted values)
inv_yhat_1 = scaler_y_1.inverse_transform(yhat_1)
inv_yhat_2 = scaler_y_2.inverse_transform(yhat_2)
inv_yhat_3 = scaler_y_3.inverse_transform(yhat_3)
inv_yhat_4 = scaler_y_4.inverse_transform(yhat_4)
inv_yhat_5 = scaler_y_5.inverse_transform(yhat_5)
inv_yhat_6 = scaler_y_6.inverse_transform(yhat_6)
inv_yhat_7 = scaler_y_7.inverse_transform(yhat_7)
inv_yhat_8 = scaler_y_8.inverse_transform(yhat_8)
inv_yhat_9 = scaler_y_9.inverse_transform(yhat_9)
inv_yhat_10 = scaler_y_10.inverse_transform(yhat_10)
inv_yhat_11 = scaler_y_11.inverse_transform(yhat_11)

# Invert scaling for test_y (actual values)
inv_test_y_1 = scaler_y_1.inverse_transform(test_y_1)
inv_test_y_2 = scaler_y_2.inverse_transform(test_y_2)
inv_test_y_3 = scaler_y_3.inverse_transform(test_y_3)
inv_test_y_4 = scaler_y_4.inverse_transform(test_y_4)
inv_test_y_5 = scaler_y_5.inverse_transform(test_y_5)
inv_test_y_6 = scaler_y_6.inverse_transform(test_y_6)
inv_test_y_7 = scaler_y_7.inverse_transform(test_y_7)
inv_test_y_8 = scaler_y_8.inverse_transform(test_y_8)
inv_test_y_9 = scaler_y_9.inverse_transform(test_y_9)
inv_test_y_10 = scaler_y_10.inverse_transform(test_y_10)
inv_test_y_11 = scaler_y_11.inverse_transform(test_y_11)
inv_test_y_r = scaler_y_r.inverse_transform(test_y_r)

# Lists of actual and predicted data
test_y_values = [inv_test_y_1, inv_test_y_2, inv_test_y_3, inv_test_y_4, inv_test_y_5, inv_test_y_6, inv_test_y_7,
                 inv_test_y_8, inv_test_y_9, inv_test_y_10, inv_test_y_11]
predicted_y_values = [inv_yhat_1, inv_yhat_2, inv_yhat_3, inv_yhat_4, inv_yhat_5, inv_yhat_6, inv_yhat_7, inv_yhat_8,
                      inv_yhat_9, inv_yhat_10, inv_yhat_11]

# Calculate and print RMSE for each set
for i, (test_y, predicted_y) in enumerate(zip(test_y_values, predicted_y_values)):
    rmse = sqrt(mean_squared_error(test_y, predicted_y))
    print(f'Test RMSE for set {i + 1}: {rmse:.3f}')

weights = [ 4.2071162,  -0.27886751, -0.72538315,  0.70591745,  0.16879582, -0.6070801,
 -2.0075128,  -1.7296125,   2.12595223, -1.24848525,  0.03396689]
#optimal 15 min 2.64850176, -1.72261054, -1.6885003, 2.99814526, 0.99191934, -1.0066344, -1.51027201, 1.59738008, 2.47617303, 0.11018393, -0.55571876
# Weighted sum of predicted outputs from all four LSTM models
final_prediction_weighted = (inv_yhat_1 * weights[0] +
                             inv_yhat_2 * weights[1] +
                             inv_yhat_3 * weights[2] +
                             inv_yhat_4 * weights[3] +
                             inv_yhat_5 * weights[4] +
                             inv_yhat_6 * weights[5] +
                             inv_yhat_7 * weights[6] +
                             inv_yhat_8 * weights[7] +
                             inv_yhat_9 * weights[8] +
                             inv_yhat_10 * weights[9] +
                             inv_yhat_11 * weights[10]
                             )

# Calculate RMSE for the final prediction
rmse_final = sqrt(mean_squared_error(inv_test_y_r, final_prediction_weighted))
print('Test RMSE for final prediction: %.3f' % rmse_final)
# Calculate NRMSE using range
range_value = np.max(inv_test_y_r) - np.min(inv_test_y_r)
nrmse_range = (rmse_final / range_value) * 100
print('Test NRMSE (Range): %.2f%%' % nrmse_range)

# Convert test_indices to datetime if they are not already
test_indices = pd.to_datetime(test_indices_r, dayfirst=True)

# Define DateTime range for plotting
start_date = pd.to_datetime('2023-10-7')  # Example start date, modify as needed
end_date = pd.to_datetime('2023-10-11')  # Example end date, modify as needed

# Filter the predicted and actual values for the plot range
plot_indices = test_indices[(test_indices >= start_date) & (test_indices <= end_date)]
plot_inv_test_y = inv_test_y_r[(test_indices >= start_date) & (test_indices <= end_date)]
plot_inv_yhat = final_prediction_weighted[(test_indices >= start_date) & (test_indices <= end_date)]

results = pd.DataFrame({
    'actual': plot_inv_test_y.flatten(),
    'pred': plot_inv_yhat.flatten()})
results.to_csv('results_emd_multi_1h.csv')

# Plotting the last column in the datasets
plt.figure(figsize=(12, 6))
plt.plot(plot_indices, plot_inv_test_y[:, -1], label='Actual')
plt.plot(plot_indices, plot_inv_yhat[:, -1], label='Predicted (t+1)')
plt.xlabel('DateTime')
plt.ylabel('Power')
plt.title('Single Time Step Ahead Prediction')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()


def calculate_rmse(weights, inv_yhat_1, inv_yhat_2, inv_yhat_3, inv_yhat_4, inv_yhat_5, inv_yhat_6, inv_yhat_7,
                   inv_yhat_8, inv_yhat_9, inv_yhat_10, inv_yhat_11, inv_test_y_r):
    final_prediction = (inv_yhat_1 * weights[0] +
                        inv_yhat_2 * weights[1] +
                        inv_yhat_3 * weights[2] +
                        inv_yhat_4 * weights[3] +
                        inv_yhat_5 * weights[4] +
                        inv_yhat_6 * weights[5] +
                        inv_yhat_7 * weights[6] +
                        inv_yhat_8 * weights[7] +
                        inv_yhat_9 * weights[8] +
                        inv_yhat_10 * weights[9] +
                        inv_yhat_11 * weights[10]
                        )
    rmse = sqrt(mean_squared_error(inv_test_y_r, final_prediction))
    if isnan(rmse) or isinf(rmse):
        return float('inf')  # handle numerical issues
    return rmse


# Trying different initial weights
initial_weights = [1, 55, 21, 10, 1, 55, 21, 10, 1, 55, 21]  # Adjust as needed to test sensitivity

# Using a different optimization method
#result = minimize(calculate_rmse, initial_weights,
 #                 args=(inv_yhat_1, inv_yhat_2, inv_yhat_3, inv_yhat_4, inv_yhat_5, inv_yhat_6, inv_yhat_7,
  #                      inv_yhat_8, inv_yhat_9, inv_yhat_10, inv_yhat_11, inv_test_y_r),
   #               method='Nelder-Mead')  # Using a derivative-free method

#optimized_weights = result.x
#print("Optimized weights:", optimized_weights)
#print("Reduced RMSE:", result.fun)
