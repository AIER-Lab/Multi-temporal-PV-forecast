import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.optimizers import Adam, RMSprop
import seaborn as sns
import numpy as np


# section 1: series_to_supervised function-------------------------------------

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


# # section 3: Load the CSV file------------------------------------------------------------

csv_file = "data\dataset_with_emd.csv"
df = pd.read_csv(csv_file, parse_dates=['DateTime'], index_col='DateTime')
# Select specific columns to keep in the DataFrame 'GHI_1','GHI_2','GHI_3','GHI_4','SS1','SS2','SS3','SS4',
#columns_to_keep = ['MinuteOfDay', 'Day', 'Month', 'Tair', 'WS', 'RH', 'GHI', 'P input', 'P']
#df = df[columns_to_keep]
data_opt = {
    'tr_per': 0.75,  # Training percentage
    'val_per': 0.1,  # Validation percentage
    'n_in': 10,  # Original number of input timesteps
    'n_features': 30,  # Number of input features
    'n_out': 1,
    'lag': 4
}

# section 4: Transform the data into a supervised learning dataset-------------
supervised_df, original_indices = series_to_supervised(df, 'P', data_opt['n_in'], data_opt['n_out'], data_opt['lag'],
                                                       True, 'DateTime', False)
train_X, train_y, val_X, val_y, test_X, test_y, scaler_X, scaler_y, train_indices, val_indices, test_indices = prepare_data(
    data_opt, supervised_df, original_indices)


# section 5: Define and train the LSTM model-----------------------------------
def lstm_model(n_in, n_features, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', return_sequences=False, input_shape=(n_in, n_features)))
    model.add(Dropout(0.5))  # Dropout to prevent overfitting
    # model.add(LSTM(112))  # Additional LSTM layer if needed
    model.add(Dense(50, activation='tanh'))  # New dense layer with 14 units and ReLU activation
    model.add(Dense(data_opt['n_out']))  # Output layer
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model


# Train LSTM models for each sub-series
model = lstm_model(data_opt['n_in'], data_opt['n_features'])
history = model.fit(train_X, train_y, epochs=50, batch_size=128, validation_data=(val_X, val_y))

# section 6: Plot training and validation loss---------------------------------

plt.plot(history.history['loss'], label='train - loss')
plt.plot(history.history['val_loss'], label='val - val_loss')
plt.xlabel('Epochs')  # Set the label for the x-axis.
plt.ylabel('Loss')  # Set the label for the y-axis.
plt.legend()  # Display a legend for the lines.
plt.show()  # Display the plot.

# section 7: Make predictions for each model-----------------------------------
yhat = model.predict(test_X)
# Invert scaling for yhat (predicted values)
inv_yhat = scaler_y.inverse_transform(yhat)
# Invert scaling for test_y (actual values)
inv_test_y = scaler_y.inverse_transform(test_y)

# Calculate RMSE
rmse = sqrt(mean_squared_error(inv_test_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
# Calculate NRMSE using range
range_value = np.max(inv_test_y) - np.min(inv_test_y)
nrmse_range = (rmse / range_value) * 100  # Convert to percentage
print('Test NRMSE (Range): %.2f%%' % nrmse_range)
# Calculate NRMSE using mean (if needed for comparison)
mean_value = np.mean(inv_test_y)
nrmse_mean = (rmse / mean_value) * 100  # Convert to percentage
print('Test NRMSE (Mean): %.2f%%' % nrmse_mean)

# section 7: Plot the results---------------------------------------------------

# If it is the case of 15min ahead prediction activate the following part-------
# Convert test_indices to datetime if they are not already
test_indices = pd.to_datetime(test_indices)

# Define DateTime range for plotting
start_date = pd.to_datetime('2023-10-7')  # Example start date, modify as needed
end_date = pd.to_datetime('2023-10-10')  # Example end date, modify as needed
plot_indices = test_indices[(test_indices >= start_date) & (test_indices <= end_date)]

# save results to a csv
results = pd.DataFrame({
    'date' :test_indices,
    'actual': inv_test_y.flatten(),
    'pred': inv_yhat.flatten()})
results.to_csv('results_emd_single_1h.csv')

# Filter the predicted and actual values for the plot range
plot_inv_test_y = inv_test_y[(test_indices >= start_date) & (test_indices <= end_date)]
plot_inv_yhat = inv_yhat[(test_indices >= start_date) & (test_indices <= end_date)]

# Plotting a Single Time Step Ahead
plt.figure(figsize=(12, 6))
plt.plot(plot_indices, plot_inv_test_y[:, 0], label='Actual')
plt.plot(plot_indices, plot_inv_yhat[:, 0], label='Predicted (t+1)')
plt.xlabel('DateTime')
plt.ylabel('Power')
plt.title('Single Time Step Ahead Prediction')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()

# If it is the case of day ahead prediction activate the following part---------

# # Convert test_indices to datetime if they are not already
# test_indices = pd.to_datetime(test_indices)

# # Define DateTime range for plotting
# start_date = pd.to_datetime('2023-10-6')  # Example start date, modify as needed
# end_date = pd.to_datetime('2023-10-9')    # Example end date, modify as needed
# plot_indices = test_indices[(test_indices >= start_date) & (test_indices <= end_date)]

# # Filter the predicted and actual values for the plot range
# plot_inv_test_y = inv_test_y[(test_indices >= start_date) & (test_indices <= end_date)]
# plot_inv_yhat = inv_yhat[(test_indices >= start_date) & (test_indices <= end_date)]

# # Plotting the last column in the datasets
# plt.figure(figsize=(12, 6))
# plt.plot(plot_indices, plot_inv_test_y[:, -1], label='Actual')
# plt.plot(plot_indices, plot_inv_yhat[:, -1], label='Predicted (t+1)')
# plt.xlabel('DateTime')
# plt.ylabel('Power')
# plt.title('Single Time Step Ahead Prediction')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.legend()
# plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
# plt.show()

