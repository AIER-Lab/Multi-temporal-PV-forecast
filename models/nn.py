from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

def build_lstm_model(n_in, n_features, n_out=1, learning_rate=0.001):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=(n_in, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='tanh'))
    model.add(Dense(n_out))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model