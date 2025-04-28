import keras
import tensorflow as tf
import numpy as np
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Flatten, GRU, TimeDistributed, RepeatVector, Layer, Input
from keras import callbacks
from keras.layers import Layer
from keras.src.saving import register_keras_serializable
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
# from numpy.lib.financial import rate

# from keras import optimizers
# from keras_tuner import RandomSearch, BayesianOptimization pip install


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Alignment scores. Pass them through tanh function
        e = tf.tanh(tf.linalg.matmul(x, self.W) + self.b)
        # Remove dimension of size 1
        e = tf.squeeze(e, axis=-1)
        # Compute the weights
        alpha = tf.nn.softmax(e)
        # Reshape to TensorFlow format
        alpha = tf.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = tf.reduce_sum(context, axis=1)
        return context


def create_model_attention(model_opt, train_X, train_y):
    input_train = Input(shape=(train_X.shape[1], train_X.shape[2]))
    output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

    # Encoder
    encoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder(input_train)

    for _ in range(model_opt["LSTM_layers"] - 1):  # Add extra LSTM layers
        encoder_output, state_h, state_c = LSTM(model_opt["LSTM_num_hidden_units"],
                                                return_sequences=True, return_state=True)(encoder_output)

    # Attention Layer
    attention_layer = AttentionLayer()(encoder_output)

    # Decoder
    decoder = RepeatVector(output_train.shape[1])(attention_layer)
    decoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True)(
        decoder, initial_state=[state_h, state_c]
    )

    for _ in range(model_opt["LSTM_layers"] - 1):  # Additional decoder LSTM layers
        decoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True)(decoder)

    outputs = TimeDistributed(Dense(output_train.shape[2]))(decoder)

    # Compile model
    model = Model(inputs=input_train, outputs=outputs)
    model.compile(loss=model_opt["metrics"], optimizer=model_opt["optimizer"])

    # Callbacks
    erlstp_callback = callbacks.EarlyStopping(monitor="val_loss", patience=model_opt["patience"],
                                              mode="min", restore_best_weights=True, verbose=1)

    ckpt_callback = callbacks.ModelCheckpoint(model_opt["model_path"] + 'model.keras',
                                              save_best_only=True, save_weights_only=False,
                                              monitor='loss', mode='min')

    nan_callback = callbacks.TerminateOnNaN()

    cb_list = [erlstp_callback, nan_callback, ckpt_callback]

    model.summary()

    history = model.fit(train_X, train_y, epochs=model_opt['epochs'], callbacks=cb_list,
                        validation_split=model_opt['validation_split'])

    return model, history


def pinball_loss(q):
    """Pinball loss function for quantile regression"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        return K.mean(K.maximum(q * error, (q - 1) * error))
    return loss

def pinball_loss_reg(quantile, lambda_reg=0.01):
    """Regularized Pinball Loss to reduce excessive spread"""
    def loss(y_true, y_pred):
        error = y_true - y_pred
        pinball = K.maximum(quantile * error, (quantile - 1) * error)
        regularization = lambda_reg * K.mean(K.abs(y_pred))  # Shrinks wide intervals
        return K.mean(pinball) + regularization
    return loss

def create_model_attention_quantile(model_opt, train_X, train_y, quantile):
    input_train = Input(shape=(train_X.shape[1], train_X.shape[2]))
    output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

    # Encoder
    encoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True, return_state=True,
                   kernel_regularizer=tf.keras.regularizers.l2(0.01))  # L2 Regularization
    encoder_output, state_h, state_c = encoder(input_train)
    encoder_output = Dropout(0.2)(encoder_output)  # Dropout to reduce variance

    for _ in range(model_opt["LSTM_layers"] - 1):
        encoder_output, state_h, state_c = LSTM(model_opt["LSTM_num_hidden_units"],
                                                return_sequences=True, return_state=True,
                                                kernel_regularizer=tf.keras.regularizers.l2(0.01))(encoder_output)
        encoder_output = Dropout(0.2)(encoder_output)

    # Attention Layer
    attention_layer = AttentionLayer()(encoder_output)

    # Decoder
    decoder = RepeatVector(output_train.shape[1])(attention_layer)
    decoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True,
                   kernel_regularizer=l2(0.01))(decoder, initial_state=[state_h, state_c])
    decoder = Dropout(0.2)(decoder)

    for _ in range(model_opt["LSTM_layers"] - 1):
        decoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True,
                       kernel_regularizer=l2(0.01))(decoder)
        decoder = Dropout(0.2)(decoder)

    outputs = TimeDistributed(Dense(output_train.shape[2]))(decoder)

    # Compile model
    model = Model(inputs=input_train, outputs=outputs)
    model.compile(loss=pinball_loss_reg(quantile), optimizer=model_opt["optimizer"])

    # Callbacks
    erlstp_callback = callbacks.EarlyStopping(monitor="val_loss", patience=model_opt["patience"],
                                              mode="min", restore_best_weights=True, verbose=1)

    ckpt_callback = callbacks.ModelCheckpoint(model_opt["model_path"] + 'model.keras',
                                              save_best_only=True, save_weights_only=False,
                                              monitor='loss', mode='min')

    nan_callback = callbacks.TerminateOnNaN()

    cb_list = [erlstp_callback, nan_callback, ckpt_callback]

    model.summary()

    history = model.fit(train_X, train_y, epochs=model_opt['epochs'], callbacks=cb_list,
                        validation_split=model_opt['validation_split'])

    return model, history

@tf.keras.utils.register_keras_serializable()
def gaussian_nll(y_true, y_pred):
    y_true = tf.expand_dims(y_true, axis=-1)
    mean = y_pred[..., 0]  # First output (mean)
    std = tf.nn.softplus(y_pred[..., 1]) + 1e-6  # Second output (std deviation, ensuring positivity)

    log_likelihood = 0.5 * tf.math.log(2 * np.pi) + tf.math.log(std) + (y_true - mean) ** 2 / (2 * std ** 2)

    return tf.reduce_mean(log_likelihood)

def create_model_attention_gaussian(model_opt, train_X, train_y):
    input_train = Input(shape=(train_X.shape[1], train_X.shape[2]))
    output_train = Input(shape=(train_y.shape[1], train_y.shape[2]))

    # Encoder
    encoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True, return_state=True)
    encoder_output, state_h, state_c = encoder(input_train)

    for _ in range(model_opt["LSTM_layers"] - 1):  # Add extra LSTM layers
        encoder_output, state_h, state_c = LSTM(model_opt["LSTM_num_hidden_units"],
                                                return_sequences=True, return_state=True)(encoder_output)

    # Attention Layer
    attention_layer = AttentionLayer()(encoder_output)

    # Decoder
    decoder = RepeatVector(output_train.shape[1])(attention_layer)
    decoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True)(decoder, initial_state=[state_h, state_c])

    for _ in range(model_opt["LSTM_layers"] - 1):  # Additional decoder LSTM layers
        decoder = LSTM(model_opt["LSTM_num_hidden_units"], return_sequences=True)(decoder)

    # Two output layers: one for the mean and one for the standard deviation
    mean_output = TimeDistributed(Dense(1))(decoder)  # Mean (mu)
    std_output = TimeDistributed(Dense(1, activation='softplus'))(decoder)  # Standard deviation (sigma, softplus to ensure positivity)


    # Concatenate mean and std so each time step has two values
    output = tf.keras.layers.Concatenate(name="output")([mean_output, std_output])

    model = Model(inputs=input_train, outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss=gaussian_nll)


    # Callbacks
    erlstp_callback = callbacks.EarlyStopping(monitor="val_loss", patience=model_opt["patience"],
                                              mode="min", restore_best_weights=True, verbose=1)

    ckpt_callback = callbacks.ModelCheckpoint(model_opt["model_path"] + 'model.keras',
                                              save_best_only=True, save_weights_only=False,
                                              monitor='loss', mode='min')

    nan_callback = callbacks.TerminateOnNaN()

    cb_list = [erlstp_callback, nan_callback, ckpt_callback]

    # Train the model
    history = model.fit(train_X, train_y[..., np.newaxis], epochs=model_opt['epochs'], callbacks=cb_list,
                        validation_split=model_opt['validation_split'])

    return model, history


def create_model(model_opt, X, y):

    # tf.debugging.set_log_device_placement(True)
    #with tf.device("/gpu:0"):
    not_use_gpu = True
    if not_use_gpu:
        model = Sequential()
        model.add(Dense(model_opt["Dense_input_dim"], input_shape=model_opt["input_dim"]))
        # model.add(Dropout(rate=model_opt["Dropout_rate"]))
        # model.add(LSTM(model_opt["LSTM_num_hidden_units"][0], return_sequences=True))
        # model.add(Dropout(rate=model_opt["Dropout_rate"]))
        # model.add(LSTM(model_opt["LSTM_num_hidden_units"][1], return_sequences=True))
        # # model.add(LSTM(model_opt["LSTM_num_hidden_units"][2], return_sequences=True))
        # model.add(LSTM(model_opt["LSTM_num_hidden_units"][3]))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Dense(model_opt["dense_out"]))
        # model.add(Dense(1, activation=model_opt["neurons_activation"]))
        # tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        # compile the keras model
        model.compile(loss=model_opt["metrics"],
                      optimizer=model_opt["optimizer"])

        # Create early stopping function
        erlstp_callback = callbacks.EarlyStopping(monitor="val_loss",  # loss o val_loss
                                                  patience=model_opt["patience"],
                                                  mode="min",
                                                  restore_best_weights=True,
                                                  verbose=1)
        # Create a callback that saves the model's weights
        ckpt_callback = callbacks.ModelCheckpoint(model_opt["model_path"] + 'model_prova.keras',
                                                  save_best_only=True,
                                                  save_weights_only=False,
                                                  monitor='loss',
                                                  mode='min')
        # Callback stop on NaN
        nan_callback = callbacks.TerminateOnNaN()

        cb_list = [erlstp_callback,  nan_callback, ckpt_callback]
        model.summary()
        # fit network
        history = model.fit(X, y, epochs=model_opt['epochs'], callbacks=cb_list, validation_split=model_opt['validation_split'])
        # history = 1
        # model.fit(X, y, epochs=model_opt['epochs'], callbacks=cb_list, validation_split=model_opt['validation_split'])

    return model, history



if __name__ == '__main__':
    output_dim = 2
    features = 1
    t_s = 15
    model_opt = {'num_hidden_units_1': output_dim,
                 'input_dim': (3, features),
                 'neurons_activation': 'relu',
                 'metrics': 'mse',
                 'optimizer': 'adam',
                 'patience': 10,
                 'model_path': '/'}
    model = create_model(model_opt)
    model.get_weights()
    # X = np.array([[0,0,1,2],
    # [0,0,1,2],
    # [0,0,1,2]])
    # X = np.array([[0,1],
    # [1,0],
    # [0,2]])
    # data = X.reshape((1, t_s, features))
    print('Computed variables: 4 x out_dim x (features + out_dim + 1) = {:d}'.format(
        4 * output_dim * (features + output_dim + 1)))
    # model.predict(data)

    a = 1
