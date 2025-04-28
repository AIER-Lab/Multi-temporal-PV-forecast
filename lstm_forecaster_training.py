import pandas as pd
from datetime import datetime
from pickle import dump
import os

# from PROPHET_LOAD_LSTM.util import util
from data_util import prepare_data_train
from model_util import create_model_attention


# from PROPHET_DB import mysql

# import daiquiri
# import sklearn.preprocessing

# from PROPHET_DB.setup_logger import setup_logger

# Logger definition
# LOGGER = daiquiri.getLogger(__name__)


def main(units,layers, model_folder):
    data_opt = {
        'n_back': 96,  # config.getint("data_opt", "n_back"),  # 4*24*7
        'n_timesteps': int(96),  # config.getint("data_opt", "n_timesteps"),  # 4*4
        'lag': 0,  # config.getint("data_opt", "lag"),
        'tr_per': 0.80,  # config.getfloat("data_opt", "tr_per"),
        'out_col': ['power'],  # config.get("data_opt", "out_col").split(','),
        'features': ['month', 'day', 'hour', 'minute', 'T', 'Q','VV', 'N', 'U', 'active_sessions'],  # config.get("data_opt", "features").split(','),['month', 'day', 'hour', 'minute']
        'freq': 15,  # config.getint("data_opt", "freq"),
        'tr_days_step': 1,  # config.getint("data_opt", "tr_days_step"),
    }
    data_opt['columns'] = data_opt['features'] + data_opt['out_col']
    data_opt['n_features'] = len(data_opt['columns'])

    model_opt = {'Dense_input_dim': 24,  # config.getint("model_opt", "Dense_input_dim"),
                 'LSTM_num_hidden_units': units,
                 # list(map(int, config.get("model_opt", "LSTM_num_hidden_units").split(','))),
                 'LSTM_layers': layers,  # config.getint("model_opt", "LSTM_layers"),
                 'metrics': 'mse',
                 # config.get("model_opt", "metrics"), 'optimizer': config.get("model_opt", "optimizer"),
                 'patience': 5,
                 # config.getint("model_opt", "patience"), 'epochs': config.getint("model_opt", "epochs"),
                 'validation_split': 0.2,  # config.getfloat("model_opt", "validation_split"),
                 'model_path': 'models/',  # config.get("model_opt", "model_path"),
                 'Dropout_rate': 0.2,  # config.getfloat("model_opt", "Dropout_rate"),
                 'input_dim': (data_opt['n_back'], data_opt['n_features']),
                 'dense_out': data_opt['n_timesteps'],
                 'optimizer': 'Adam',
                 'epochs': 1,
                 }


    df = pd.read_csv('data/power_profile_weather.csv', sep=",")
    df['times'] = pd.to_datetime(df['times'], format='%Y-%m-%d %H:%M:%S')
    fine_tr = int(len(df.index) * data_opt['tr_per'])
    fine_tr = df['times'][fine_tr]
    df.set_index(df['times'], inplace=True)

    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.dayofweek
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df = df[data_opt['columns']]

    ## definizione tempi inizio
    #now = datetime.utcnow()
    #fine_tr = pd.to_datetime(now)  # .tz_localize('UTC')  ## METTERE A POSTO MA SECONDARIO

    mask_tr = (df.index < fine_tr)
    train = df.loc[mask_tr]    #todo torna a questoooooo
    #train = df
    #### addestra il modello
    train_X, train_y, scaler_X, scaler_y = prepare_data_train(train, data_opt)
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    model, history = create_model_attention(model_opt, train_X, train_y)


    # Save the model inside the new folder
    model.save(model_folder + "model.keras")

    # Save the scalers in the same folder
    dump(scaler_X, open(model_folder + "scaler_in.pkl", 'wb'))
    dump(scaler_y, open(model_folder + "scaler_out.pkl", 'wb'))

    print(f"Model and scalers saved in: {model_folder}")


if __name__ == "__main__":
    units_list = [24]
    layers_list = [1]
    timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")

    for units in units_list:
        for layers in layers_list:
            model_folder = 'models/' + f"model_{timestamp}/" + f"{str(units)}_{str(layers)}/"
            os.makedirs(model_folder, exist_ok=True)
            main(units, layers, model_folder)
