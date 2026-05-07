import numpy as np
from keras.models import load_model
from models.nn import build_lstm_model

class ForecastModel:

    def __init__(self, model_type="lstm", config=None):
        self.model_type = model_type
        self.config = config
        self.model = None

    def build_model(self):
        if self.model_type == "lstm":
            self.model = build_lstm_model(
                self.config["n_in"],
                self.config["n_features"],
                self.config["n_out"]
            )
        else:
            raise ValueError("Unsupported model type")

    def train(self, train_X, train_y, val_X, val_y):
        history = self.model.fit(
            train_X, train_y,
            epochs=self.config.get("epochs", 25),
            batch_size=self.config.get("batch_size", 128),
            validation_data=(val_X, val_y),
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)