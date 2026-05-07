import os
import numpy as np

from data.dataset import SeqDataset
from models.forecast import ForecastModel
from utils.assets import evaluate_and_plot

config = {
    "n_in": 56,
    "n_features": 10,
    "n_out": 1,  # prediction horizon 1  -> single-step , 56 -> multi-step
    "tr_per": 0.75,
    "val_per": 0.10,
    "epochs": 25,
    "batch_size": 128,
    "lag": 1,
    "mode": "single_model",  # modes "single_model or "multi_model"
    "save_dir": "saved_models"
}

os.makedirs(config["save_dir"], exist_ok=True)

if config["mode"] == "single_model":

    dataset = SeqDataset(
        "data/Dataset with WPD.csv",
        config
    )

    (train_X, train_y, val_X, val_y, test_X, test_y, scaler_X, scaler_y, *_) = dataset.prepare()

    config["n_features"] = train_X.shape[2]

    model = ForecastModel("lstm", config)
    model.build_model()

    model.train(train_X, train_y, val_X, val_y)

    model_path = os.path.join(config["save_dir"], f"single_model_{config['mode']}.keras")

    model.save_model(model_path)

    print(f"\nModel saved in: {model_path}")

    loaded_model = ForecastModel("lstm", config)
    loaded_model.load_model(model_path)

    print("Model loaded")

    yhat = loaded_model.predict(test_X)

    yhat_inv = scaler_y.inverse_transform(yhat)
    test_y_inv = scaler_y.inverse_transform(test_y)

    if config["n_out"] > 1:

        # last horizon only
        evaluate_and_plot(
            test_y_inv[:, -1],
            yhat_inv[:, -1],
            title="Single Model - Multi-step"
        )

    else:

        evaluate_and_plot(
            test_y_inv,
            yhat_inv,
            title="Single Model - Single-step"
        )

elif config["mode"] == "multi_model":

    datasets = [
        SeqDataset("data/SS1.csv", config),
        SeqDataset("data/SS2.csv", config),
        SeqDataset("data/SS3.csv", config),
        SeqDataset("data/SS4.csv", config),
    ]

    target_dataset = SeqDataset("data/PV.csv", config)

    models = []
    preds = []
    scalers_y = []

    for i, ds in enumerate(datasets):
        print(f"\nTraining model {i + 1}/4\n")

        (train_X, train_y, val_X, val_y, test_X, test_y, scaler_X, scaler_y, *_) = ds.prepare()

        scalers_y.append(scaler_y)

        config["n_features"] = train_X.shape[2]

        model = ForecastModel("lstm", config)
        model.build_model()

        model.train(train_X, train_y, val_X, val_y)

        model_path = os.path.join(
            config["save_dir"],
            f"model_{i + 1}_{config['mode']}.keras"
        )

        model.save_model(model_path)

        print(f"Model {i + 1} saved.")

        loaded_model = ForecastModel("lstm", config)
        loaded_model.load_model(model_path)

        print(f"Model {i + 1} reloaded.")

        yhat = loaded_model.predict(test_X)

        preds.append(yhat)

        models.append(loaded_model)

    (_, _, _, _, test_X_r, test_y_r, _, scaler_y_r, *_) = target_dataset.prepare()

    inv_preds = []

    for i in range(len(preds)):
        inv_preds.append(
            scalers_y[i].inverse_transform(preds[i])
        )

    test_y_inv = scaler_y_r.inverse_transform(test_y_r)

    weights = np.array([0.9, -0.8, -0.1, 0.9])

    final_pred = (
            inv_preds[0] * weights[0] +
            inv_preds[1] * weights[1] +
            inv_preds[2] * weights[2] +
            inv_preds[3] * weights[3]
    )

    if config["n_out"] > 1:

        evaluate_and_plot(
            test_y_inv[:, -1],
            final_pred[:, -1],
            title="Multi-model Multi-step"
        )

    else:

        evaluate_and_plot(
            test_y_inv,
            final_pred,
            title="Multi-model Single-step"
        )

else:

    raise ValueError(
        "mode must be 'single_model' or 'multi_model'"
    )
