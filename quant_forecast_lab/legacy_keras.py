"""Load the repository's historical Keras 2 HDF5 model under Keras 3.

Only the layer types used by the archived sequential model are accepted. The
weights are read by Keras from the original HDF5 file; no retraining occurs.
"""

import json

import h5py
import tensorflow as tf


def load_legacy_h5(path):
    with h5py.File(path, "r") as archive:
        config = json.loads(archive.attrs["model_config"])
    if config.get("class_name") != "Sequential":
        raise ValueError("Only archived Sequential Keras HDF5 models are supported.")

    layers = config["config"]["layers"]
    first = layers[0]["config"]
    input_shape = first.get("batch_input_shape") or first.get("batch_shape")
    if not input_shape or len(input_shape) != 3:
        raise ValueError("The archived model must have a three-dimensional LSTM input.")
    model = tf.keras.Sequential([tf.keras.Input(shape=tuple(input_shape[1:]))])
    for entry in layers:
        kind = entry["class_name"]
        source = entry["config"]
        name = source["name"]
        if kind == "InputLayer":
            continue
        if kind == "LSTM":
            model.add(tf.keras.layers.LSTM(
                source["units"],
                activation=source["activation"],
                recurrent_activation=source["recurrent_activation"],
                use_bias=source["use_bias"],
                unit_forget_bias=source["unit_forget_bias"],
                return_sequences=source["return_sequences"],
                name=name,
            ))
        elif kind == "Dropout":
            model.add(tf.keras.layers.Dropout(source["rate"], name=name))
        elif kind == "Dense":
            model.add(tf.keras.layers.Dense(source["units"], activation=source["activation"], name=name))
        else:
            raise ValueError(f"Unsupported archived Keras layer: {kind}")
    model.load_weights(path)
    return model
