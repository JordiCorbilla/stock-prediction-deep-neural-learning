"""Train the repository's actual v9 return head on externally prepared causal splits."""

import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    import numpy as np
    import tensorflow as tf

    from stock_prediction_lstm import LongShortTermMemory

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    with np.load(args.input, allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    if max(data["train_end"]) >= min(data["val_history_start"]) or max(data["val_end"]) >= min(data["test_start"]):
        raise ValueError("Training, validation or inference chronology overlaps")
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = []
    results = {}
    for seed in (7, 42, 123):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            factory = LongShortTermMemory(str(args.output))
            base = factory.create_return_multitask_model(data["train_x"])
        model = tf.keras.Model(base.inputs, base.get_layer("expected_return").output)
        model.compile(optimizer=factory.get_optimizer("v9"), loss="mean_squared_error")
        history = model.fit(data["train_x"], data["train_y"],
                            validation_data=(data["val_x"], data["val_y"]),
                            epochs=args.epochs, batch_size=256, shuffle=False,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                                       restore_best_weights=True)], verbose=0)
        prediction = model.predict(data["test_x"], batch_size=256, verbose=0).reshape(-1) * .02
        np.save(args.output / f"predictions-{seed}.npy", prediction)
        model.save(args.output / f"model-{seed}.keras")
        predictions.append(prediction)
        results[str(seed)] = {"epochs": len(history.history["loss"]), "history": history.history,
                              "best_epoch": int(np.argmin(history.history["val_loss"])) + 1}
        print(f"Seed {seed}: {results[str(seed)]['epochs']} epochs", flush=True)
    np.save(args.output / "predictions-ensemble.npy", np.mean(predictions, axis=0))
    repo = Path(__file__).resolve().parents[1]
    metadata = {"seeds": results, "tensorflow": tf.__version__, "max_epochs": args.epochs,
                "source_files": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [repo / "stock_prediction_lstm.py", Path(__file__),
                                              repo / "quant_forecast_lab/icaif.py"]},
                "input_hash": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "architecture": "Jordi Corbilla v9 128/64 LSTM encoder, expected_return head",
                "last_training_label": max(data["train_end"].tolist()),
                "first_validation_history": min(data["val_history_start"].tolist()),
                "last_validation_label": max(data["val_end"].tolist()),
                "first_test_start": min(data["test_start"].tolist()),
                "scale": "Fixed relative returns / 0.02; no full-data scaler"}
    (args.output / "training.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
