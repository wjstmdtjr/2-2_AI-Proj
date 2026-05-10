from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


PROJECT_DIR = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
RESULT_DIR = PROJECT_DIR / "outputs" / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_mnist(validation_size: int, sample_size: int | None) -> tuple:
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_valid = x_train_full[:validation_size]
    y_valid = y_train_full[:validation_size]
    x_train = x_train_full[validation_size:]
    y_train = y_train_full[validation_size:]

    if sample_size is not None:
        x_train = x_train[:sample_size]
        y_train = y_train[:sample_size]
        valid_limit = min(validation_size, max(1000, sample_size // 5))
        test_limit = min(len(x_test), max(1000, sample_size // 5))
        x_valid = x_valid[:valid_limit]
        y_valid = y_valid[:valid_limit]
        x_test = x_test[:test_limit]
        y_test = y_test[:test_limit]

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def build_mlp() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(784,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="mlp_baseline",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(10, activation="softmax"),
        ],
        name="cnn_baseline",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
    valid_loss, valid_acc = model.evaluate(x_valid, y_valid, verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return {
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "validation_loss": float(valid_loss),
        "validation_accuracy": float(valid_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }


def plot_histories(histories: dict[str, keras.callbacks.History], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, history in histories.items():
        axes[0].plot(history.history["loss"], label=f"{model_name} train")
        axes[0].plot(history.history["val_loss"], label=f"{model_name} valid")
        axes[1].plot(history.history["accuracy"], label=f"{model_name} train")
        axes[1].plot(history.history["val_accuracy"], label=f"{model_name} valid")

    axes[0].set_title("Image Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Image Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_experiment(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_mnist(
        validation_size=args.validation_size,
        sample_size=args.sample_size,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        )
    ]

    x_train_mlp = x_train.reshape(len(x_train), -1)
    x_valid_mlp = x_valid.reshape(len(x_valid), -1)
    x_test_mlp = x_test.reshape(len(x_test), -1)

    mlp_model = build_mlp()
    history_mlp = mlp_model.fit(
        x_train_mlp,
        y_train,
        validation_data=(x_valid_mlp, y_valid),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    x_train_cnn = np.expand_dims(x_train, axis=-1)
    x_valid_cnn = np.expand_dims(x_valid, axis=-1)
    x_test_cnn = np.expand_dims(x_test, axis=-1)

    cnn_model = build_cnn()
    history_cnn = cnn_model.fit(
        x_train_cnn,
        y_train,
        validation_data=(x_valid_cnn, y_valid),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    results = {
        "dataset": {
            "name": "MNIST",
            "train_shape": list(x_train.shape),
            "validation_shape": list(x_valid.shape),
            "test_shape": list(x_test.shape),
            "num_classes": 10,
        },
        "models": {
            "MLP": evaluate_model(
                mlp_model, x_train_mlp, y_train, x_valid_mlp, y_valid, x_test_mlp, y_test
            ),
            "CNN": evaluate_model(
                cnn_model, x_train_cnn, y_train, x_valid_cnn, y_valid, x_test_cnn, y_test
            ),
        },
        "history": {
            "MLP": {key: [float(v) for v in values] for key, values in history_mlp.history.items()},
            "CNN": {key: [float(v) for v in values] for key, values in history_cnn.history.items()},
        },
    }

    plot_histories(
        {"MLP": history_mlp, "CNN": history_cnn},
        FIGURE_DIR / "image_loss_accuracy.png",
    )
    with (RESULT_DIR / "image_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MNIST MLP and CNN baselines.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
