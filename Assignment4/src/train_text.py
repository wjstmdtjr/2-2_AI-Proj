from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
RESULT_DIR = PROJECT_DIR / "outputs" / "results"
NSMC_TRAIN_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
NSMC_TEST_URL = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

try:
    from konlpy.tag import Okt

    OKT = Okt()
    KONLPY_AVAILABLE = True
except Exception:
    OKT = None
    KONLPY_AVAILABLE = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_korean(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if KONLPY_AVAILABLE and OKT is not None:
        return OKT.morphs(text, stem=True)
    return text.split()


def download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def find_local_review_files() -> tuple[Path | None, Path | None]:
    candidates = [
        DATA_DIR,
        PROJECT_DIR,
        PROJECT_DIR.parent,
        Path.cwd(),
    ]
    train_names = ["ratings_train.txt", "train.txt"]
    test_names = ["ratings_test.txt", "test.txt"]

    train_path = None
    test_path = None
    for base in candidates:
        for name in train_names:
            path = base / name
            if path.exists():
                train_path = path
                break
        for name in test_names:
            path = base / name
            if path.exists():
                test_path = path
                break
        if train_path and test_path:
            return train_path, test_path
    return train_path, test_path


def read_review_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "document" in df.columns:
        text_col = "document"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError(f"No text column found in {path}")
    if "label" not in df.columns:
        raise ValueError(f"No label column found in {path}")

    df = df[[text_col, "label"]].rename(columns={text_col: "text"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype("int32")
    return df


def load_text_data(sample_size: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    train_path, test_path = find_local_review_files()
    if train_path is None or test_path is None:
        train_path = DATA_DIR / "ratings_train.txt"
        test_path = DATA_DIR / "ratings_test.txt"
        download_if_missing(train_path, NSMC_TRAIN_URL)
        download_if_missing(test_path, NSMC_TEST_URL)

    train_full = read_review_file(train_path)
    test_df = read_review_file(test_path)

    if sample_size is not None:
        train_full = train_full.sample(n=min(sample_size, len(train_full)), random_state=seed)
        test_df = test_df.sample(n=min(max(1000, sample_size // 5), len(test_df)), random_state=seed)

    train_df, valid_df = train_test_split(
        train_full,
        test_size=0.2,
        random_state=seed,
        stratify=train_full["label"],
    )

    source = f"train={train_path.name}, test={test_path.name}"
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True), source


def prepare_texts(df: pd.DataFrame) -> list[str]:
    return df["text"].astype(str).apply(lambda x: " ".join(tokenize_korean(x))).tolist()


def make_datasets(
    train_texts: list[str],
    valid_texts: list[str],
    test_texts: list[str],
    train_labels: np.ndarray,
    valid_labels: np.ndarray,
    test_labels: np.ndarray,
    max_tokens: int,
    sequence_length: int,
    batch_size: int,
) -> tuple[layers.TextVectorization, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    vectorizer = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorizer.adapt(train_texts)

    def vectorize(text: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return vectorizer(text), label

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_texts, train_labels))
        .shuffle(buffer_size=len(train_texts), seed=42)
        .batch(batch_size)
        .map(vectorize)
        .prefetch(tf.data.AUTOTUNE)
    )
    valid_ds = (
        tf.data.Dataset.from_tensor_slices((valid_texts, valid_labels))
        .batch(batch_size)
        .map(vectorize)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_texts, test_labels))
        .batch(batch_size)
        .map(vectorize)
        .prefetch(tf.data.AUTOTUNE)
    )
    return vectorizer, train_ds, valid_ds, test_ds


def build_lstm(max_tokens: int, embedding_dim: int, sequence_length: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length,)),
            layers.Embedding(max_tokens, embedding_dim),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="lstm_baseline",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_text_cnn(max_tokens: int, embedding_dim: int, sequence_length: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length,)),
            layers.Embedding(max_tokens, embedding_dim),
            layers.Conv1D(64, kernel_size=3, activation="relu"),
            layers.GlobalMaxPooling1D(),
            layers.Dropout(0.3),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="text_cnn_baseline",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_binary_model(model: keras.Model, dataset: tf.data.Dataset) -> dict[str, float]:
    loss, accuracy = model.evaluate(dataset, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def plot_histories(histories: dict[str, keras.callbacks.History], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, history in histories.items():
        axes[0].plot(history.history["loss"], label=f"{model_name} train")
        axes[0].plot(history.history["val_loss"], label=f"{model_name} valid")
        axes[1].plot(history.history["accuracy"], label=f"{model_name} train")
        axes[1].plot(history.history["val_accuracy"], label=f"{model_name} valid")

    axes[0].set_title("Text Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Text Model Accuracy")
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df, source = load_text_data(args.sample_size, args.seed)

    train_texts = prepare_texts(train_df)
    valid_texts = prepare_texts(valid_df)
    test_texts = prepare_texts(test_df)
    train_labels = train_df["label"].astype("int32").to_numpy()
    valid_labels = valid_df["label"].astype("int32").to_numpy()
    test_labels = test_df["label"].astype("int32").to_numpy()

    vectorizer, train_ds, valid_ds, test_ds = make_datasets(
        train_texts,
        valid_texts,
        test_texts,
        train_labels,
        valid_labels,
        test_labels,
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
        )
    ]

    lstm_model = build_lstm(args.max_tokens, args.embedding_dim, args.sequence_length)
    history_lstm = lstm_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    cnn_model = build_text_cnn(args.max_tokens, args.embedding_dim, args.sequence_length)
    history_cnn = cnn_model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    results = {
        "dataset": {
            "name": "NSMC or class Korean review data",
            "source": source,
            "train_sentences": int(len(train_df)),
            "validation_sentences": int(len(valid_df)),
            "test_sentences": int(len(test_df)),
            "num_classes": 2,
            "tokenizer": "Okt morphological tokenizer" if KONLPY_AVAILABLE else "whitespace tokenizer",
            "max_tokens": int(args.max_tokens),
            "sequence_length": int(args.sequence_length),
            "vocabulary_size": int(len(vectorizer.get_vocabulary())),
        },
        "models": {
            "LSTM": {
                "validation": evaluate_binary_model(lstm_model, valid_ds),
                "test": evaluate_binary_model(lstm_model, test_ds),
            },
            "Text CNN": {
                "validation": evaluate_binary_model(cnn_model, valid_ds),
                "test": evaluate_binary_model(cnn_model, test_ds),
            },
        },
        "history": {
            "LSTM": {key: [float(v) for v in values] for key, values in history_lstm.history.items()},
            "Text CNN": {key: [float(v) for v in values] for key, values in history_cnn.history.items()},
        },
    }

    plot_histories(
        {"LSTM": history_lstm, "Text CNN": history_cnn},
        FIGURE_DIR / "text_loss_accuracy.png",
    )
    with (RESULT_DIR / "text_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Korean sentiment LSTM and Text CNN baselines.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
