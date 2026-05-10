# Week 10 HW04 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Week 10 HW04 submission comparing MLP/CNN for MNIST images and LSTM/Text CNN for Korean sentiment text, with graphs, results, notebook, and report.

**Architecture:** Keep the implementation close to the professor's TensorFlow/Keras notebooks. Put repeatable training code in `Assignment4/src/`, store generated figures/results under `Assignment4/outputs/`, and create a notebook/report that summarize the same outputs.

**Tech Stack:** Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, optional KoNLPy/Okt, Jupyter notebook, Markdown/PDF report.

---

## File Structure

- Create `Assignment4/src/train_image.py`
  - Loads MNIST, trains MLP and CNN, saves `image_results.json` and `image_loss_accuracy.png`.
- Create `Assignment4/src/train_text.py`
  - Loads NSMC or Week10 local Korean review files, preprocesses/tokenizes text, trains LSTM and Text CNN, saves `text_results.json` and `text_loss_accuracy.png`.
- Create `Assignment4/notebooks/week10_hw04.ipynb`
  - Submission-facing notebook that explains and runs/loads both experiment outputs.
- Create `Assignment4/reports/week10_hw04_report.md`
  - Korean 2-4 page report source.
- Create `Assignment4/reports/week10_hw04_report.pdf`
  - PDF export of the report.
- Create `Assignment4/outputs/figures/.gitkeep`
  - Keeps figure output folder in git before generated PNGs exist.
- Create `Assignment4/outputs/results/.gitkeep`
  - Keeps result output folder in git before generated JSON files exist.
- Modify `.gitignore` only if generated local caches or downloaded datasets appear and are not already ignored.

## Task 1: Create Assignment4 Skeleton

**Files:**
- Create: `Assignment4/src/`
- Create: `Assignment4/notebooks/`
- Create: `Assignment4/reports/`
- Create: `Assignment4/outputs/figures/.gitkeep`
- Create: `Assignment4/outputs/results/.gitkeep`
- Create: `Assignment4/data/.gitkeep`

- [ ] **Step 1: Create directories**

Run:

```powershell
New-Item -ItemType Directory -Force -Path 'Assignment4/src','Assignment4/notebooks','Assignment4/reports','Assignment4/outputs/figures','Assignment4/outputs/results','Assignment4/data'
```

Expected: all listed directories exist.

- [ ] **Step 2: Add `.gitkeep` files**

Use `apply_patch` to create:

```text
Assignment4/outputs/figures/.gitkeep
Assignment4/outputs/results/.gitkeep
Assignment4/data/.gitkeep
```

Each file should be empty.

- [ ] **Step 3: Verify structure**

Run:

```powershell
Get-ChildItem -Recurse Assignment4
```

Expected: `src`, `notebooks`, `reports`, `outputs`, and `data` directories are visible.

- [ ] **Step 4: Commit skeleton**

Run:

```powershell
git add Assignment4
git commit -m "Create assignment4 structure"
```

Expected: commit succeeds on branch `assignment4-week10-hw04`.

## Task 2: Implement Image Experiment Script

**Files:**
- Create: `Assignment4/src/train_image.py`
- Output: `Assignment4/outputs/results/image_results.json`
- Output: `Assignment4/outputs/figures/image_loss_accuracy.png`

- [ ] **Step 1: Create `train_image.py` with CLI, data loading, and model builders**

Use `apply_patch` to create `Assignment4/src/train_image.py` with this structure:

```python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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
```

- [ ] **Step 2: Add training, evaluation, plotting, and main function**

Append these functions to the same file:

```python
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
```

- [ ] **Step 3: Run quick image smoke test**

Run:

```powershell
python Assignment4/src/train_image.py --epochs 1 --sample-size 1000 --batch-size 128
```

Expected:

- Command exits with code `0`.
- `Assignment4/outputs/results/image_results.json` exists.
- `Assignment4/outputs/figures/image_loss_accuracy.png` exists.
- JSON contains `MLP` and `CNN` under `models`.

- [ ] **Step 4: Inspect image result fields**

Run:

```powershell
python -c "import json; p='Assignment4/outputs/results/image_results.json'; d=json.load(open(p, encoding='utf-8')); print(d['dataset']['train_shape']); print(d['models'].keys())"
```

Expected: first line is a list-like MNIST shape, second line includes `MLP` and `CNN`.

- [ ] **Step 5: Commit image script**

Run:

```powershell
git add Assignment4/src/train_image.py Assignment4/outputs/figures/.gitkeep Assignment4/outputs/results/.gitkeep Assignment4/data/.gitkeep
git commit -m "Add MNIST image baselines"
```

Expected: commit succeeds.

## Task 3: Implement Text Experiment Script

**Files:**
- Create: `Assignment4/src/train_text.py`
- Output: `Assignment4/outputs/results/text_results.json`
- Output: `Assignment4/outputs/figures/text_loss_accuracy.png`

- [ ] **Step 1: Create `train_text.py` imports, paths, tokenizer, and data discovery**

Use `apply_patch` to create `Assignment4/src/train_text.py` with:

```python
from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

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
```

- [ ] **Step 2: Add dataset loading and vectorization**

Append:

```python
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
```

- [ ] **Step 3: Add LSTM/Text CNN builders and evaluation**

Append:

```python
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
```

- [ ] **Step 4: Add experiment runner and CLI**

Append:

```python
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
```

- [ ] **Step 5: Run quick text smoke test**

Run:

```powershell
python Assignment4/src/train_text.py --epochs 1 --sample-size 1000 --batch-size 16
```

Expected:

- Command exits with code `0`.
- `Assignment4/outputs/results/text_results.json` exists.
- `Assignment4/outputs/figures/text_loss_accuracy.png` exists.
- JSON includes `LSTM` and `Text CNN`.

If this fails because NSMC download is blocked, rerun the same command with escalated network permission.

- [ ] **Step 6: Inspect text result fields**

Run:

```powershell
python -c "import json; p='Assignment4/outputs/results/text_results.json'; d=json.load(open(p, encoding='utf-8')); print(d['dataset']['tokenizer']); print(d['models'].keys())"
```

Expected: first line names either Okt or whitespace tokenizer, second line includes `LSTM` and `Text CNN`.

- [ ] **Step 7: Commit text script**

Run:

```powershell
git add Assignment4/src/train_text.py
git commit -m "Add Korean sentiment baselines"
```

Expected: commit succeeds.

## Task 4: Run Full Experiments and Save Outputs

**Files:**
- Create/Update: `Assignment4/outputs/results/image_results.json`
- Create/Update: `Assignment4/outputs/results/text_results.json`
- Create/Update: `Assignment4/outputs/figures/image_loss_accuracy.png`
- Create/Update: `Assignment4/outputs/figures/text_loss_accuracy.png`

- [ ] **Step 1: Run image experiment**

Run:

```powershell
python Assignment4/src/train_image.py --epochs 5 --batch-size 128
```

Expected: image script completes and saves result/figure files.

- [ ] **Step 2: Run text experiment**

Run:

```powershell
python Assignment4/src/train_text.py --epochs 5 --sample-size 10000 --batch-size 16
```

Expected: text script completes and saves result/figure files. If local runtime is too slow, use `--sample-size 5000` and record that sample size in the report.

- [ ] **Step 3: Check all generated outputs exist**

Run:

```powershell
Test-Path Assignment4/outputs/results/image_results.json
Test-Path Assignment4/outputs/results/text_results.json
Test-Path Assignment4/outputs/figures/image_loss_accuracy.png
Test-Path Assignment4/outputs/figures/text_loss_accuracy.png
```

Expected: four lines of `True`.

- [ ] **Step 4: Confirm loss histories have at least one epoch**

Run:

```powershell
python -c "import json; files=['image_results.json','text_results.json']; base='Assignment4/outputs/results'; [print(f, {m: len(v['loss']) for m,v in json.load(open(base+'/'+f, encoding='utf-8'))['history'].items()}) for f in files]"
```

Expected: each model reports a positive epoch count.

- [ ] **Step 5: Commit generated experiment outputs**

Run:

```powershell
git add Assignment4/outputs/results Assignment4/outputs/figures
git commit -m "Add assignment4 experiment outputs"
```

Expected: commit succeeds.

## Task 5: Create Submission Notebook

**Files:**
- Create: `Assignment4/notebooks/week10_hw04.ipynb`

- [ ] **Step 1: Create notebook with sections**

Create `Assignment4/notebooks/week10_hw04.ipynb` with cells in this order:

```text
1. Markdown: title and assignment objective
2. Markdown: Part A Image Task overview
3. Code: load and display image_results.json
4. Code: display image_loss_accuracy.png
5. Markdown: image model structure explanation
6. Markdown: Part B Text Task overview
7. Code: load and display text_results.json
8. Code: display text_loss_accuracy.png
9. Markdown: tokenization and model structure explanation
10. Markdown: interpretation comparing image and text input structure
```

Use `nbformat` or Jupyter to create the notebook. The JSON must reference paths relative to the notebook:

```python
from pathlib import Path
import json
from IPython.display import Image, display

ROOT = Path("..")
with open(ROOT / "outputs" / "results" / "image_results.json", encoding="utf-8") as f:
    image_results = json.load(f)
image_results["models"]
```

and:

```python
display(Image(filename=str(ROOT / "outputs" / "figures" / "image_loss_accuracy.png")))
```

- [ ] **Step 2: Verify notebook JSON opens**

Run:

```powershell
python -c "import nbformat; nb=nbformat.read('Assignment4/notebooks/week10_hw04.ipynb', as_version=4); print(len(nb.cells)); print(nb.cells[0].cell_type)"
```

Expected: cell count is at least `10`, first cell type is `markdown`.

- [ ] **Step 3: Commit notebook**

Run:

```powershell
git add Assignment4/notebooks/week10_hw04.ipynb
git commit -m "Add assignment4 submission notebook"
```

Expected: commit succeeds.

## Task 6: Write Korean Report and Export PDF

**Files:**
- Create: `Assignment4/reports/week10_hw04_report.md`
- Create: `Assignment4/reports/week10_hw04_report.pdf`

- [ ] **Step 1: Draft report markdown**

Create `Assignment4/reports/week10_hw04_report.md` in Korean with this structure:

```markdown
# Week 10 HW04 보고서

## 1. 사용 데이터

### 이미지 데이터: MNIST
- 입력 shape:
- 클래스 수:
- train/validation/test 크기:

### 텍스트 데이터: NSMC 또는 한국어 리뷰 데이터
- 문장 수:
- 클래스 수:
- 토큰화 방식:

## 2. 모델 구조

### 이미지 모델
- MLP:
- CNN:

### 텍스트 모델
- LSTM:
- Text CNN:

## 3. 실험 결과

### 이미지 모델 성능

| 모델 | Train Accuracy | Validation Accuracy | Test Accuracy |
|---|---:|---:|---:|
| MLP |  |  |  |
| CNN |  |  |  |

### 텍스트 모델 성능

| 모델 | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| LSTM |  |  |
| Text CNN |  |  |

![Image loss/accuracy](../outputs/figures/image_loss_accuracy.png)

![Text loss/accuracy](../outputs/figures/text_loss_accuracy.png)

## 4. 해석

이미지에서는 CNN이 MLP보다 유리하다...

텍스트에서는 단어 순서가 의미에 영향을 주기 때문에 sequence 모델이 필요하다...

두 task를 비교하면 입력 구조의 차이가 모델 설계에 직접적인 영향을 준다...
```

Fill the blanks from `image_results.json` and `text_results.json`.

- [ ] **Step 2: Validate report includes required sections**

Run:

```powershell
rg -n "사용 데이터|모델 구조|실험 결과|해석|CNN|sequence|토큰화" Assignment4/reports/week10_hw04_report.md
```

Expected: all required report topics appear.

- [ ] **Step 3: Export report PDF**

Preferred command if Pandoc is available:

```powershell
pandoc Assignment4/reports/week10_hw04_report.md -o Assignment4/reports/week10_hw04_report.pdf
```

Fallback if Pandoc is unavailable:

```powershell
python -m pip install markdown weasyprint
python -c "from pathlib import Path; import markdown; from weasyprint import HTML; md=Path('Assignment4/reports/week10_hw04_report.md').read_text(encoding='utf-8'); HTML(string=markdown.markdown(md, extensions=['tables'])).write_pdf('Assignment4/reports/week10_hw04_report.pdf')"
```

Expected: `Assignment4/reports/week10_hw04_report.pdf` exists. If PDF export tooling is unavailable locally, keep the completed Markdown and note the blocker before final handoff.

- [ ] **Step 4: Verify PDF exists**

Run:

```powershell
Test-Path Assignment4/reports/week10_hw04_report.pdf
```

Expected: `True`.

- [ ] **Step 5: Commit report**

Run:

```powershell
git add Assignment4/reports/week10_hw04_report.md Assignment4/reports/week10_hw04_report.pdf
git commit -m "Add assignment4 report"
```

Expected: commit succeeds.

## Task 7: Final Verification

**Files:**
- Read: `Assignment4/src/train_image.py`
- Read: `Assignment4/src/train_text.py`
- Read: `Assignment4/notebooks/week10_hw04.ipynb`
- Read: `Assignment4/reports/week10_hw04_report.md`
- Read: `Assignment4/reports/week10_hw04_report.pdf`
- Read: `Assignment4/outputs/results/image_results.json`
- Read: `Assignment4/outputs/results/text_results.json`

- [ ] **Step 1: Check branch and status**

Run:

```powershell
git branch --show-current
git status --short
```

Expected: branch is `assignment4-week10-hw04`; status is clean before final verification or contains only intentional uncommitted edits.

- [ ] **Step 2: Run fast smoke commands**

Run:

```powershell
python Assignment4/src/train_image.py --epochs 1 --sample-size 1000 --batch-size 128
python Assignment4/src/train_text.py --epochs 1 --sample-size 1000 --batch-size 16
```

Expected: both commands exit with code `0`.

- [ ] **Step 3: Verify required deliverables**

Run:

```powershell
Test-Path Assignment4/src/train_image.py
Test-Path Assignment4/src/train_text.py
Test-Path Assignment4/notebooks/week10_hw04.ipynb
Test-Path Assignment4/reports/week10_hw04_report.md
Test-Path Assignment4/reports/week10_hw04_report.pdf
Test-Path Assignment4/outputs/figures/image_loss_accuracy.png
Test-Path Assignment4/outputs/figures/text_loss_accuracy.png
```

Expected: seven lines of `True`.

- [ ] **Step 4: Verify result JSON metrics are present**

Run:

```powershell
python -c "import json; img=json.load(open('Assignment4/outputs/results/image_results.json', encoding='utf-8')); txt=json.load(open('Assignment4/outputs/results/text_results.json', encoding='utf-8')); print(img['models']['MLP']['test_accuracy']); print(img['models']['CNN']['test_accuracy']); print(txt['models']['LSTM']['test']['accuracy']); print(txt['models']['Text CNN']['test']['accuracy'])"
```

Expected: four numeric accuracy values print.

- [ ] **Step 5: Commit any final verification changes**

Run:

```powershell
git status --short
```

If smoke tests changed generated outputs, run:

```powershell
git add Assignment4/outputs/results Assignment4/outputs/figures
git commit -m "Refresh assignment4 verification outputs"
```

Expected: final `git status --short` is clean.

## Task 8: Merge Back to Main

**Files:**
- No file edits expected.

- [ ] **Step 1: Confirm branch is clean**

Run:

```powershell
git status --short
```

Expected: no output.

- [ ] **Step 2: Switch to main**

Run:

```powershell
git switch main
```

Expected: branch switches to `main`.

- [ ] **Step 3: Merge feature branch**

Run:

```powershell
git merge --no-ff assignment4-week10-hw04 -m "Merge assignment4 week10 hw04"
```

Expected: merge commit succeeds without conflicts.

- [ ] **Step 4: Verify merged files on main**

Run:

```powershell
git branch --show-current
Test-Path Assignment4/src/train_image.py
Test-Path Assignment4/reports/week10_hw04_report.pdf
```

Expected: current branch is `main`; both paths return `True`.

## Self-Review Notes

- Spec coverage:
  - Part A MNIST MLP/CNN: Task 2 and Task 4.
  - Part A train/validation/test accuracy and graph: Task 2 and Task 4.
  - Part B Korean sentiment tokenization and LSTM/Text CNN: Task 3 and Task 4.
  - Part B validation/test accuracy and graph: Task 3 and Task 4.
  - Notebook and PDF report deliverables: Task 5 and Task 6.
  - Merge back to main: Task 8.
- Placeholder scan: no red-flag patterns or open-ended implementation placeholders remain.
- Type consistency: result JSON keys are consistent across script, notebook, report, and verification steps.
