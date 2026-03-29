import warnings
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# =========================
# 1. 경로 및 전역 설정
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2


# =========================
# 2. 데이터 로드
# =========================

def load_data(file_path: Path) -> pd.DataFrame:
    """CSV 데이터를 불러온다."""
    return pd.read_csv(file_path)


# =========================
# 3. 데이터 구조 확인용 요약
# =========================

def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼별 dtype, 결측치 개수, 결측치 비율을 요약한다."""
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isna().sum(),
        "missing_ratio": (df.isna().mean() * 100).round(2),
    })
    return summary


# =========================
# 4. 입력/타깃 분리 및 컬럼 선택
# =========================

def prepare_features_and_target(df: pd.DataFrame):
    """타깃과 입력 데이터를 분리하고, baseline에 사용할 컬럼만 남긴다."""
    target_col = "Survived"

    # baseline에서 제외할 컬럼
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = X.drop(columns=drop_cols)

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    return X, y, numeric_features, categorical_features


# =========================
# 5. Train/Test Split
# =========================

def split_data(X: pd.DataFrame, y: pd.Series):
    """학습/테스트 데이터를 분리한다."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# =========================
# 6. 전처리기 생성
# =========================

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scaler=None,
) -> ColumnTransformer:
    """숫자형/범주형 전처리기를 ColumnTransformer로 구성한다."""
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scaler is not None:
        numeric_steps.append(("scaler", scaler))

    numeric_transformer = Pipeline(steps=numeric_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


# =========================
# 7. 모델 파이프라인 생성
# =========================

def build_pipeline(
    model,
    numeric_features: list[str],
    categorical_features: list[str],
    scaler=None,
) -> Pipeline:
    """전처리 + 모델을 하나의 Pipeline으로 묶는다."""
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scaler=scaler,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


# =========================
# 8. 단일 실험 실행
# =========================

def run_experiment(
    experiment_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[dict, str]:
    """단일 실험을 수행하고 주요 지표와 상세 리포트를 반환한다."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)

    result = {
        "experiment": experiment_name,
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": str(cm.tolist()),
    }

    report_text = (
        f"[{experiment_name}]\n"
        f"Accuracy: {acc:.4f}\n"
        f"F1-score: {f1:.4f}\n"
        f"Confusion Matrix:\n{cm}\n\n"
        f"Classification Report:\n{clf_report}\n"
        f"{'-' * 60}\n"
    )

    return result, report_text


# =========================
# 9. 전체 실험 정의
# =========================

def build_experiments(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    """과제 요구사항에 맞는 baseline 실험 세트를 정의한다."""
    experiments = {
        "SVM_no_scaling": build_pipeline(
            model=SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaler=None,
        ),
        "StandardScaler_SVM": build_pipeline(
            model=SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaler=StandardScaler(),
        ),
        "StandardScaler_KNN": build_pipeline(
            model=KNeighborsClassifier(n_neighbors=5),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaler=StandardScaler(),
        ),
        "MinMaxScaler_KNN": build_pipeline(
            model=KNeighborsClassifier(n_neighbors=5),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scaler=MinMaxScaler(),
        ),
    }
    return experiments


# =========================
# 10. 결과 저장
# =========================

def save_results(results: list[dict], report_texts: list[str]) -> None:
    """실험 결과를 csv와 txt로 저장한다."""
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "metrics.csv", index=False, encoding="utf-8-sig")

    with open(RESULTS_DIR / "confusion_matrices.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_texts))


# =========================
# 11. 메인 실행
# =========================

def main():
    print("=== Assignment 2: Baseline Modeling & Versioning ===")

    # 1) 데이터 로드
    df = load_data(DATA_PATH)
    print(f"데이터 shape: {df.shape}")

    # 2) 데이터 구조 확인
    summary_df = summarize_dataframe(df)
    print("\n[데이터 요약]")
    print(summary_df)

    # 3) 입력/타깃 및 컬럼 준비
    X, y, numeric_features, categorical_features = prepare_features_and_target(df)

    # 4) 학습/테스트 분리
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5) baseline 실험 정의
    experiments = build_experiments(numeric_features, categorical_features)

    # 6) 실험 반복 실행
    results = []
    report_texts = []

    for experiment_name, pipeline in experiments.items():
        result, report_text = run_experiment(
            experiment_name=experiment_name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(result)
        report_texts.append(report_text)

    # 7) 결과 저장
    save_results(results, report_texts)

    # 8) 결과 출력
    results_df = pd.DataFrame(results).sort_values(
        by=["f1_score", "accuracy"],
        ascending=False,
    )
    print("\n[실험 결과 요약]")
    print(results_df)

    print("\n결과 파일 저장 완료:")
    print(f"- {RESULTS_DIR / 'metrics.csv'}")
    print(f"- {RESULTS_DIR / 'confusion_matrices.txt'}")


if __name__ == "__main__":
    main()