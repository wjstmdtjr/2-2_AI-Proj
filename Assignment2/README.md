# Assignment 2 - Baseline Modeling & Versioning

## 1. Objective
이번 과제의 목표는 Titanic 데이터셋에 대해 재현 가능한 baseline 실험 구조를 설계하고,  
KNN과 SVM의 성능을 비교하는 것이다.  
특히 단순 모델 비교가 아니라 전처리와 모델을 하나의 Pipeline으로 묶어 실험하는 데 초점을 두었다.

## 2. Dataset
- Dataset: Titanic
- Target: `Survived`
- 데이터 특성:
  - 수치형 + 범주형 혼합
  - 결측치 존재 (`Age`, `Embarked`, `Cabin` 등)

## 3. Preprocessing Design
Titanic 데이터는 수치형과 범주형이 혼합되어 있기 때문에, 컬럼 유형별로 다른 전처리가 필요하다.

- Numeric features: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
  - `SimpleImputer(strategy="median")`
  - `StandardScaler` 또는 `MinMaxScaler`
- Categorical features: `Sex`, `Embarked`
  - `SimpleImputer(strategy="most_frequent")`
  - `OneHotEncoder(handle_unknown="ignore")`

이를 `ColumnTransformer`로 결합하고, 이후 모델과 함께 `Pipeline`으로 구성하였다.

## 4. Experiments
다음 4개의 실험을 수행하였다.

1. `SVM_no_scaling`
2. `StandardScaler_SVM`
3. `StandardScaler_KNN`
4. `MinMaxScaler_KNN`

## 5. Evaluation Metrics
- Accuracy
- F1-score
- Confusion Matrix

## 6. Results

| Experiment | Accuracy | F1-score |
|---|---:|---:|
| StandardScaler_KNN | 0.8156 | 0.7442 |
| StandardScaler_SVM | 0.8156 | 0.7360 |
| MinMaxScaler_KNN | 0.8101 | 0.7258 |
| SVM_no_scaling | 0.6201 | 0.3200 |

## 7. Interpretation
- `SVM_no_scaling`의 성능이 크게 낮게 나왔다.
- `StandardScaler`를 적용한 SVM과 KNN은 모두 성능이 크게 향상되었다.
- 이번 설정에서는 `StandardScaler + KNN`이 가장 높은 F1-score를 기록했다.
- 따라서 Titanic 데이터에서는 전처리, 특히 scaling 여부가 모델 성능에 큰 영향을 준다는 점을 확인할 수 있었다.

## 8. Project Structure
```text
Assignment2/
├── data/
│   └── titanic.csv
├── src/
│   └── assignment2_baseline.py
├── results/
│   ├── metrics.csv
│   └── confusion_matrices.txt
├── README.md
├── report.md
├── requirements.txt
└── .gitignore