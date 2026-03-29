# Assignment 2 Report
## Titanic Baseline Modeling & Versioning

## 1. 데이터 개요
이번 과제에서는 Titanic 데이터셋을 이용해 생존 여부(`Survived`)를 예측하는 분류 문제를 수행하였다.  
Titanic 데이터는 수치형 변수와 범주형 변수가 혼합되어 있으며, `Age`, `Embarked`, `Cabin` 등 일부 컬럼에는 결측치가 존재한다.  
따라서 단순히 모델만 학습하는 것이 아니라, 실제 데이터 구조에 맞는 전처리 설계가 필요했다.

## 2. 전처리 구조
Titanic 데이터는 컬럼 유형별로 필요한 전처리가 다르기 때문에 `ColumnTransformer`를 사용하였다.

### 2.1 수치형 컬럼
- `Pclass`
- `Age`
- `SibSp`
- `Parch`
- `Fare`

수치형 컬럼에는 다음 전처리를 적용하였다.
- 결측치 대체: `SimpleImputer(strategy="median")`
- 스케일링: `StandardScaler` 또는 `MinMaxScaler`

### 2.2 범주형 컬럼
- `Sex`
- `Embarked`

범주형 컬럼에는 다음 전처리를 적용하였다.
- 결측치 대체: `SimpleImputer(strategy="most_frequent")`
- 인코딩: `OneHotEncoder(handle_unknown="ignore")`

### 2.3 Pipeline 사용 이유
전처리와 모델을 하나의 `Pipeline`으로 묶으면 학습과 예측 과정에서 동일한 순서가 일관되게 적용된다.  
또한 train/test split 이후의 전처리 흐름이 자동으로 관리되어 데이터 누수 가능성을 줄일 수 있다.

## 3. 실험 설계
다음 4가지 baseline 실험을 수행하였다.

1. **SVM (No Scaling)**
2. **StandardScaler + SVM**
3. **StandardScaler + KNN**
4. **MinMaxScaler + KNN**

실험의 목적은 단순히 어떤 모델이 더 높은 점수를 내는지 확인하는 것이 아니라,  
스케일링 여부와 전처리 구조가 성능에 어떤 영향을 주는지 비교하는 데 있다.

## 4. 성능 비교 결과

| Experiment | Accuracy | F1-score |
|---|---:|---:|
| StandardScaler_KNN | 0.8156 | 0.7442 |
| StandardScaler_SVM | 0.8156 | 0.7360 |
| MinMaxScaler_KNN | 0.8101 | 0.7258 |
| SVM_no_scaling | 0.6201 | 0.3200 |

## 5. 결과 해석

### 5.1 Scaling의 영향
가장 눈에 띄는 결과는 `SVM_no_scaling`의 성능이 매우 낮게 나온 점이다.  
Accuracy는 0.6201, F1-score는 0.3200으로 다른 실험보다 크게 낮았다.  
이는 SVM이 feature scale에 민감하며, 스케일링이 되지 않은 상태에서는 decision boundary를 안정적으로 학습하지 못했기 때문으로 해석할 수 있다.

반면 `StandardScaler + SVM`은 Accuracy 0.8156, F1-score 0.7360으로 성능이 크게 향상되었다.  
즉 SVM에서는 scaling이 사실상 필수적인 전처리임을 확인할 수 있었다.

### 5.2 KNN에서의 scaling 효과
KNN 역시 거리 기반 알고리즘이므로 스케일링의 영향을 크게 받는다.  
`StandardScaler + KNN`은 Accuracy 0.8156, F1-score 0.7442로 가장 좋은 결과를 보였다.  
`MinMaxScaler + KNN`도 나쁘지 않았지만 F1-score는 0.7258로 StandardScaler보다 다소 낮았다.

즉 KNN에서도 scaling은 필수적이었으며, 이번 데이터와 설정에서는 StandardScaler가 더 적합했다.

### 5.3 Confusion Matrix 관점 해석
`StandardScaler_KNN`의 confusion matrix는 `[[98, 12], [21, 48]]`이었다.  
이는 실제 생존자 클래스(1)에 대해 48명을 맞추고 21명을 놓친 것이다.  
반면 `StandardScaler_SVM`은 `[[100, 10], [23, 46]]`으로 비생존자 분류는 조금 더 강했지만, 생존자 recall 측면에서는 KNN이 약간 더 나았다.

따라서 이번 과제에서는 단순 Accuracy만 아니라 F1-score와 confusion matrix까지 함께 보는 것이 중요했다.

## 6. KNN vs SVM 비교
KNN과 SVM은 모두 좋은 성능을 보였지만 해석 관점은 다르다.

- **KNN**
  - 장점: 직관적이고 local similarity를 반영한다.
  - 단점: 거리 계산에 민감하여 scaling이 매우 중요하다.
- **SVM**
  - 장점: clear boundary를 잘 학습할 수 있고 baseline에서 안정적이다.
  - 단점: scaling이 없으면 성능이 크게 저하될 수 있다.

이번 데이터에서는 `StandardScaler + KNN`이 가장 높은 F1-score를 보였으므로 baseline으로 가장 적합하다고 판단하였다.  
다만 `StandardScaler + SVM`도 매우 근접한 성능을 보였기 때문에, Titanic 데이터에서는 두 모델 모두 유효한 baseline 후보라고 볼 수 있다.

## 7. MLOps / Git-DVC 회고
이번 과제에서는 `.py` 스크립트 기반으로 재현 가능한 실험 구조를 만들고, 결과를 `metrics.csv`와 `confusion_matrices.txt`로 저장하였다.  
이 방식은 동일한 코드와 동일한 데이터로 다시 실행했을 때 같은 결과를 재현할 수 있다는 점에서 의미가 있다.

Week 3 수업에서 배운 것처럼, MLOps에서 버전관리는 단순히 파일을 저장하는 것이 아니라 특정 시점의 **상태(state)**를 기록하는 것이다.  
Git은 코드와 문서 버전을 관리하는 데 유용하고, DVC는 데이터 버전과 중간 산출물 추적에 더 적합하다.  
이번 Titanic 과제에서는 DVC를 필수로 적용하지는 않았지만, 데이터가 자주 변경되거나 대용량 데이터셋을 사용하는 프로젝트에서는 Git + DVC 구조가 더 적합할 것이다.

## 8. 결론
이번 과제의 핵심은 단순 모델 점수 비교가 아니라,  
실제 혼합형 데이터에 대해 적절한 전처리 구조를 설계하고 이를 Pipeline으로 재현 가능하게 관리하는 것이었다.

실험 결과:
- No Scaling SVM은 성능이 크게 낮았다.
- StandardScaler를 적용한 SVM과 KNN은 모두 성능이 크게 향상되었다.
- 이번 설정에서는 `StandardScaler + KNN`이 가장 좋은 baseline 결과를 보였다.

따라서 Titanic 데이터에서는 `ColumnTransformer + Pipeline` 구조가 적절했고,  
스케일링은 KNN과 SVM 모두에서 매우 중요한 요소임을 확인할 수 있었다.