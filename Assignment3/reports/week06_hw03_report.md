# Week06 HW03 Report

## 1. 데이터셋 소개

- **데이터셋 이름:** Kaggle Titanic
- **문제 유형:** Binary Classification
- **Target 변수:** `Survived`

Titanic 데이터셋은 승객의 정보(성별, 나이, 객실 등급, 요금 등)를 바탕으로 생존 여부를 예측하는 이진 분류 문제이다.  
수치형 변수와 범주형 변수가 함께 존재하고 일부 결측치가 포함되어 있어, Tree 계열 및 Boosting 계열 모델을 비교하기에 적절한 tabular 데이터셋이다.

---

## 2. 실험 구조

### 2.1 데이터 분할 방식
원본 `train.csv`를 다음과 같이 다시 분할하여 실험하였다.

- Train: 모델 학습
- Validation: 하이퍼파라미터 비교 및 Early Stopping 기준
- Test: 최종 일반화 성능 평가

분류 문제이므로 클래스 비율 유지를 위해 `stratify`를 적용하였다.

### 2.2 전처리
다음과 같은 최소 전처리를 적용하였다.

- 수치형 변수: median imputation
- 범주형 변수: most frequent imputation + one-hot encoding

Tree 계열 모델은 스케일링에 덜 민감하므로 별도의 scaling은 적용하지 않았다.

### 2.3 평가 지표
분류 문제 평가를 위해 다음 지표를 사용하였다.

- Accuracy
- F1-score
- ROC-AUC

이번 과제에서는 특히 **train 성능보다 validation 성능과 test 성능을 함께 보는 구조**를 중요하게 두었다.

---

## 3. 모델 비교 결과

### 3.1 Baseline 비교

| Model | Train Acc | Valid Acc | Test Acc | Train F1 | Valid F1 | Test F1 | Train ROC-AUC | Valid ROC-AUC | Test ROC-AUC |
|------|----------:|----------:|---------:|---------:|---------:|--------:|--------------:|---------------:|-------------:|
| XGBoost | 0.9244 | 0.8531 | 0.8101 | 0.8959 | 0.8037 | 0.7344 | 0.9798 | 0.8815 | 0.8159 |
| RandomForest | 0.9807 | 0.8392 | 0.7710 | 0.9748 | 0.7810 | 0.7050 | 0.9988 | 0.8439 | 0.8208 |
| LightGBM | 0.9631 | 0.8322 | 0.7933 | 0.9506 | 0.7692 | 0.7176 | 0.9960 | 0.8668 | 0.8233 |

Baseline 비교 결과, **XGBoost가 validation accuracy, validation F1, validation ROC-AUC 기준으로 가장 우수한 모델**이었다.

---

## 4. XGBoost 튜닝

이번 과제에서는 XGBoost에 대해 다음 하이퍼파라미터를 조정하였다.

- `learning_rate`
- `n_estimators`
- `max_depth`

### 4.1 튜닝 결과

| Model | learning_rate | n_estimators | max_depth | Valid Acc | Valid F1 | Valid ROC-AUC | Test Acc | Test F1 | Test ROC-AUC |
|------|--------------:|-------------:|----------:|----------:|---------:|---------------:|---------:|--------:|-------------:|
| XGB_tuned_2 | 0.05 | 300 | 4 | 0.8531 | 0.8037 | 0.8815 | 0.8101 | 0.7344 | 0.8159 |
| XGB_tuned_3 | 0.05 | 500 | 3 | 0.8462 | 0.7925 | 0.8817 | 0.8156 | 0.7442 | 0.8114 |
| XGB_tuned_1 | 0.03 | 300 | 3 | 0.8322 | 0.7692 | 0.8879 | 0.7989 | 0.7143 | 0.8125 |
| XGB_tuned_4 | 0.10 | 300 | 3 | 0.8322 | 0.7736 | 0.8732 | 0.8045 | 0.7287 | 0.8138 |
| XGB_tuned_5 | 0.05 | 500 | 4 | 0.8252 | 0.7706 | 0.8720 | 0.8101 | 0.7385 | 0.8186 |

Validation accuracy를 기준으로 보면 `learning_rate=0.05, n_estimators=300, max_depth=4` 조합이 가장 좋았다.  
따라서 이를 최종 후보 조합으로 선택하였다.

---

## 5. Early Stopping 적용

선택한 XGBoost 조합에 대해 Early Stopping을 적용하여, validation 성능 개선이 멈추는 시점에서 학습을 조기 종료하도록 하였다.

### 5.1 적용 전후 비교

| Model | Valid Acc | Valid F1 | Valid ROC-AUC | Test Acc | Test F1 | Test ROC-AUC | Best Iteration |
|------|----------:|---------:|---------------:|---------:|--------:|-------------:|---------------:|
| XGBoost_no_early_stopping | 0.8531 | 0.8037 | 0.8815 | 0.8101 | 0.7344 | 0.8159 | - |
| XGBoost_with_early_stopping | 0.8462 | 0.7925 | 0.8918 | 0.8156 | 0.7402 | 0.8116 | 150 |

Early Stopping 적용 결과, validation accuracy와 validation F1은 약간 감소했지만, validation ROC-AUC는 상승하였다.  
또한 train 성능이 낮아져 과적합이 완화되는 신호를 보였다.  
즉, Early Stopping은 단순히 점수를 무조건 올리는 장치라기보다, **모델 복잡도를 제어하고 일반화 성능을 더 안정적으로 관리하는 장치**로 해석할 수 있다.

---

## 6. 분석 코멘트

### 6.1 Random Forest와 Boosting의 차이
Random Forest는 여러 트리를 병렬로 학습하는 Bagging 계열 모델로, 안정적인 baseline 역할을 수행했다.  
하지만 이번 실험에서는 train 성능이 매우 높고 validation/test 성능과의 차이가 커서 과적합 경향이 상대적으로 강하게 나타났다.

반면 XGBoost와 LightGBM은 이전 단계의 오차를 순차적으로 보완하는 Boosting 계열 모델로, validation 기준에서 더 강한 예측 성능을 보였다.  
즉, 이번 결과에서는 **Boosting 계열이 더 높은 일반화 성능을 달성할 가능성이 크다**는 점을 확인할 수 있었다.

### 6.2 XGBoost와 LightGBM 비교
두 모델 모두 강력한 성능을 보였지만, 이번 실험에서는 XGBoost가 validation accuracy, validation F1, validation ROC-AUC에서 더 우수했다.  
따라서 이번 Titanic 실험에서는 XGBoost가 더 다루기 쉽고 안정적인 모델로 느껴졌다.

### 6.3 왜 train 점수보다 validation 구조가 중요한가
Random Forest의 train accuracy는 0.9807로 가장 높았지만, validation accuracy는 0.8392로 XGBoost보다 낮았다.  
이 결과는 train 점수만 보면 모델이 좋아 보일 수 있지만, 실제 모델 선택에서는 validation 성능이 훨씬 중요하다는 점을 보여준다.  
특히 Boosting 계열은 성능이 강력한 만큼 과적합 제어가 중요하므로, validation 구조가 필수적이다.

### 6.4 Early Stopping의 효과
Early Stopping은 validation 성능이 더 이상 개선되지 않을 때 학습을 멈추게 하여 과적합을 줄이는 역할을 한다.  
이번 실험에서는 validation accuracy는 약간 감소했지만 validation ROC-AUC가 개선되었고, train 성능이 낮아져 과적합 제어 효과를 확인할 수 있었다.

---

## 7. 최종 모델 선택

최종 모델은 **XGBoost (`learning_rate=0.05, n_estimators=300, max_depth=4`)** 로 선택하였다.

선택 이유는 다음과 같다.

1. Validation accuracy가 가장 높았다.
2. Validation F1도 가장 높아 분류 성능이 안정적이었다.
3. Baseline 3개 모델 중 가장 우수한 validation 성능을 보였다.
4. Train 성능만 높은 Random Forest보다 일반화 관점에서 더 적절했다.

다만 Early Stopping 적용 결과도 의미가 있었기 때문에, 실제 실무 상황이라면 metric 우선순위에 따라 Early Stopping 모델을 선택할 수도 있다.

---

## 8. 결론

이번 과제를 통해 같은 tabular 데이터셋에서도 Bagging 계열과 Boosting 계열이 다른 성격을 가진다는 점을 확인할 수 있었다.  
Random Forest는 안정적인 baseline 역할을 했지만, XGBoost와 LightGBM 같은 Boosting 계열은 더 높은 validation 성능을 보였다.  
또한 Boosting 모델은 성능이 강한 만큼, 하이퍼파라미터 튜닝과 Early Stopping, 그리고 validation 구조가 매우 중요하다는 점을 실험적으로 확인할 수 있었다.