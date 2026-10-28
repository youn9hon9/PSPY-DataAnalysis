# 5주차 — Classification

## 이번 주 목표

1~4주차에는 숫자를 예측하는 회귀를 다뤘습니다. 이번 주부터는 대출 승인 여부처럼 정해진 범주를 예측하는 **분류(classification)** 로 넘어갑니다.

- 혼동행렬과 accuracy, precision, recall, F1을 손으로 계산하고 각 지표가 어떤 실수를 보는지 이해합니다.
- 로지스틱 회귀가 점수를 확률로 바꾸는 과정과 **분류 임계값(threshold)** 의 역할을 이해합니다.
- 불균형 데이터에서 accuracy만으로는 성능을 충분히 판단하기 어려운 이유를 확인합니다.
- ROC/PR 곡선과 확률 보정(calibration)이 답하는 질문을 구분합니다.
- 전처리 누수를 예방하고, 검증 데이터와 테스트 데이터의 역할을 구분합니다.

## 1. 분류 평가지표 — 원리부터 확인합니다

### 1.1 먼저 양성(positive)을 정합니다

10명의 합격 여부를 예측한다고 가정합니다. 여기서는 **합격을 양성(1)**, 불합격을 음성(0)으로 정합니다. 실제 합격자는 6명이고 불합격자는 4명이며, 모델의 일부 예측은 실제 결과와 달랐습니다. 그 결과를 **Confusion Matrix**(혼동행렬)라는 2×2 표로 정리합니다.

| | 예측: 합격(1) | 예측: 불합격(0) |
|---|---|---|
| **실제: 합격(1)** | TP — 맞게 합격으로 예측 | FN — 합격인데 불합격으로 예측한 놓침 |
| **실제: 불합격(0)** | FP — 불합격인데 합격으로 예측한 오탐 | TN — 맞게 불합격으로 예측 |

다음 코드를 실행해 혼동행렬의 네 칸과 주요 지표를 직접 계산합니다.

```python
import pandas as pd
from sklearn.metrics import confusion_matrix

toy = pd.DataFrame({
    "실제": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    "예측": [1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
})

cm = confusion_matrix(toy["실제"], toy["예측"])
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * precision * recall / (precision + recall)

print(cm)
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"accuracy={accuracy:.3f}, precision={precision:.3f}, "
      f"recall={recall:.3f}, f1={f1:.3f}")
```

대표 출력은 다음과 같습니다.

```text
[[3 1]
 [2 4]]
TP=4, FP=1, FN=2, TN=3
accuracy=0.700, precision=0.800, recall=0.667, f1=0.727
```

각 지표는 다음 질문에 답합니다.

- **accuracy** = $\frac{TP+TN}{\text{전체}} = 0.700$이며, 전체 중 맞힌 비율을 나타냅니다.
- **precision** = $\frac{TP}{TP+FP} = 0.800$이며, 양성이라고 예측한 것 중 실제 양성의 비율을 나타냅니다.
- **recall** = $\frac{TP}{TP+FN} = 0.667$이며, 실제 양성 중 모델이 찾아낸 비율을 나타냅니다.
- **F1** = $\frac{2 \times (precision \times recall)}{precision+recall} = 0.727$이며, precision과 recall의 조화평균을 나타냅니다.

recall이 precision보다 낮은 이유는 실제 합격자 6명 중 2명을 불합격으로 놓쳤기 때문입니다. 어느 오류의 영향이 더 큰지는 문제에 따라 달라집니다. 암 진단에서는 환자를 놓치는 FN의 영향이 특히 클 수 있고, 스팸 필터에서는 정상 메일을 스팸으로 처리하는 FP도 큰 비용이 될 수 있습니다.

![실제값을 행, 예측값을 열로 둔 합격 예제 혼동행렬. TN 3, FP 1, FN 2, TP 4.](assets/toy_confusion_matrix.png)

*그림 5-1. 실제값을 행, 예측값을 열로 둔 합격 예제 혼동행렬입니다. `practice.ipynb` Part 1의 `toy = pd.DataFrame(...)`와 `cm = confusion_matrix(...)` 셀을 사용했습니다.*

> **그림 읽기 질문:** 합격을 양성으로 두었을 때 놓친 합격자는 어느 칸이며 몇 명입니까?

> 💡 **함께 확인할 점:** TP, FP, FN, TN은 양성 클래스를 무엇으로 정했는지에 따라 이름이 바뀝니다. 분석을 시작할 때 “이 장에서 양성은 무엇입니까?”를 먼저 적어두면 뒤의 지표를 더 편안하게 해석할 수 있습니다.

### 1.2 업무 질문에 맞는 지표를 고릅니다

| 알고 싶은 것 | 특히 경계하는 실수 | 먼저 확인할 지표 |
|---|---|---|
| 전체적으로 얼마나 맞혔는지 | 두 클래스의 비용이 비슷한 상황 | accuracy |
| 양성이라고 한 판단을 얼마나 믿을 수 있는지 | FP(오탐) | precision |
| 실제 양성을 얼마나 놓치지 않았는지 | FN(놓침) | recall |
| precision과 recall을 한 수치로 절충하고 싶은지 | FP와 FN 모두 | F1 |
| 소수 클래스를 포함해 클래스별 성능을 고르게 보고 싶은지 | 다수 클래스 편향 | 클래스별 지표, macro F1 |

F1이 모든 문제에 가장 알맞은 지표인 것은 아닙니다. FN 비용이 FP보다 열 배 큰 문제라면 두 오류를 대칭적으로 절충하는 F1보다 recall이나 실제 비용을 반영한 지표가 더 적절합니다. **모델을 학습하기 전에 업무상 오류 비용을 먼저 정의하면 목적에 맞는 지표를 선택하기 쉬워집니다.**

---

## 2. 로지스틱 회귀가 확률을 만드는 방법

이름에는 Regression이 들어가지만 로지스틱 회귀는 대표적인 분류 모델입니다. 먼저 입력 변수의 가중합으로 점수 $z$를 만듭니다.

$$
z = b_0 + b_1x_1 + b_2x_2 + \cdots + b_px_p
$$

$z$는 음의 무한대부터 양의 무한대까지 갈 수 있으므로 그대로는 확률이 아닙니다. 이를 **시그모이드(sigmoid)** 함수에 넣어 0과 1 사이 값으로 바꿉니다.

$$
P(y=1 \mid X) = \sigma(z) = \frac{1}{1+e^{-z}}
$$

- $z=0$이면 확률은 0.5입니다.
- $z$가 커질수록 양성일 확률은 1에 가까워집니다.
- $z$가 작아질수록 양성일 확률은 0에 가까워집니다.

![시그모이드 확률곡선과 0.3, 0.5, 0.7 임계값 및 각 경계의 선형 점수.](assets/sigmoid_thresholds.png)

*그림 5-2. 선형 점수를 0~1 사이 확률로 바꾸는 시그모이드 곡선입니다. 기본 학습에서는 S자 곡선과 `z=0`일 때 확률 0.5라는 점만 확인하고, 세 임계값의 차이는 확장 학습에서 살펴봅니다.*

> **기본 그림 읽기 질문:** 선형 점수 `z`가 음수에서 양수로 커질 때 양성 확률은 어느 방향으로 움직입니까?

`practice.ipynb` Part 6에서 학습한 모델은 다음처럼 양성 확률과 최종 클래스를 만듭니다.

```python
model.fit(X_train, y_train)

proba_y = model.predict_proba(X_val)[:, 1]
pred_050 = (proba_y >= 0.50).astype(int)
```

### 2.1 0.5는 자연법칙이 아니라 선택한 기준입니다

`model.predict()`는 이진분류에서 보통 확률 0.5를 기준으로 클래스를 나눕니다. 그러나 0.5가 반드시 업무상 최적인 것은 아닙니다.

- 임계값을 **낮추면** 양성 예측이 많아집니다. 일반적으로 recall은 올라가고 precision은 내려갈 수 있습니다.
- 임계값을 **높이면** 양성 예측이 적어집니다. 일반적으로 precision은 올라가고 recall은 내려갈 수 있습니다.

대출 예제에서는 Y(승인)를 1로 둡니다. 승인 임계값을 0.5에서 0.7로 높이면 승인에 더 확신이 있는 신청자만 Y로 분류하므로, 거절해야 할 신청자를 잘못 승인하는 사례를 줄일 가능성이 있습니다. 대신 실제 우량 신청자를 거절하는 사례가 늘어날 수 있습니다.

임계값은 검증 데이터에서 업무 비용과 precision-recall의 균형을 보고 선택합니다. **선택에 사용한 검증 데이터에서 최종 성능까지 보고하면 낙관적인 평가가 될 수 있으므로**, 마지막에는 모델이나 임계값 선택에 사용하지 않은 테스트 데이터가 필요합니다.

> **확장 질문:** 임계값을 0.7로 높이면 어떤 신청자가 새로 거절되며, FP와 FN은 일반적으로 어느 방향으로 움직입니까?

---

## 3. ROC/PR 곡선과 확률 보정을 구분합니다

한 임계값의 혼동행렬은 실제 운영 지점을 평가합니다. 반면 ROC와 PR 곡선은 임계값을 0부터 1까지 움직였을 때 성능이 어떻게 변하는지 보여줍니다.

| 도구 | 답하는 질문 | 특히 유용한 상황 | 주의할 점 |
|---|---|---|---|
| ROC-AUC | 양성을 음성보다 높은 점수로 순위화합니까? | 클래스가 심하게 쏠리지 않았거나 전반적 구분력을 비교할 때 | 희소 양성 문제에서 좋아 보일 수 있으며 운영 임계값을 정해주지 않습니다. |
| PR 곡선 / Average Precision | recall을 높일 때 precision이 얼마나 유지됩니까? | 지연·이상 탐지처럼 관심 있는 양성이 드물 때 | 어떤 클래스를 양성으로 두었는지 밝혀야 합니다. |
| Calibration curve | 예측확률과 실제 발생 비율이 일치합니까? | 확률 자체로 의사결정하거나 비용을 계산할 때 | AUC가 높아도 확률 보정은 나쁠 수 있습니다. |

예를 들어 확률 0.8을 받은 신청자 100명 중 실제 승인 대상이 약 80명이라면 확률이 잘 보정되었다고 말할 수 있습니다. 반면 AUC는 확률값 자체가 맞는지보다 **순서를 잘 매기는지**에 가깝습니다.

분석 목적을 정하지 않은 채 여러 지표 중 가장 높아 보이는 값만 고르면 결과를 일관되게 해석하기 어렵습니다. 항공편 지연처럼 지연 비율이 17.6%라면 지연을 양성으로 두고, “지연을 놓치지 않는 것이 중요합니까, 지연 경보를 믿을 수 있는 것이 중요합니까?”를 먼저 정해 보는 것이 좋습니다.

---

## 4. 오늘 다룰 데이터 — Loan Prediction

614명의 대출 신청자 정보로 대출 승인 여부(`Loan_Status`: Y/N)를 예측합니다. 제공 데이터에는 승인 Y가 422건, 거절 N이 192건 있습니다. 이 예제에서는 **Y(승인)=1**로 정의하므로 별도 설정 없이 계산한 precision과 recall은 승인(Y)을 기준으로 합니다.

> 📥 **예제 데이터 다운로드**  
> [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) 페이지에서 원본 데이터를 내려받을 수 있습니다. Kaggle에 로그인한 뒤 **Download**를 선택합니다. 압축을 푼 `train_u6lujuX_CVtuZ9i.csv`를 아래 코드에 적힌 경로에 두면 준비된 셀을 그대로 실행할 수 있습니다.

---

## 5. 노트북을 순서대로 실행합니다

### Part 2 — 타깃 분포를 확인합니다

다음 셀은 원본 데이터의 크기와 타깃 비율을 확인합니다.

```python
df = pd.read_csv(
    "../dataset/open/extracted/Loan Prediction Problem Dataset/"
    "train_u6lujuX_CVtuZ9i.csv"
)

print(df.shape)
print(df["Loan_Status"].value_counts())
print(df["Loan_Status"].value_counts(normalize=True).round(3))
```

대표 출력은 다음과 같습니다.

```text
(614, 13)
Loan_Status
Y    422
N    192

Loan_Status
Y    0.687
N    0.313
```

승인(Y) 68.7%, 거절(N) 31.3%로 쏠려 있습니다. 이 비율은 뒤에서 다수 클래스만 예측해도 얻는 accuracy의 기준이 됩니다. 전체 데이터의 분포를 탐색하는 것은 가능하지만, 테스트 라벨을 보면서 모델이나 임계값을 바꾸면 테스트 데이터가 공정한 최종 평가용으로 남지 않습니다.

![대출 승인 Y 422건 68.7퍼센트와 거절 N 192건 31.3퍼센트의 불균형 분포.](assets/loan_target_distribution.png)

*그림 5-3. 승인 Y 422건(68.7%)과 거절 N 192건(31.3%)의 타깃 분포입니다. `practice.ipynb` Part 2의 `df['Loan_Status'].value_counts(normalize=True)` 셀과 같은 원본 데이터를 사용했습니다.*

> **그림 읽기 질문:** 모든 신청자를 Y로 예측하면 왜 0.687의 accuracy를 얻습니까?

### Part 3 — 결측치를 확인하고 전처리 대상을 지정합니다

노트북에서는 결측치의 위치와 개수를 먼저 확인하고, 범주형 변수와 수치형 변수를 나눠 적습니다. 이 시점에는 아직 값을 채우지 않습니다. 결측이 특정 집단이나 결과에 몰려 있는지도 함께 살펴보면 좋습니다. 결측치 처리법도 모델의 일부이기 때문입니다.

### Part 4 — 원본 입력과 타깃을 준비합니다

`Loan_Status`를 0/1로 바꾸되, 입력 X는 결측치가 남은 원본 열로 준비합니다. 범주형 변수의 one-hot 인코딩과 결측치 대체는 데이터를 나눈 다음 Pipeline 안에서 수행합니다.

### Part 5 — 학습 데이터와 검증 데이터를 분리합니다

`train_test_split(..., stratify=y)`를 사용하면 학습·검증 데이터의 클래스 비율이 전체와 비슷하게 유지됩니다. 제공 데이터와 고정된 시드에서는 다음과 같은 출력이 나옵니다.

```text
학습: (491, 11) 검증: (123, 11)
학습 타겟 비율: 0.686 검증 타겟 비율: 0.691
```

위의 11개 열은 Pipeline에 들어가기 전 원본 입력의 열 수입니다. Part 6에서 범주형 변수가 one-hot 열로 펼쳐지므로 모델이 실제로 받는 변환 행렬의 열 수는 더 많아집니다.

`stratify`는 **클래스 비율을 유지하는 옵션이며, 누수나 대표성까지 해결해 주지는 않습니다.** 같은 사람이 여러 행에 있다면 사람 단위 분할을 사용하고, 미래를 예측하는 문제라면 시간 순서 분할을 사용하면 실제 예측 상황에 더 가까운 평가가 됩니다.

### Part 3~6 — 전처리 누수를 예방합니다

현재 노트북은 다음 순서로 검증 데이터의 분포가 전처리 기준에 들어가지 않도록 구성되어 있습니다.

1. 원본 데이터를 먼저 학습 데이터와 검증 데이터로 나눕니다.
2. 중앙값, 최빈값, 스케일 같은 전처리 기준은 **학습 데이터에서만 학습(fit)** 합니다.
3. 학습한 기준으로 검증 데이터에는 변환(transform)만 수행합니다.
4. `Pipeline`과 `ColumnTransformer`를 사용하면 교차검증 안에서도 이 순서를 일관되게 적용할 수 있습니다.

로지스틱 회귀는 기본적으로 규제를 사용하므로 수치 Pipeline에는 `StandardScaler`도 포함합니다. 스케일이 다르면 규제가 변수마다 다르게 작용할 수 있고, 계수 크기도 단순 비교하기 어렵습니다. 이 데이터에서는 스케일링 뒤에도 아래 대표 혼동행렬과 지표가 같으며, 학습 과정의 수치적 안정성은 좋아집니다.

### Part 6~8 — 모델과 혼동행렬을 확인합니다

`practice.ipynb` Part 6~8의 핵심 실행 흐름은 다음과 같습니다.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

categorical_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')),
])
numeric_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler()),
])
preprocessor = ColumnTransformer([
    ('categorical', categorical_pipeline, categorical_features),
    ('numeric', numeric_pipeline, numeric_features),
])

model = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=5000)),
])
model.fit(X_train, y_train)
pred = model.predict(X_val)

cm = confusion_matrix(y_val, pred)
print(cm)
print("Accuracy :", accuracy_score(y_val, pred))
print("Precision:", precision_score(y_val, pred))
print("Recall   :", recall_score(y_val, pred))
print("F1       :", f1_score(y_val, pred))
```

제공 데이터와 `random_state=42`의 대표 출력은 다음과 같습니다.

```text
[[22 16]
 [ 1 84]]
Accuracy : 0.8617886178861789
Precision: 0.84
Recall   : 0.9882352941176471
F1       : 0.9081081081081082
```

scikit-learn의 `confusion_matrix(y_true, y_pred)`는 기본적으로 행이 실제, 열이 예측이며 배열 순서는 `[[TN, FP], [FN, TP]]`입니다. 따라서 실제 N 38건 중 16건을 Y로 예측했고, 실제 Y 85건 중 1건을 N으로 예측했습니다.

| 클래스 | precision | recall | 해석 |
|---|---:|---:|---|
| N(거절) | 0.96 | **0.58** | 실제 N 중 약 42%를 Y로 놓쳤습니다. |
| Y(승인) | 0.84 | 0.99 | 실제 Y는 거의 모두 찾았습니다. |

**Y를 양성으로 둔 전체 혼동행렬에서는 ‘실제 N, 예측 Y’가 FP입니다.** `classification_report`의 N 행에서는 N을 일대나머지 방식의 기준 클래스로 보므로 같은 사례를 ‘N을 놓친 경우’라고 설명할 수 있습니다. 실제 클래스와 예측 클래스를 함께 적으면 두 표현을 더 쉽게 구분할 수 있습니다.

대출 심사처럼 잘못된 승인이 큰 손실로 이어질 수 있다면 accuracy보다 N recall, Y precision 또는 승인 오류의 실제 비용이 중요할 수 있습니다. 다만 이 데이터의 `Loan_Status`는 과거 승인 결과이지 실제 상환 여부가 아닙니다. **승인 결과를 잘 예측하는 모델이 부도 위험을 잘 예측하는 모델은 아닙니다.**

![검증 데이터에서 실제 거절 38건 중 16건을 승인으로 예측한 로지스틱 회귀 혼동행렬.](assets/model_confusion_matrix.png)

*그림 5-4. 검증 데이터의 혼동행렬 `[[22, 16], [1, 84]]`입니다. `practice.ipynb` Part 7의 `cm = confusion_matrix(y_val, pred)` 셀을 재현했으며, 실제 N을 Y로 예측한 FP 칸을 빨간 테두리로 강조했습니다.*

> **그림 읽기 질문:** accuracy 86.2%만 보았을 때 놓치기 쉬운 오류는 어느 칸입니까?

### Part 9 — 다수 클래스 베이스라인과 비교합니다

```python
dummy_pred = [1] * len(y_val)

print("더미 accuracy :", accuracy_score(y_val, dummy_pred))
print("더미 precision:", precision_score(y_val, dummy_pred))
print("더미 recall   :", recall_score(y_val, dummy_pred))
print("모델 accuracy :", accuracy_score(y_val, pred))
```

대표 출력은 다음과 같습니다.

```text
더미 accuracy : 0.6910569105691057
더미 precision: 0.6910569105691057
더미 recall   : 1.0
모델 accuracy : 0.8617886178861789
```

항상 Y만 예측하는 모델도 accuracy 69.1%를 얻으며 Y recall은 100%입니다. 실제 모델의 accuracy는 약 17%p 높지만, 이 차이만으로 결론을 내리기보다 클래스별 오류와 업무 비용도 함께 비교하는 것이 좋습니다.

한 번의 8:2 분할 결과는 우연에 따라 달라질 수 있습니다. 모델 후보나 하이퍼파라미터를 비교한다면 **Stratified K-fold 교차검증의 평균과 변동성**을 확인하고, 최종 테스트 데이터는 마지막에 한 번만 평가하는 편이 타당합니다.

![항상 승인만 예측하는 더미 모델과 로지스틱 회귀의 accuracy, Y precision, Y recall, N recall 비교.](assets/model_metric_comparison.png)

*그림 5-5. `practice.ipynb` Part 9의 항상 승인하는 더미 모델과 로지스틱 회귀를 비교했습니다. 다수 클래스 지표만으로는 더미 모델의 한계를 발견하기 어려울 수 있어 N recall도 함께 표시했습니다.*

> **그림 읽기 질문:** 더미 모델의 Y recall 1.0이 좋은 모델을 뜻하지 않는 이유는 무엇입니까?

---

## 6. 임계값을 직접 바꾸어 봅니다

다음 코드를 `practice.ipynb` Part 8 뒤에 붙여 실행하고, Part 6에서 학습한 모델의 확률을 0.3, 0.5, 0.7에서 나누어 precision, recall과 혼동행렬을 비교합니다.

```python
from sklearn.metrics import precision_score, recall_score, confusion_matrix

proba_y = model.predict_proba(X_val)[:, 1]

for threshold in [0.3, 0.5, 0.7]:
    pred_t = (proba_y >= threshold).astype(int)
    print(
        f"threshold={threshold:.1f}",
        "precision=", round(precision_score(y_val, pred_t), 3),
        "recall=", round(recall_score(y_val, pred_t), 3),
        "cm=", confusion_matrix(y_val, pred_t).tolist(),
    )
```

이 출력에서 가장 좋아 보이는 숫자가 곧바로 최종 임계값이 되는 것은 아닙니다. 같은 검증 데이터로 임계값을 선택했다면 별도의 테스트 데이터에서 최종 성능을 평가하는 것이 좋습니다. 이렇게 하면 선택 과정의 낙관적 편향을 줄일 수 있습니다. 예측확률을 실제 위험도나 비용 계산에 사용하려면 calibration도 별도로 확인하는 것이 좋습니다.

---

## 7. 인사이트를 근거와 함께 씁니다

다음 문장 구조를 활용합니다.

> 무엇을 보호하려는지 설명합니다 → 양성 클래스를 정의합니다 → 지표와 임계값을 밝힙니다 → 남은 오류를 실제·예측 클래스로 씁니다 → 업무상 조정과 검증 계획을 제시합니다.

예시는 다음과 같습니다.

> “승인(Y)을 양성으로 두고 클래스별 recall을 확인했습니다. Y recall은 0.99였지만 N recall은 0.58이어서 실제 거절 대상의 상당수를 승인으로 분류했습니다. 잘못된 승인의 비용이 크다면 accuracy 0.86만으로 운영하지 않고 승인 임계값을 높인 후보를 검증하겠습니다. 임계값은 검증 데이터에서 선택하고, 최종 테스트 데이터에서는 N recall과 우량 신청자 거절 비용을 함께 확인하겠습니다.”

상관관계와 과거 승인 패턴을 학습한 결과는 인과관계나 공정성의 증거와 구분할 필요가 있습니다. 실제 대출 의사결정에 사용한다면 성별·혼인 여부 등 민감할 수 있는 변수에 대해 집단별 오류율과 법적·윤리적 요건을 별도로 검토하는 것이 중요합니다. 과거의 편향이 모델을 통해 반복될 수 있기 때문입니다.

---

## 8. 학습하면서 자주 확인할 점

1. **precision과 recall을 보고할 때 양성 클래스를 함께 적습니다.**  
   “지연(1) recall”처럼 기준 클래스를 함께 적으면 지표의 의미가 분명해집니다.
2. **accuracy는 다수 클래스 기준선과 클래스별 지표를 함께 봅니다.**  
   이 비교를 통해 높은 accuracy가 실제로 어느 오류를 줄였는지 확인할 수 있습니다.
3. **데이터를 먼저 나눈 뒤 전처리를 학습 데이터에만 fit합니다.**  
   이 순서를 지키면 검증 정보가 학습 과정에 들어가는 누수를 예방할 수 있습니다.
4. **선택용 검증과 최종 평가용 테스트를 구분합니다.**  
   중첩 교차검증을 활용하는 방법도 있으며, 같은 점수로 선택과 최종 평가를 모두 하면 성능이 낙관적으로 보일 수 있습니다.
5. **ROC-AUC와 운영 임계값의 혼동행렬을 따로 확인합니다.**  
   AUC는 임계값 전반의 순위 능력이므로 임계값 0.5의 성능을 직접 보장하지는 않습니다.
6. **예측확률을 의사결정에 사용할 때 calibration을 확인합니다.**  
   이렇게 하면 예측확률 0.8이 실제 발생 비율 80%와 비슷한지 확인할 수 있습니다.

---

## 9. 과제 안내 — 항공편 지연 데이터

[항공편 지연 예측 AI 경진대회](https://dacon.io/competitions/official/236094)에서 제공하는 미국 항공편 데이터입니다. 한 행은 항공편 하나이며, 목표 변수 `Delay`는 `Delayed`와 `Not_Delayed`로 지연 여부를 나타냅니다.

원본 `train.csv`는 100만 행입니다. 약 74.5%의 행에는 지연 여부 정답이 없으며, 정답이 있는 행은 약 25.5만 개입니다. 정답이 있는 데이터의 지연 비율은 약 17.6%로 클래스가 불균형합니다. 노트북에서는 `Delayed`를 1, `Not_Delayed`를 0으로 표현합니다.

항공편의 출발·도착 지역 등 운항 정보와 식별자 성격의 `ID`, `Tail_Number`가 포함되어 있습니다. 파일 위치는 `dataset/open/extracted/월간 데이콘 항공편 지연 예측 AI 경진대회/train.csv`입니다.

## 10. 5주차 요약

- 혼동행렬은 예측 오류를 TP, FP, FN, TN으로 나누며, 네 이름은 양성 클래스에 따라 달라집니다.
- accuracy, precision, recall, F1은 서로 다른 질문에 답하므로 업무상 오류 비용을 기준으로 선택하는 것이 적절합니다.
- 로지스틱 회귀는 선형 점수를 sigmoid로 변환해 양성 확률을 만들고, 임계값으로 클래스를 나눕니다.
- 0.5는 기본값일 뿐이며, 임계값을 바꾸면 precision과 recall의 균형이 달라집니다.
- ROC/PR은 임계값 전반의 구분력을 보여주고, calibration은 확률값 자체의 신뢰도를 보여줍니다.
- `stratify`는 클래스 비율만 유지하며 전처리 누수를 막아주지 않습니다.
- 모델과 임계값을 선택하는 데이터와 최종 성능을 보고하는 데이터를 구분하면 평가의 신뢰도를 지킬 수 있습니다.
