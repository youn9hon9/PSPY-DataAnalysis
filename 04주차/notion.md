# 4주차 — Random Forest와 모델 비교

## 이번 주 목표

2~3주차와 같은 따릉이 데이터를 이어서 사용합니다. 이번 주에는 Random Forest를 추가해 **선형회귀, 단일 트리, Random Forest를 같은 기준으로 비교합니다.**

본문 데이터는 [서울시 따릉이 일별 대여건수 페이지](https://data.seoul.go.kr/dataList/OA-14994/A/1/datasetView.do)에서 설명과 원본 파일을 확인할 수 있습니다. 필요한 기간의 `서울특별시 공공자전거 일별 대여건수_*.csv`를 내려받아 앞 주차와 같은 로컬 이용정보 경로에 두면 됩니다. 출처·다운로드 날짜·버전도 함께 기록해 둡니다.

- 단일 트리가 작은 데이터 변화에 민감한 이유를 설명합니다.
- 부트스트랩, 변수 부분집합, 평균이 각각 어떤 역할을 하는지 구분합니다.
- OOB(out-of-bag) 평가가 무엇이며, 언제 보조 평가로 활용할 수 있고 언제 다른 평가가 필요한지 이해합니다.
- Random Forest의 하이퍼파라미터를 검증 데이터에서 선택합니다.
- 불순도 기반 feature importance의 편향을 알고 대안을 사용합니다.
- 범주형 ID를 임의의 정수로 바꾸는 문제와 안전한 전처리 원칙을 익힙니다.

## 출발점 — 단일 트리는 왜 불안정합니까?

3주차의 기온→아이스크림 판매량 소규모 예제 데이터를 다시 사용합니다. 전체 6개 관측치로 학습한 트리와 기온 20도인 관측치 하나를 뺀 5개로 학습한 트리를 만듭니다.

```python
tree_all = DecisionTreeRegressor(
    max_depth=2, random_state=42
).fit(toy[['기온']], toy['판매량'])

toy_drop = toy.drop(index=3)  # 기온=20, 판매량=55인 점 하나를 제외합니다.
tree_drop = DecisionTreeRegressor(
    max_depth=2, random_state=42
).fit(toy_drop[['기온']], toy_drop['판매량'])

print('전체 6개로 학습:')
print(export_text(tree_all, feature_names=['기온']))
print('점 1개를 뺀 5개로 학습:')
print(export_text(tree_drop, feature_names=['기온']))
```

두 트리는 분기 기준과 잎의 평균이 달라집니다. 기온 22도에서 예측값을 비교하는 코드는 다음과 같습니다.

```python
print('기온=22일 때 예측값 비교')
print('전체 데이터로 학습한 트리:',
      tree_all.predict(pd.DataFrame({'기온': [22]}))[0])
print('점 1개 뺀 데이터로 학습한 트리:',
      tree_drop.predict(pd.DataFrame({'기온': [22]}))[0])
```

```text
기온=22일 때 예측값 비교
전체 데이터로 학습한 트리: 55.0
점 1개 뺀 데이터로 학습한 트리: 70.0
```

분기 알고리즘은 현재 노드에서 오차 감소가 가장 큰 기준을 고릅니다. 후보들의 감소량이 비슷할 때 관측치 하나만 달라져도 1등 후보가 바뀔 수 있고, 상위 분기가 바뀌면 아래의 모든 분기도 영향을 받습니다. 이렇게 **학습 데이터가 조금 달라졌을 때 모델의 예측이 크게 달라지는 성질**을 분산(variance)이 크다고 표현합니다.

> 🔄 **예측 → 실행 → 비교 → 해석 ①: 단일 트리의 흔들림**
>
> 1. 관측치 하나를 제외하면 기온 22도의 예측이 어느 방향으로 바뀔지 먼저 예상합니다.
> 2. `example.ipynb` Part 1의 `toy = pd.DataFrame(...)`와 예측 비교 셀을 실행합니다.
> 3. 두 트리의 첫 분기와 `55.0 → 70.0`의 변화량을 비교합니다.
> 4. 결과를 "한 점의 영향"에서 한 걸음 더 확장해, 탐욕적 상위 분기가 바뀌면서 전체 구조가 달라진 현상으로 해석합니다.

![기온 20도 관측치 포함 여부에 따른 깊이 2 회귀 트리의 분기와 잎 비교](assets/toy-tree-rules.png)

> **실행 결과:** 본문의 예제 데이터로 두 트리를 실제 학습한 뒤 `plot_tree`로 그렸습니다. 기온 22도의 예측값은 55.0에서 70.0으로 바뀝니다. 가장 먼저 달라지는 분기와 그 아래 잎을 비교해 봅니다.

같은 두 데이터셋으로 트리 100개의 Random Forest를 학습합니다.

```python
rf_all = RandomForestRegressor(
    n_estimators=100, max_depth=2, random_state=42
).fit(toy[['기온']], toy['판매량'])

rf_drop = RandomForestRegressor(
    n_estimators=100, max_depth=2, random_state=42
).fit(toy_drop[['기온']], toy_drop['판매량'])

print('전체 데이터 RF 예측(기온=22):',
      rf_all.predict(pd.DataFrame({'기온': [22]}))[0])
print('점 1개 뺀 RF 예측(기온=22) :',
      rf_drop.predict(pd.DataFrame({'기온': [22]}))[0])
```

같은 설정의 대표 실행 결과는 다음과 같습니다.

```text
전체 데이터 RF 예측(기온=22): 약 59.5
점 1개 뺀 RF 예측(기온=22) : 약 62.8
```

단일 트리의 변화량은 15.0이었지만 Random Forest의 변화량은 약 3.3입니다. 이 한 번의 비교만으로 분산 감소를 증명할 수는 없지만, 단일 트리보다 변화 폭이 작아지는 원리를 관찰할 수 있습니다. 더 엄밀하게 확인하려면 여러 번 표본을 다시 뽑아 예측값의 분포나 표준편차를 비교하는 과정이 필요합니다.

![관측치 하나를 제외했을 때 단일 트리와 Random Forest의 예측값 및 변화량](assets/toy-prediction-change.png)

> **실행 결과:** 같은 예제 데이터와 `random_state=42`로 실행한 결과입니다. 단일 트리의 변화량은 15.00, Random Forest의 변화량은 약 3.3입니다. 한 번의 실험이므로 이 결과만으로 일반적인 분산 감소를 증명할 수는 없습니다.

## Random Forest의 작동 원리

회귀용 Random Forest는 서로 조금씩 다른 회귀 트리를 만든 뒤 예측값을 평균합니다.

### 1단계 — 행을 다르게 뽑습니다: bootstrap

원본 학습 데이터가 (n)개라면 중복을 허용해 (n)번 뽑아 한 트리의 학습 표본을 만듭니다. 어떤 행은 여러 번 뽑히고 어떤 행은 한 번도 뽑히지 않습니다. 이 과정을 각 트리마다 새로 반복하므로 트리들이 서로 다른 데이터를 보게 됩니다.

표본 수가 충분히 크면 한 부트스트랩 표본에 한 번 이상 포함되는 고유 행은 원본의 약 63.2%이며, 포함되지 않는 행은 약 36.8%입니다. 포함되지 않은 행은 해당 트리의 **OOB 표본**이 됩니다.

### 2단계 — 분기 후보 변수를 제한할 수 있습니다: `max_features`

각 분기에서 전체 변수 중 무작위 부분집합만 후보로 보여주면 강한 변수 하나가 모든 트리를 지배하는 현상을 줄이고 트리 사이의 상관을 낮출 수 있습니다. 서로 덜 비슷한 트리의 오차를 평균할수록 분산 감소 효과가 커집니다.

다만 이 동작은 `max_features` 설정에 달려 있습니다. **현재 `example.ipynb`는 `max_features`를 명시하지 않습니다.** 최근 scikit-learn의 `RandomForestRegressor` 기본값 `max_features=1.0`은 각 분기에서 모든 변수를 후보로 봅니다. 따라서 현재 노트북의 결과는 주로 행 부트스트랩의 효과로 설명하는 편이 정확합니다. 변수 부분집합의 효과까지 관찰하고 패키지 버전에 따른 기본값 변화의 영향도 줄이려면 실습 코드에서 다음처럼 값을 명시할 수 있습니다.

```python
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    max_features=0.7,  # 각 분기에서 변수의 70%를 후보로 사용합니다.
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)
```

`max_features`를 작게 하면 트리 다양성은 커지지만 유용한 변수를 자주 놓쳐 각 트리의 편향이 커질 수 있습니다. 따라서 이 값도 검증 결과를 살펴 선택하는 것이 좋습니다. 위 코드는 원리를 분명히 하기 위한 후보일 뿐 정답 설정이 아니며, 기존 노트북의 수치와 달라질 수 있습니다.

> 🔄 **예측 → 실행 → 비교 → 해석 ②: 기본값을 코드에서 확인하기**
>
> 1. 현재 노트북 코드에서 `max_features`가 설정되었는지 먼저 찾아봅니다.
> 2. Part 1의 `rf_all`과 Part 3의 `rf_model`을 만드는 `RandomForestRegressor(...)` 인자를 확인합니다.
> 3. 자신의 환경에서 `rf_model.max_features`를 출력해 명시된 값과 기본값을 구분합니다.
> 4. 현재 결과를 행 부트스트랩과 변수 부분집합 효과 중 어디까지로 설명할 수 있는지 기록합니다.

### 3단계 — 예측을 평균합니다

트리 (B)개의 예측을 \(\hat{f}_1(x),\ldots,\hat{f}_B(x)\)라고 하면 Random Forest 회귀 예측은 다음과 같습니다.

\[
\hat{f}_{RF}(x)=\frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x)
\]

개별 트리의 흔들림이 완전히 같은 방향이 아니라면 평균 과정에서 일부가 상쇄됩니다. 그러나 모든 트리가 같은 편향을 가지면 평균으로 없앨 수 없습니다. 예를 들어 각 트리가 추세를 외삽하지 못하므로 그 평균인 Random Forest도 외삽하지 못합니다.

## OOB 평가는 언제 활용할 수 있습니까?

각 행은 자신을 뽑지 않은 여러 트리의 OOB 표본이 됩니다. 그 트리들의 예측만 평균해 행별 OOB 예측을 만들면 학습 데이터만으로도 일반화 오차를 대략 추정할 수 있습니다.

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

rf_oob = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=10,
    max_features=0.7,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1,
)
rf_oob.fit(X_train, y_train)

oob_rmse = root_mean_squared_error(y_train, rf_oob.oob_prediction_)
```

OOB는 유용한 보조 평가이지만 독립 테스트와는 역할이 다릅니다.

- 행들이 독립이고 같은 분포에서 왔다는 가정에 더 잘 맞습니다.
- **따릉이처럼 미래 예측이 목적인 시간 데이터에는 시간순 검증을 최종 평가로 사용하는 편이 적절합니다.** OOB에서는 미래 행이 부트스트랩 학습에 들어간 트리가 과거 행을 예측할 수 있어 실제 예측 상황의 시간 순서와 달라지기 때문입니다.
- 같은 고객·건물·지역의 중복 행이 있으면 그룹 정보가 다른 트리에 남아 점수가 낙관적일 수 있습니다.
- 트리 수가 너무 적으면 일부 행의 OOB 예측이 불안정할 수 있습니다.
- 기본 `oob_score_`는 R²입니다. RMSE 비교표와 혼동하지 않도록 같은 지표를 직접 계산해 비교합니다.

따릉이 예제에서는 마지막 기간을 시간순 홀드아웃으로 두고 OOB는 원리를 확인하는 보조 진단으로만 다룹니다. 이 홀드아웃으로 설정까지 고른다면 validation 역할을 하므로, 최종 성능을 따로 보고하려면 더 뒤의 기간을 test로 남겨야 합니다. 물류 과제도 실제 예측 대상이 새 지역인지 기존 지역의 새 운송인지에 따라 무작위 홀드아웃 또는 그룹 홀드아웃을 우선합니다.

## 주요 하이퍼파라미터

| 설정 | 늘리거나 완화했을 때 | 확인할 점 |
|---|---|---|
| `n_estimators` | 평균이 안정되지만 시간·메모리가 증가 | validation/OOB 성능이 평탄해지는 지점 |
| `max_depth` | 개별 트리가 복잡해짐 | train-validation 격차 |
| `min_samples_leaf` | 값을 키우면 잎 예측이 부드러워짐 | 희소 구간을 지나치게 뭉개는지 여부 |
| `max_features` | 값을 줄이면 트리 다양성이 커짐 | 편향 증가와 분산 감소의 균형 |
| `max_samples` | 트리마다 사용하는 행 수를 조절 | 속도와 개별 트리 정보량 |
| `bootstrap` | `True`일 때 행 재표본화와 OOB가 가능 | OOB를 사용하려면 `True`가 필요함 |
| `random_state` | 무작위 과정을 재현 | 비교 모델에서 고정 |

`n_estimators`를 늘리면 대체로 예측이 안정되지만 적절하지 않은 분할, 누수, 외삽 한계까지 해결되지는 않습니다. 깊은 트리를 많이 만들수록 메모리 사용량과 학습 시간도 함께 늘어나므로 성능이 평탄해지는 지점을 살펴보는 것이 좋습니다.

## 검증 설계 — 모델 비교 전에 먼저 고정합니다

3주차와 마찬가지로 train, validation, test의 역할을 분리합니다. 따릉이처럼 시간 순서가 있는 문제라면 다음 절차가 기본입니다.

1. 마지막 기간은 **최종 test**로 따로 보관합니다.
2. 그보다 앞선 여러 구간에서 확장 창 방식의 시간순 교차검증을 수행합니다.
3. 평균 validation RMSE와 시점별 변동을 기준으로 세 모델과 설정을 선택합니다.
4. 선택이 끝나면 train+validation으로 다시 학습합니다.
5. 따로 보관한 test를 마지막에 한 번 평가해 최종 성능으로 기록합니다. 이 결과를 보고 설정을 다시 바꾸면 test가 선택용 데이터가 되므로, 추가 수정이 필요할 때는 새로운 평가 데이터를 마련합니다.

현재 `example.ipynb`의 마지막 60일 점수는 같은 데이터와 분할에서 모델을 비교한 홀드아웃 결과입니다. 그 점수를 보며 `max_depth`, `min_samples_leaf`, `max_features`를 바꿨다면 마지막 60일은 validation이며, 별도 test 없이는 최종 일반화 성능이라고 부를 수 없습니다.

전처리는 각 학습 폴드 안에서만 `fit`하는 것이 중요합니다. 전체 데이터로 결측 대푯값, 빈도, 타깃 평균, 범주 목록을 만든 뒤 교차검증하면 검증 행의 정보가 학습에 들어가기 때문입니다. `Pipeline`으로 전처리와 모델을 묶으면 이 순서를 일관되게 지킬 수 있습니다.

## 순서대로 따라가기

아래 Part 번호와 코드 이름을 기준으로 `example.ipynb`의 대응 셀을 찾습니다.

### Part 2 — 지난주 데이터와 모델 재구성

2~3주차와 같은 X(`day_index`, 월 더미, 요일 더미), y(대여건수), 마지막 60일 홀드아웃을 사용합니다. Part 2의 `linear_model`과 `tree_model`을 학습하는 셀에서 선형회귀와 사전 가지치기 단일 트리의 기준점을 다시 계산합니다.

```python
linear_model = LinearRegression().fit(X_train, y_train)
rmse_linear = root_mean_squared_error(y_test, linear_model.predict(X_test))

tree_model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_leaf=20,
    random_state=42,
).fit(X_train, y_train)
rmse_tree = root_mean_squared_error(y_test, tree_model.predict(X_test))

print('선형회귀 RMSE:', rmse_linear)
print('단일 트리(가지치기) RMSE:', rmse_tree)
```

```text
선형회귀 RMSE: 약 29,196
단일 트리(가지치기) RMSE: 약 31,516
```

### Part 3 — Random Forest 학습

현재 노트북의 설정은 `n_estimators=100`, `max_depth=6`, `min_samples_leaf=10`이며 `max_features`는 명시하지 않습니다.

```python
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=6,
    min_samples_leaf=10,
    random_state=42,
)
rf_model.fit(X_train, y_train)

rmse_rf_train = root_mean_squared_error(y_train, rf_model.predict(X_train))
rmse_rf_test = root_mean_squared_error(y_test, rf_model.predict(X_test))
print('Random Forest - train RMSE:', rmse_rf_train)
print('Random Forest - test RMSE :', rmse_rf_test)
```

```text
Random Forest - train RMSE: 약 28,242
Random Forest - test RMSE : 약 30,939
```

같은 홀드아웃에서 Random Forest의 RMSE는 단일 트리의 31,516보다 낮았습니다. 여러 부트스트랩 트리의 평균이 단일 트리의 불안정성을 일부 줄였다는 설명과 일치합니다. 다만 한 홀드아웃의 차이는 특정 구간에서 얻은 결과이므로, 여러 시간 구간의 점수 분포도 함께 비교하면 결론을 더 안정적으로 설명할 수 있습니다.

> 🔄 **예측 → 실행 → 비교 → 해석 ③: 실제 데이터에서 앙상블 효과 확인**
>
> 1. 실행 전에 RF의 train RMSE와 마지막 60일 RMSE가 단일 트리보다 각각 높을지 낮을지 예상합니다.
> 2. Part 2의 기준 모델 셀과 Part 3의 `rf_model` 셀을 실행합니다.
> 3. `29,196 / 31,516 / 30,939`의 모델 순위와 train-test 격차를 비교합니다.
> 4. 단일 트리 대비 개선은 분산 감소로, 선형회귀 대비 열세는 외삽 편향으로 나누어 해석합니다.

> 📷 **스크린샷 플레이스홀더 — Random Forest 훈련·홀드아웃 RMSE 출력**
> - **캡처 출처:** `04주차/example.ipynb` Part 3의 `rf_model = RandomForestRegressor(...)`로 시작하는 코드 셀과 출력입니다.
> - **의도:** 하이퍼파라미터와 train/test RMSE가 한 화면에서 연결되도록 보여줍니다.
> - **관찰 질문:** train과 test의 격차는 제한 없는 단일 트리보다 작습니까? 같은 test로 설정을 골랐다면 이 구간의 올바른 이름은 무엇입니까?
> - **캡션/대체 텍스트:** "100개 트리, 깊이 6, 최소 잎 표본 10으로 학습한 Random Forest의 train RMSE 약 28,242와 마지막 60일 RMSE 약 30,939"를 사용합니다.

### Part 4 — Feature Importance 비교

Random Forest의 `feature_importances_`는 각 트리의 불순도 감소 기반 중요도(MDI)를 평균한 값입니다. 노트북에서는 단일 트리와 Random Forest의 상위 10개 중요도를 나란히 그립니다.

```python
importance_compare = pd.DataFrame({
    '단일 트리': tree_model.feature_importances_,
    'Random Forest': rf_model.feature_importances_,
}, index=X.columns).sort_values('Random Forest', ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
importance_compare.head(10).plot(kind='barh', ax=ax)
ax.invert_yaxis()
ax.set_title('Feature Importance 비교 (상위 10개)')
plt.show()
```

여러 트리를 평균하면 표본 변화에 덜 흔들릴 수 있지만, **평균을 냈다고 해서 중요도 편향이 사라지는 것은 아닙니다.**

- 고유값이 많은 연속형 변수와 ID는 분기 후보가 많아 중요도가 과대평가될 수 있습니다.
- 상관된 변수들은 중요도를 나눠 갖거나 한쪽에 몰아줄 수 있습니다.
- 훈련 데이터에서 계산하므로 과적합한 신호도 중요해 보일 수 있습니다.
- 중요도에는 방향이 없으며 인과 효과를 뜻하지 않습니다.

검증 데이터에서 permutation importance를 함께 계산하면 중요도를 더 균형 있게 이해할 수 있습니다. 변수 하나를 여러 번 섞어 RMSE가 얼마나 증가하는지 평균과 표준편차로 확인합니다. 모델 자체의 validation 성능이 충분한지 먼저 살펴보고, 상관 변수가 있으면 개별 순위는 신중하게 해석합니다.

> 🔄 **예측 → 실행 → 비교 → 해석 ④: 평균된 중요도 읽기**
>
> 1. 단일 트리와 RF에서 중요도 순위가 얼마나 비슷할지 예상합니다.
> 2. Part 4의 `importance_compare`를 만드는 셀을 실행합니다.
> 3. 같은 변수의 두 막대와 상위 순위 변화를 비교합니다.
> 4. 막대가 안정적으로 보인다는 사실과 편향이 제거되었다는 주장을 구분합니다.

![단일 Decision Tree와 100개 트리 Random Forest의 불순도 기반 변수 중요도 상위 10개를 나란히 표시한 가로 막대그래프](assets/feature_importance_tree_vs_rf.png)

*그림 4-1. 단일 Decision Tree와 100개 트리 Random Forest의 불순도 기반 변수 중요도 상위 10개를 나란히 표시한 가로 막대그래프입니다.*

> **그림에서 볼 점:** 같은 변수에 대한 단일 트리와 RF의 MDI를 한 그래프에서 비교합니다.  
> **관찰 질문:** RF에서 막대가 덜 쏠려 보이더라도 고유값 수 편향과 상관 변수 문제는 왜 남습니까?  
> **노트북 연결:** `example.ipynb` Part 4에서 같은 변수 중요도 그래프를 직접 생성해 본문 그림과 비교합니다.

### Part 5 — 세 모델 종합 비교

노트북은 세 모델의 홀드아웃 RMSE를 정렬해 한 표로 만듭니다.

```python
comparison = pd.DataFrame({
    '모델': ['선형회귀', '단일 트리(가지치기)', 'Random Forest'],
    'test RMSE': [rmse_linear, rmse_tree, rmse_rf_test],
}).sort_values('test RMSE')
comparison
```

대표 실행 결과를 정수 단위로 반올림하면 다음과 같습니다.

```text
                  모델  test RMSE
0               선형회귀      29,196
2        Random Forest      30,939
1       단일 트리(가지치기)      31,516
```

Random Forest는 이 구간에서 단일 트리보다 낮은 RMSE를 보였지만 선형회귀보다는 RMSE가 높았습니다. Random Forest도 결국 구간별 상수인 트리들의 평균입니다. 학습 범위를 벗어난 `day_index`가 들어오면 기존 잎들의 평균을 조합할 뿐 상승 직선을 연장하지 못합니다.

따라서 이번 결과는 다음 두 문장으로 분리해 해석합니다.

1. **단일 트리 대비:** 여러 트리를 평균해 분산을 줄였습니다.
2. **선형회귀 대비:** 트리 계열의 외삽 한계가 남았습니다.

앙상블은 기반 모델의 불안정성을 줄일 수 있지만, 적절하지 않은 변수, 데이터 누수, 분포 변화, 구조적 편향을 자동으로 해결하지는 않습니다.

> 🔄 **예측 → 실행 → 비교 → 해석 ⑤: 점수와 시계열 모양 함께 보기**
>
> 1. 마지막 60일에서 RF 예측선이 선형회귀와 단일 트리 중 어느 쪽에 더 가까울지 예상합니다.
> 2. Part 5의 `comparison` 표와 날짜별 예측 그래프를 만드는 셀을 실행합니다.
> 3. RMSE 순위와 날짜별 예측선의 모양을 함께 비교합니다.
> 4. RF가 단일 트리보다 나아진 이유와 선형회귀를 이기지 못한 이유를 서로 다른 문장으로 기록합니다.

![마지막 60일의 실제 따릉이 대여건수와 선형회귀, 가지치기 단일 트리, Random Forest 예측을 날짜순으로 비교한 선그래프](assets/bike_predictions_three_models.png)

*그림 4-2. 마지막 60일의 실제 따릉이 대여건수와 선형회귀, 가지치기 단일 트리, Random Forest 예측을 날짜순으로 비교한 선그래프입니다.*

> **그림에서 볼 점:** RMSE 한 개로 가려지는 시점별 오차와 트리 계열의 외삽 한계를 확인합니다.  
> **관찰 질문:** Random Forest와 단일 트리의 선은 미래 구간에서 어떤 공통 모양을 보이며, 선형회귀와 무엇이 다릅니까?  
> **노트북 연결:** `example.ipynb` Part 5에서 같은 예측 비교 그래프를 직접 생성해 본문 그림과 비교합니다.

## 과제 안내 — 물류 유통량 데이터

[물류 유통량 예측 경진대회](https://dacon.io/competitions/official/235867)에서 제공하는 데이터입니다. 한 행은 송하인 지역·수하인 지역·물품 카테고리의 조합에 해당하며, 목표값은 해당 조합의 운송장 건수입니다.

| 변수 | 의미 |
|---|---|
| `송하인_격자공간고유번호` | 물품을 보내는 지역의 격자 ID |
| `수하인_격자공간고유번호` | 물품을 받는 지역의 격자 ID |
| `물품_카테고리` | 운송 물품의 분류 |
| `운송장_건수` | 예측 대상인 운송장 수 |

`train.csv`에는 목표값이 포함되어 있고, 대회 `test.csv`는 예측 대상입니다. 파일 위치는 `dataset/open/extracted/물류 유통량 예측 경진대회/`입니다.

격자 ID는 범주를 구분하는 코드이며 숫자의 크기가 실제 공간 거리나 지역의 순서를 뜻하지 않습니다. test에는 train에서 보지 못한 격자 ID가 포함되어 있어, 학습 데이터에 등장한 지역과 새로운 지역의 구성이 다릅니다.

## 자주 만나는 문제와 점검 방법

| 증상 | 가능한 원인 | 확인·대응 |
|---|---|---|
| RF가 실행할 때마다 달라짐 | 무작위 표본 또는 동률 분기 | `random_state`를 고정하고 여러 seed의 안정성을 확인합니다. |
| 트리를 늘려도 점수가 좋아지지 않음 | 편향, 변수 부족, 외삽 한계 | validation 잔차와 데이터 구조를 점검합니다. |
| OOB는 좋지만 미래 test가 나쁨 | OOB가 시간 순서를 무시 | 시간순 validation을 사용합니다. |
| 격자 ID 중요도가 압도적임 | 정수 순서 또는 고유값 편향 | one-hot 또는 의미 있는 공간 변수를 사용하고 permutation으로 검증합니다. |
| validation 변환에서 unknown 오류 발생 | encoder의 fit 대상이 알맞지 않음 | train에만 `OneHotEncoder(handle_unknown='ignore')`를 fit합니다. |
| 내부 validation은 좋지만 competition test가 나쁨 | 미지 ID 비율 또는 분포가 다름 | 그룹 분할과 known/unknown별 점수를 보고합니다. |
| 타깃 인코딩 점수가 지나치게 높음 | 같은 행의 y가 인코딩에 포함됨 | fold 밖 통계로 cross-fitting합니다. |

## 인사이트를 쓰는 방법

좋은 설명은 단순 순위가 아니라 **어느 약점이 줄었고 어느 한계가 남았는지**를 구분합니다.

> 같은 마지막 60일에서 Random Forest RMSE는 30,939로 단일 트리의 31,516보다 낮았습니다. 부트스트랩으로 달라진 트리 예측을 평균해 단일 트리의 분산을 줄였다는 설명과 일치합니다. 그러나 선형회귀의 29,196보다는 높았습니다. 모든 기반 트리가 학습 범위 밖의 시간 추세를 외삽하지 못하므로 평균 후에도 그 편향이 남았기 때문입니다. 이 결론은 단일 홀드아웃 관찰이므로 여러 미래 구간에서 재검증하면 신뢰도를 더 높일 수 있습니다.

## 4주차 요약

- Random Forest 회귀는 부트스트랩 표본으로 여러 트리를 학습하고 예측을 평균해 분산을 줄입니다.
- 변수 부분집합은 `max_features`로 제어하며, 현재 노트북은 이를 명시하지 않아 최신 기본값에서는 모든 변수를 후보로 봅니다.
- OOB는 각 트리에 뽑히지 않은 행으로 내부 성능을 추정하지만 시간·그룹 구조를 무시하므로 모든 문제의 검증 대안은 아닙니다.
- 트리를 평균해도 외삽 불가, 적절하지 않은 변수 표현, 누수 같은 공통 편향은 사라지지 않습니다.
- RF의 불순도 중요도 역시 고유값 수와 상관관계의 영향을 받으므로 홀드아웃 permutation importance로 보완합니다.
- 범주형 격자 ID를 `LabelEncoder` 정수로 넣지 않고 train에만 fit한 unknown-safe 전처리를 Pipeline으로 묶습니다.
- 모델 선택은 validation에서, 최종 성능 평가는 따로 보관한 holdout/test에서 수행합니다.

## 참고 자료

- [scikit-learn RandomForestRegressor 문서](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [scikit-learn LabelEncoder 문서](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
- [scikit-learn OneHotEncoder 문서](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [scikit-learn permutation importance 안내](https://scikit-learn.org/stable/modules/permutation_importance.html)
