# PSPY 어깨너머 — 데이터 분석

PSPY 어깨너머 데이터 분석 스터디를 위한 8주 과정의 학습 자료입니다. 주차별 개념 설명(`notion.md`), 실습 노트북(`practice.ipynb`), 과제 시작점(`assignment_baseline.ipynb`)을 통해 탐색적 데이터 분석과 시각화, 회귀·분류·군집화, 공간분석과 데이터 결합을 다룹니다.

## 퀵 스타트

### 로컬

Python 3.10 이상을 설치하고 레포를 내려받은 뒤, 프로젝트 루트의 터미널에서 실행합니다. Windows에서는 `py`, macOS/Linux에서는 `python3`를 사용해도 됩니다.

데이터 ZIP은 Git LFS로 관리합니다. Git LFS를 설치하고 `git lfs install`을 실행한 뒤 레포를 clone하세요. 이미 clone했다면 레포에서 `git lfs pull`로 실제 ZIP을 받습니다. Drive에도 LFS 포인터가 아닌 실제 ZIP 파일을 복사합니다.

```bash
python scripts/setup_local.py
```

스크립트가 `.venv` 생성, `requirements.txt`의 공통 라이브러리 설치, 커널 등록과 JupyterLab 실행을 처리합니다. 노트북에서 **Python (PSPY .venv)** 커널을 선택하고 첫 준비 셀부터 실행하세요. 다음에도 같은 명령을 사용하며, 이미 준비된 환경은 재사용합니다. Ubuntu/Debian에서 `ensurepip` 오류가 나면 `python3-venv` 패키지를 먼저 설치합니다.

### Colab

레포의 코드·노트북과 `dataset/open/raw/`의 ZIP을 Google Drive에 같은 폴더 구조로 둡니다. 로컬 `.venv`는 복사할 필요가 없습니다. Colab에서 노트북을 열고 파일 패널에서 Drive를 연결한 뒤, 첫 준비 셀을 다음 한 줄로 바꿉니다.

```python
%run "/content/drive/MyDrive/Colab Notebooks/PSPY-DataAnalysis/scripts/notebook_setup.py" --week 1
```

경로는 자신의 Drive 위치로, `--week`는 해당 주차로 맞춥니다. 과제는 기존 셀의 `--kind assignment`도 유지하세요. 별도 venv 없이 Colab 런타임을 사용하며, 스크립트가 필요한 공통 패키지와 한글 폰트를 준비합니다.

두 환경 모두 준비 셀이 필요한 ZIP 항목만 `dataset/open/extracted/`에 풀고 기존 파일은 재사용합니다. 8주차 과제는 `--data jeju` 또는 `--data credit`으로 선택합니다. KRX는 개인 API 키·캐시를 사용하고, 7주차 자료는 별도 준비합니다. 선택 활동에만 `folium` 또는 `pyarrow`가 필요하며, 자세한 설치·데이터 준비 방법은 [OT](00_OT/notion.md#로컬-가상환경-만들기)에 정리되어 있습니다.

## 주차별 개요

### [1주차 — EDA·시각화와 API](01주차/notion.md)

따릉이 대여이력으로 데이터 구조, 결측치와 이상치를 확인하고 질문에 맞는 그래프를 만듭니다. KRX API의 지수·코스닥 데이터를 수집하고 검산하는 흐름을 익힌 뒤, 과제에서는 코스닥 종목의 하루 단면을 탐색합니다.

### [2주차 — 선형회귀](02주차/notion.md)

따릉이 일별 대여량으로 단순·다중 선형회귀를 비교하며 변수 선택, 시간순 분할, RMSE와 잔차 해석을 익힙니다. 과제에서는 감귤나무의 생육 정보로 착과량을 예측하고 입력 변수를 바꿨을 때의 차이를 살펴봅니다.

### [3주차 — 의사결정나무](03주차/notion.md)

같은 따릉이 데이터에 의사결정나무를 적용해 분기 규칙을 읽고, 트리 깊이와 잎 크기에 따른 과적합을 확인합니다. 과제에서는 전력사용량과 건물정보를 결합하고 결측치를 처리해 예측 모델을 만듭니다.

### [4주차 — 랜덤포레스트](04주차/notion.md)

여러 트리의 예측을 평균하는 랜덤포레스트를 익히고, 같은 검증 구간에서 선형회귀·단일 트리와 비교합니다. 과제에서는 물류 데이터의 격자 ID와 물품 카테고리를 인코딩해 운송장 건수를 예측합니다.

### [5주차 — 분류와 평가 지표](05주차/notion.md)

대출 승인 데이터로 분류 모델을 만들고 혼동행렬, 정확도, 정밀도, 재현율과 F1을 해석합니다. 과제에서는 항공편 지연 여부를 예측하며 정답이 없는 행을 구분하고, 학습 데이터에만 전처리를 맞추는 파이프라인을 사용합니다.

### [6주차 — 군집화와 고객 세분화](06주차/notion.md)

고객 성향 데이터로 스케일링과 K-means를 실습하고, 군집 수에 따른 결과와 군집별 특성을 비교합니다. 과제에서는 이커머스 거래내역으로 RFM 지표를 구성해 고객을 나누고 각 군집의 특징을 설명합니다.

### [7주차 — 공간 접근성과 사각지대](07주차/notion.md)

서울시 단속 CCTV와 S-DoT 유동인구 자료로 거리, 커버리지, 그리디 후보지 선정과 민감도 분석을 실습합니다. 과제에서는 심야약국 접근성 분석에 필요한 데이터와 질문을 정리하며, 약국 원본 출처는 아직 확정되지 않았습니다.

### [8주차 — 미니 프로젝트](08주차/notion.md)

분석 질문을 정하고 외부 데이터를 결합할 때의 키 매칭과 단위 통일을 점검합니다. 제주도 교통량, 신용카드 고객 세그먼트 또는 직접 고른 데이터로 작은 분석을 진행하고 관찰한 결과와 한계를 정리합니다.

## 데이터셋 출처

| 데이터셋 | 출처 |
|---|---|
| 따릉이 대여소 정보 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-13252/F/1/datasetView.do) |
| 따릉이 대여이력 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15182/A/1/datasetView.do) |
| 따릉이 일별 대여건수 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-14994/A/1/datasetView.do) |
| KRX 지수·코스닥 종목 시세 | [KRX OPEN API](https://openapi.krx.co.kr/contents/OPP/INFO/service/OPPINFO004.cmd) |
| 감귤 착과량 예측 | [DACON](https://dacon.io/competitions/official/236038) |
| 전력사용량 예측 | [DACON](https://dacon.io/competitions/official/236125) |
| 물류 유통량 예측 | [DACON](https://dacon.io/competitions/official/235867) |
| Loan Prediction Problem Dataset | [Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) |
| 항공편 지연 예측 | [DACON](https://dacon.io/competitions/official/236094) |
| Customer Personality Analysis | [Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) |
| 이커머스 고객 세분화 | [DACON](https://dacon.io/competitions/official/236222) |
| 서울시 불법주정차·전용차로 단속 CCTV 위치정보 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-20471/S/1/datasetView.do) |
| S-DoT 유동인구 설치 위치·측정 정보 | [서울 열린데이터광장](https://data.seoul.go.kr/dataList/OA-15964/S/1/datasetView.do) |
| 제주도 도로 교통량 예측 | [DACON](https://dacon.io/competitions/official/235985) |
| 신용카드 고객 세그먼트 분류 | [DACON](https://dacon.io/competitions/official/236460) |
