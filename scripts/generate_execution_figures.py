"""Generate actual execution figures for the course's screenshot callouts.

Run from any directory:
    python scripts/generate_execution_figures.py
    python scripts/generate_execution_figures.py --data-root /path/to/dataset

Missing source datasets are reported and skipped; no synthetic or transcribed
results are substituted. Existing course images are left untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor, plot_tree


ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, NAVY = "#3A7BD5", "#E99325", "#183B56"


def configure_style() -> None:
    # WSL can use the host's font without copying it into the repository.
    windows_font = Path("/mnt/c/Windows/Fonts/malgun.ttf")
    if windows_font.exists():
        font_manager.fontManager.addfont(windows_font)
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        raise RuntimeError("한국어 글꼴(Malgun Gothic, Noto Sans CJK, NanumGothic)이 필요합니다.")
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "font.size": 11,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig, week: str, name: str) -> None:
    path = ROOT / week / "assets" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", metadata={
        "Software": "Matplotlib; scripts/generate_execution_figures.py",
    })
    plt.close(fig)
    print(f"생성: {path.relative_to(ROOT)}")


def table(ax, columns, rows, title: str, widths=None) -> None:
    ax.axis("off")
    ax.set_title(title, color=NAVY, pad=18)
    chart = ax.table(cellText=rows, colLabels=columns, cellLoc="left",
                     colLoc="left", colWidths=widths, bbox=[0, 0, 1, 0.88])
    chart.auto_set_font_size(False)
    chart.set_fontsize(11)
    for (row, _), cell in chart.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(NAVY if row == 0 else ("#EDF3FA" if row % 2 else "#F8FAFC"))
        if row == 0:
            cell.set_text_props(color="white", weight="bold")


def make_orientation() -> None:
    paths = ["00_OT", *(f"{i:02d}주차" for i in range(1, 9)),
             "dataset/open/raw", "dataset/open/extracted", "dataset/open/processed"]
    roles = ["오리엔테이션", *["주차별 교재·예제·과제"] * 8,
             "원본 보관", "노트북 입력", "분석 산출물"]
    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    rows = [[path, role, "있음" if (ROOT / path).is_dir() else "없음"]
            for path, role in zip(paths, roles)]
    table(ax, ["프로젝트 루트 기준 경로", "용도", "실제 확인"], rows,
          "프로젝트 폴더 존재 여부 확인", [0.48, 0.36, 0.16])
    save(fig, "00_OT", "project-folder-check.png")

    filenames = ["notion.md", "example.ipynb", "assignment_baseline.ipynb"]
    roles = ["개념 읽기", "예제 따라 실행하기", "과제에 직접 적용하기"]
    counts = [sum((ROOT / f"{i:02d}주차" / name).is_file() for i in range(1, 9))
              for name in filenames]
    fig, ax = plt.subplots(figsize=(11, 3.6), layout="constrained")
    table(ax, ["파일", "역할", "1~8주차 존재 확인"],
          [[name, role, f"{count}/8"] for name, role, count in zip(filenames, roles, counts)],
          "주차별 세 파일 확인 결과", [0.43, 0.32, 0.25])
    save(fig, "00_OT", "weekly-file-check.png")

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    rows = [["Python", platform.python_version()], ["pandas", pd.__version__],
            ["scikit-learn", sklearn.__version__], ["matplotlib", matplotlib.__version__]]
    table(ax, ["실행 환경", "실제로 불러온 버전"], rows, "패키지 import 실행 결과")
    save(fig, "00_OT", "environment-versions.png")


def make_toy_stability() -> None:
    toy = pd.DataFrame({"기온": [5, 10, 15, 20, 25, 30],
                        "판매량": [20, 35, 40, 55, 70, 90]})
    datasets = [toy, toy.drop(index=3)]
    trees = [DecisionTreeRegressor(max_depth=2, random_state=42).fit(
        data[["기온"]], data["판매량"]) for data in datasets]
    forests = [RandomForestRegressor(n_estimators=100, max_depth=2,
        max_features=1.0, random_state=42, n_jobs=1).fit(
            data[["기온"]], data["판매량"]) for data in datasets]
    query = pd.DataFrame({"기온": [22]})
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), layout="constrained")
    for ax, model, data in zip(axes, trees, datasets):
        plot_tree(model, feature_names=["기온"], filled=True, rounded=True,
                  precision=2, fontsize=10, ax=ax, impurity=False)
        ax.set_title(f"{len(data)}개 관측치 학습 · 22°C 예측 {model.predict(query)[0]:.1f}")
    fig.suptitle("20°C 관측치 하나를 제외하면 트리 규칙이 달라집니다", color=NAVY, fontsize=18)
    save(fig, "04주차", "toy-tree-rules.png")

    predictions = np.array([[m.predict(query)[0] for m in models] for models in [trees, forests]])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    x = np.arange(2)
    for i, label in enumerate(["전체 6개", "20°C 관측치 제외"]):
        bars = axes[0].bar(x + (i - 0.5) * 0.32, predictions[:, i], width=0.32,
                           label=label, color=[BLUE, ORANGE][i])
        axes[0].bar_label(bars, fmt="%.2f", padding=4)
    axes[0].set(xticks=x, xticklabels=["단일 트리", "Random Forest"],
                ylabel="22°C에서 예측한 판매량", title="같은 입력의 예측값", ylim=(0, 85))
    axes[0].legend(loc="lower right")
    changes = np.abs(predictions[:, 1] - predictions[:, 0])
    bars = axes[1].bar(["단일 트리", "Random Forest"], changes, color=[ORANGE, BLUE], width=0.5)
    axes[1].bar_label(bars, fmt="%.2f", padding=4)
    axes[1].set(ylabel="예측 변화량의 절댓값", title="관측치 하나를 제외한 영향", ylim=(0, 18))
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    save(fig, "04주차", "toy-prediction-change.png")
    print("트리 / RF 예측:", predictions.tolist())


def make_rentals(data_root: Path) -> None:
    path = data_root / "open/extracted/따릉이 공공데이터/03_대여이력/서울특별시 공공자전거 대여이력 정보_2602.csv"
    columns = ["대여일시", "이용시간(분)", "이용거리(M)", "생년", "성별", "이용자종류"]
    df = pd.read_csv(path, encoding="cp949", usecols=columns)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), layout="constrained")
    table(axes[0], ["열", "자료형"], [[col, str(df[col].dtype)] for col in columns],
          f"따릉이 2026년 2월 · {len(df):,}행 × {len(df.columns)}열")
    table(axes[1], columns, df.head().fillna("결측").astype(str).values.tolist(), "첫 5행")
    save(fig, "01주차", "rental-data-structure.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    series = [df.isna().mean(), df.groupby("이용자종류")["성별"].apply(lambda s: s.isna().mean())]
    for ax, values, title in zip(axes, series, ["전체 열별 결측률", "이용자종류별 성별 결측률"]):
        bars = ax.bar(values.index, values.values, color=BLUE)
        ax.bar_label(bars, labels=[f"{v:.1%}" for v in values], padding=4)
        ax.set(title=title, ylim=(0, 1.12), ylabel="결측 비율")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(axis="x", rotation=25)
    save(fig, "01주차", "rental-missing-rates.png")


def make_krx(data_root: Path) -> None:
    folder = data_root / "open/extracted/KRX"
    index = pd.read_csv(folder / "krx_index_20240823.csv")
    stocks = pd.read_csv(folder / "kosdaq_stocks_20240823.csv", dtype={"ISU_CD": "string"})
    zero_volume = stocks["ACC_TRDVOL"].eq(0)
    traded = stocks.loc[~zero_volume]
    violations = ((traded["TDD_HGPRC"] < traded[["TDD_OPNPRC", "TDD_CLSPRC"]].max(axis=1))
        | (traded["TDD_LWPRC"] > traded[["TDD_OPNPRC", "TDD_CLSPRC"]].min(axis=1))
        | (traded["ACC_TRDVOL"] < 0)).sum()
    rows = [
        ["행 × 열", f"{index.shape[0]:,} × {index.shape[1]}", f"{stocks.shape[0]:,} × {stocks.shape[1]}"],
        ["고유키 중복", index.duplicated(["BAS_DD", "IDX_CLSS", "IDX_NM"]).sum(), stocks.duplicated(["BAS_DD", "ISU_CD"]).sum()],
        ["결측 칸", int(index.isna().sum().sum()), int(stocks.isna().sum().sum())],
        ["무거래 종목", "해당 없음", int(zero_volume.sum())],
        ["거래 종목 OHLC·거래량 위반", "해당 없음", int(violations)],
    ]
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    table(ax, ["검산 항목", "KRX 지수", "코스닥 종목"], rows,
          "2024-08-23 KRX 로컬 CSV 검산 결과", [0.5, 0.25, 0.25])
    save(fig, "01주차", "krx-data-validation.png")


def make_bike_rmse(data_root: Path) -> None:
    # Reuse exactly the feature construction and chronological split used by
    # the course generator, with the current dataset/open path convention.
    import importlib.util
    spec = importlib.util.spec_from_file_location("week3_assets", ROOT / "03주차/assets/generate_assets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.DATA_DIR = data_root / "open/extracted/따릉이 공공데이터/02_이용정보"
    _, X, y, split = module.load_bike_data()
    models = [
        ("03주차", "tree-train-holdout-rmse.png", "제한 없는 Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("04주차", "rf-train-holdout-rmse.png", "Random Forest · 트리 100개 / 깊이 6 / 최소 잎 10",
         RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=10,
                               max_features=1.0, random_state=42, n_jobs=1)),
    ]
    for week, name, title, model in models:
        model.fit(X.iloc[:split], y.iloc[:split])
        scores = [root_mean_squared_error(y.iloc[s], model.predict(X.iloc[s]))
                  for s in (slice(None, split), slice(split, None))]
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        bars = ax.bar([f"훈련 ({split:,}일)", "홀드아웃 (마지막 60일)"], scores, color=[BLUE, ORANGE], width=0.5)
        ax.bar_label(bars, labels=[f"{value:,.1f}" for value in scores], padding=6)
        ax.set(title=title, ylabel="RMSE (대여건수)", ylim=(0, max(scores) * 1.18))
        save(fig, week, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    configure_style()
    make_orientation()
    make_toy_stability()
    for generate in (make_rentals, make_krx, make_bike_rmse):
        try:
            generate(args.data_root)
        except FileNotFoundError as exc:
            print(f"데이터 필요 ({generate.__name__}): {exc}")


if __name__ == "__main__":
    main()
