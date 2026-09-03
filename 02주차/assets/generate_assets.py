"""Generate the deterministic figures embedded in 02주차/notion.md.

Run from any directory with:
    python 02주차/assets/generate_assets.py

All model inputs and feature engineering match 02주차/example.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from sklearn.linear_model import LinearRegression


ASSET_DIR = Path(__file__).resolve().parent
WEEK_DIR = ASSET_DIR.parent
PROJECT_ROOT = WEEK_DIR.parent
DATA_DIR = (
    PROJECT_ROOT / "dataset" / "open" / "extracted" / "따릉이 공공데이터" / "02_이용정보"
)

COLORS = {
    "actual": "#333333",
    "simple": "#E45756",
    "multi": "#4C78A8",
    "green": "#54A24B",
    "orange": "#F58518",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.edgecolor": "#B8B8B8",
            "grid.color": "#D9D9D9",
            "grid.alpha": 0.55,
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    output = ASSET_DIR / filename
    fig.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output.relative_to(PROJECT_ROOT)}")


def load_daily_rentals() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("서울특별시 공공자전거 일별 대여건수_*.csv"))
    if len(files) != 5:
        raise FileNotFoundError(f"일별 대여건수 CSV 5개가 필요하지만 {len(files)}개를 찾았습니다.")
    frames = [pd.read_csv(path, encoding="cp949") for path in files]
    df = pd.concat(frames, ignore_index=True)
    df["대여일자"] = pd.to_datetime(df["대여일자"])
    df = df.sort_values("대여일자").reset_index(drop=True)
    if len(df) != 912 or df["대여일자"].duplicated().any():
        raise ValueError("노트북의 912일·중복 0일 데이터 계약과 다릅니다.")
    df["day_index"] = (df["대여일자"] - df["대여일자"].min()).dt.days
    df["월"] = df["대여일자"].dt.month
    df["요일"] = df["대여일자"].dt.day_name()
    df["주말여부"] = df["대여일자"].dt.dayofweek.isin([5, 6]).astype(int)
    return df


def fit_models(df: pd.DataFrame) -> dict[str, object]:
    split_idx = len(df) - 60
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]
    y_train, y_test = train["대여건수"], test["대여건수"]

    simple_model = LinearRegression().fit(train[["day_index"]], y_train)
    pred_simple = simple_model.predict(test[["day_index"]])

    month_dummies = pd.get_dummies(df["월"], prefix="월", drop_first=True)
    weekday_dummies = pd.get_dummies(df["요일"], prefix="요일", drop_first=True)
    features = pd.concat([df[["day_index"]], month_dummies, weekday_dummies], axis=1)
    x_train, x_test = features.iloc[:split_idx], features.iloc[split_idx:]
    multi_model = LinearRegression().fit(x_train, y_train)
    pred_multi = multi_model.predict(x_test)

    return {
        "train": train,
        "test": test,
        "y_test": y_test,
        "pred_simple": pred_simple,
        "pred_multi": pred_multi,
    }


def format_date_axis(ax: plt.Axes, interval: int = 2) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=35)


def generate_toy_regression() -> None:
    toy = pd.DataFrame(
        {
            "공부시간": [1, 2, 3, 4, 5, 6],
            "시험점수": [42, 50, 55, 68, 72, 85],
        }
    )
    model = LinearRegression().fit(toy[["공부시간"]], toy["시험점수"])
    pred = model.predict(toy[["공부시간"]])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        toy["공부시간"],
        toy["시험점수"],
        s=75,
        color=COLORS["actual"],
        label="실제 값",
        zorder=3,
    )
    ax.plot(
        toy["공부시간"],
        pred,
        color=COLORS["simple"],
        linewidth=2.5,
        label="회귀선",
    )
    for x, y, yhat in zip(toy["공부시간"], toy["시험점수"], pred, strict=True):
        ax.vlines(x, min(y, yhat), max(y, yhat), color="#999999", linestyle=":", linewidth=1)
    ax.set_xlabel("공부시간")
    ax.set_ylabel("시험점수")
    ax.set_title("공부시간과 시험점수: 실제 값과 최소제곱 회귀선")
    ax.text(
        0.03,
        0.94,
        f"예측식: 점수 = {model.coef_[0]:.1f} × 공부시간 + {model.intercept_:.1f}",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
    )
    ax.legend(loc="lower right")
    ax.grid()
    fig.tight_layout()
    save_figure(fig, "01_toy_regression.png")


def generate_full_timeseries(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(df["대여일자"], df["대여건수"], color=COLORS["actual"], linewidth=1.2)
    ax.set_title("일별 따릉이 대여건수 — 2024년 1월~2026년 6월")
    ax.set_xlabel("대여일자")
    ax.set_ylabel("대여건수")
    ax.set_xlim(df["대여일자"].min(), df["대여일자"].max())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    format_date_axis(ax, interval=2)
    ax.grid()
    fig.tight_layout()
    save_figure(fig, "02_daily_timeseries.png")


def generate_month_weekday_means(df: pd.DataFrame) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    labels = ["월", "화", "수", "목", "금", "토", "일"]
    train = df.iloc[:-60]
    monthly = train.groupby("월")["대여건수"].mean()
    weekday = train.groupby("요일")["대여건수"].mean().reindex(order)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    axes[0].bar(monthly.index, monthly.values, color=COLORS["multi"])
    axes[0].set_title("학습 구간의 월별 평균 대여건수")
    axes[0].set_xlabel("월")
    axes[0].set_ylabel("평균 대여건수")
    axes[0].set_xticks(range(1, 13))
    axes[0].grid(axis="y")

    colors = [COLORS["green"]] * 5 + [COLORS["orange"]] * 2
    axes[1].bar(labels, weekday.values, color=colors)
    axes[1].set_title("학습 구간의 요일별 평균 대여건수")
    axes[1].set_xlabel("요일")
    axes[1].grid(axis="y")
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    fig.suptitle("계절과 요일에 따라 달라지는 평균 대여량", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "03_month_weekday_means.png")


def generate_simple_prediction(results: dict[str, object]) -> None:
    test = results["test"]
    y_test = results["y_test"]
    pred_simple = results["pred_simple"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(test["대여일자"], y_test.to_numpy(), label="실제", color=COLORS["actual"], linewidth=2)
    ax.plot(test["대여일자"], pred_simple, label="단순회귀 예측", color=COLORS["simple"], linewidth=2)
    ax.set_title("단순선형회귀: 마지막 60일의 실제 값과 예측")
    ax.set_xlabel("대여일자")
    ax.set_ylabel("대여건수")
    ax.set_xlim(test["대여일자"].min(), test["대여일자"].max())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    save_figure(fig, "04_simple_prediction.png")


def generate_model_comparison(results: dict[str, object]) -> None:
    test = results["test"]
    y_test = results["y_test"]
    pred_simple = results["pred_simple"]
    pred_multi = results["pred_multi"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(test["대여일자"], y_test.to_numpy(), label="실제", color=COLORS["actual"], linewidth=2.2)
    ax.plot(
        test["대여일자"],
        pred_simple,
        label="단순회귀 예측",
        color=COLORS["simple"],
        linewidth=1.8,
        linestyle="--",
    )
    ax.plot(test["대여일자"], pred_multi, label="다중회귀 예측", color=COLORS["multi"], linewidth=2)
    ax.set_title("단순회귀와 다중회귀: 마지막 60일 비교")
    ax.set_xlabel("대여일자")
    ax.set_ylabel("대여건수")
    ax.set_xlim(test["대여일자"].min(), test["대여일자"].max())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    save_figure(fig, "05_model_comparison.png")


def generate_residual_comparison(results: dict[str, object]) -> None:
    y_test = results["y_test"].to_numpy()
    pred_simple = results["pred_simple"]
    pred_multi = results["pred_multi"]
    residuals_simple = y_test - pred_simple
    residuals_multi = y_test - pred_multi

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    axes[0].scatter(pred_simple, residuals_simple, color=COLORS["simple"], alpha=0.75)
    axes[0].axhline(0, color="black", linewidth=1.2)
    axes[0].set_title("단순회귀 잔차")
    axes[0].set_xlabel("예측값")
    axes[0].set_ylabel("잔차(실제-예측)")

    axes[1].scatter(pred_multi, residuals_multi, color=COLORS["multi"], alpha=0.75)
    axes[1].axhline(0, color="black", linewidth=1.2)
    axes[1].set_title("다중회귀 잔차")
    axes[1].set_xlabel("예측값")
    for ax in axes:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
        ax.grid()
    fig.suptitle("같은 평가 구간에서 비교한 두 모델의 잔차 구조", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "06_residual_comparison.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    generate_toy_regression()
    df = load_daily_rentals()
    results = fit_models(df)
    generate_full_timeseries(df)
    generate_month_weekday_means(df)
    generate_simple_prediction(results)
    generate_model_comparison(results)
    generate_residual_comparison(results)


if __name__ == "__main__":
    main()
