"""Generate the deterministic figures embedded in 01주차/notion.md.

Run from any directory with:
    python 01주차/assets/generate_assets.py

The script reads only the local course datasets. It never reads API credentials and
does not make network requests.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


ASSET_DIR = Path(__file__).resolve().parent
WEEK_DIR = ASSET_DIR.parent
PROJECT_ROOT = WEEK_DIR.parent

RENTAL_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "extracted"
    / "따릉이 공공데이터"
    / "03_대여이력"
    / "서울특별시 공공자전거 대여이력 정보_2602.csv"
)
KRX_INDEX_PATH = (
    PROJECT_ROOT / "dataset" / "open" / "extracted" / "KRX" / "krx_index_20240823.csv"
)
KOSDAQ_PATH = (
    PROJECT_ROOT / "dataset" / "open" / "extracted" / "KRX" / "kosdaq_stocks_20240823.csv"
)

COLORS = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "yellow": "#F2CF5B",
}


def configure_style() -> None:
    """Apply a stable, Korean-capable plotting style."""

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


def generate_toy_bar_iqr() -> None:
    toy = pd.DataFrame(
        {
            "요일": ["월", "화", "수", "목", "금"],
            "대여건수": [320, 280, 350, 300, 410],
        }
    )
    rental_counts = pd.Series([10, 12, 11, 13, 12, 14, 11, 90])
    q1 = rental_counts.quantile(0.25)
    q3 = rental_counts.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    is_outlier = (rental_counts < lower) | (rental_counts > upper)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    axes[0].bar(toy["요일"], toy["대여건수"], color=COLORS["blue"])
    axes[0].set_title("요일별 대여건수 (예제 데이터)")
    axes[0].set_xlabel("요일")
    axes[0].set_ylabel("대여건수")
    axes[0].grid(axis="y")

    y = np.zeros(len(rental_counts))
    axes[1].scatter(
        rental_counts[~is_outlier],
        y[~is_outlier],
        s=80,
        color=COLORS["blue"],
        label="IQR 경계 안",
        zorder=3,
    )
    axes[1].scatter(
        rental_counts[is_outlier],
        y[is_outlier],
        s=110,
        color=COLORS["red"],
        label="잠재적 이상치 후보",
        zorder=4,
    )
    axes[1].axvspan(lower, upper, color=COLORS["green"], alpha=0.12)
    axes[1].axvline(lower, color=COLORS["green"], linestyle="--", linewidth=1.5)
    axes[1].axvline(upper, color=COLORS["green"], linestyle="--", linewidth=1.5)
    axes[1].annotate(
        "90",
        xy=(90, 0),
        xytext=(78, 0.28),
        arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
        color=COLORS["red"],
        fontsize=11,
        weight="bold",
    )
    axes[1].set_title("IQR 경계로 후보 표시")
    axes[1].set_xlabel("대여건수")
    axes[1].set_yticks([])
    axes[1].set_ylim(-0.45, 0.55)
    axes[1].legend(loc="upper center")
    axes[1].grid(axis="x")
    axes[1].text(
        0.02,
        0.05,
        f"Q1={q1:.2f}, Q3={q3:.2f}\nIQR 경계: {lower:.3f} ~ {upper:.3f}",
        transform=axes[1].transAxes,
        fontsize=10,
        va="bottom",
    )
    fig.suptitle("큰 값과 IQR 이상치 후보는 같은 개념이 아닙니다", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "01_toy_bar_iqr.png")


def load_rentals() -> pd.DataFrame:
    if not RENTAL_PATH.exists():
        raise FileNotFoundError(f"따릉이 원본 파일이 없습니다: {RENTAL_PATH}")
    cols = ["대여일시", "이용시간(분)", "이용거리(M)"]
    df = pd.read_csv(RENTAL_PATH, encoding="cp949", usecols=cols)
    dt = pd.to_datetime(df["대여일시"])
    df["시간대"] = dt.dt.hour
    df["요일"] = dt.dt.day_name()
    return df


def generate_hourly(df: pd.DataFrame) -> None:
    hourly = df.groupby("시간대").size().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(hourly.index, hourly.values, color=COLORS["blue"])
    for hour in (8, 18):
        bars[hour].set_color(COLORS["orange"])
        ax.annotate(
            f"{hour}시\n{hourly.loc[hour]:,}건",
            xy=(hour, hourly.loc[hour]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            weight="bold",
        )
    ax.set_title("시간대별 따릉이 대여건수 — 2026년 2월")
    ax.set_xlabel("시간대(시)")
    ax.set_ylabel("대여건수")
    ax.set_xticks(range(24))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.grid(axis="y")
    fig.tight_layout()
    save_figure(fig, "02_hourly_rentals.png")


def generate_weekday_hour_heatmap(df: pd.DataFrame) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    labels = ["월", "화", "수", "목", "금", "토", "일"]
    pivot = df.pivot_table(
        index="요일",
        columns="시간대",
        values="대여일시",
        aggfunc="count",
        fill_value=0,
    ).reindex(index=order, columns=range(24), fill_value=0)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xticks(range(24), labels=range(24))
    ax.set_xlabel("시간대(시)")
    ax.set_ylabel("요일")
    ax.set_title("요일 × 시간대 따릉이 대여건수 — 2026년 2월")
    colorbar = fig.colorbar(im, ax=ax, pad=0.02)
    colorbar.set_label("대여건수")
    colorbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    fig.tight_layout()
    save_figure(fig, "03_weekday_hour_heatmap.png")


def generate_distribution_pair(df: pd.DataFrame) -> None:
    duration = df["이용시간(분)"].dropna()
    distance = df["이용거리(M)"].dropna()
    cutoff = distance.quantile(0.99)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].boxplot(
        duration,
        orientation="vertical",
        patch_artist=True,
        boxprops={"facecolor": COLORS["blue"], "alpha": 0.55},
        medianprops={"color": "black", "linewidth": 1.8},
        flierprops={
            "marker": ".",
            "markersize": 1.2,
            "markerfacecolor": COLORS["red"],
            "markeredgecolor": COLORS["red"],
            "alpha": 0.18,
        },
    )
    axes[0].set_title("이용시간 전체 분포")
    axes[0].set_ylabel("이용시간(분)")
    axes[0].set_xticks([1], ["전체 대여"])
    axes[0].grid(axis="y")

    axes[1].hist(
        distance[distance <= cutoff],
        bins=40,
        color=COLORS["green"],
        edgecolor="white",
        linewidth=0.4,
    )
    axes[1].set_title("이용거리 분포 (상위 1% 제외)")
    axes[1].set_xlabel("이용거리(M)")
    axes[1].set_ylabel("대여건수")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    axes[1].grid(axis="y")
    axes[1].text(
        0.98,
        0.95,
        f"99% 경계: {cutoff:,.0f}M",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )
    fig.suptitle("전체 극단 범위와 분포 본체를 함께 읽기", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, "04_duration_distance_distributions.png")


def generate_krx_panels() -> None:
    if not KRX_INDEX_PATH.exists() or not KOSDAQ_PATH.exists():
        raise FileNotFoundError("2024-08-23 KRX 추출 CSV 두 개가 필요합니다.")
    krx_index = pd.read_csv(KRX_INDEX_PATH)
    kosdaq = pd.read_csv(KOSDAQ_PATH, dtype={"ISU_CD": "string"})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    index_missing = krx_index.isna().sum()
    missing_only = index_missing[index_missing > 0]
    missing_only.plot(kind="bar", ax=axes[0], color="#5B8FF9")
    axes[0].set_title("KRX 지수: 필드별 결측 개수")
    axes[0].set_ylabel("결측 행 수")
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].grid(axis="y")

    kosdaq["FLUC_RT"].plot(kind="hist", bins=40, ax=axes[1], color="#61DDAA")
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("코스닥 종목 등락률 분포")
    axes[1].set_xlabel("등락률(%)")
    axes[1].set_ylabel("종목 수")
    axes[1].grid(axis="y")

    top_value = kosdaq.nlargest(10, "ACC_TRDVAL").sort_values("ACC_TRDVAL")
    axes[2].barh(top_value["ISU_NM"], top_value["ACC_TRDVAL"] / 1e8, color="#F6BD16")
    axes[2].set_title("거래대금 상위 10개 종목")
    axes[2].set_xlabel("거래대금(억원)")
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[2].grid(axis="x")

    fig.suptitle("KRX OPEN API 스냅샷 — 2024-08-23", fontsize=17, weight="bold")
    fig.text(0.99, 0.01, "출처: 한국거래소 통계정보", ha="right", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_figure(fig, "05_krx_snapshot_panels.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    generate_toy_bar_iqr()
    rentals = load_rentals()
    generate_hourly(rentals)
    generate_weekday_hour_heatmap(rentals)
    generate_distribution_pair(rentals)
    del rentals
    generate_krx_panels()


if __name__ == "__main__":
    main()
