"""Generate deterministic visual assets for the week 6 clustering chapter.

Run from any directory:
    python 06주차/assets/generate_assets.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ASSET_DIR = Path(__file__).resolve().parent
REPO_ROOT = ASSET_DIR.parents[1]
DATA_PATH = (
    REPO_ROOT
    / "dataset"
    / "extracted"
    / "Customer Personality Analysis"
    / "marketing_campaign.csv"
)

FEATURES = [
    "Income",
    "Age",
    "Total_Spending",
    "Recency",
    "Children_at_Home",
    "Purchase_Activity_Sum",
]

DPI = 160
BLUE = "#2563EB"
ORANGE = "#F59E0B"
RED = "#DC2626"
GREEN = "#059669"
NAVY = "#0F172A"
GRAY = "#64748B"
LIGHT_GRAY = "#E2E8F0"

plt.rcParams.update(
    {
        "font.family": "Malgun Gothic",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#CBD5E1",
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a white-background PNG with stable dimensions and metadata."""

    output_path = ASSET_DIR / filename
    fig.savefig(
        output_path,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "PSPY deterministic asset generator"},
    )
    plt.close(fig)


def canonicalize_labels(labels: np.ndarray, ordering_values: np.ndarray) -> np.ndarray:
    """Map arbitrary K-means labels to 0..K-1 by ascending group mean."""

    unique_labels = np.unique(labels)
    ordered_labels = sorted(
        unique_labels,
        key=lambda label: float(ordering_values[labels == label].mean()),
    )
    mapping = {label: index for index, label in enumerate(ordered_labels)}
    return np.array([mapping[label] for label in labels], dtype=int)


def load_customer_data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Reproduce Parts 2-5 of the week 6 notebook."""

    df = pd.read_csv(DATA_PATH, sep="\t")
    df = df.dropna(subset=["Income"]).reset_index(drop=True)

    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    reference_year = int(df["Dt_Customer"].dt.year.max())
    df["Age"] = reference_year - df["Year_Birth"]

    spending_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",
    ]
    df["Total_Spending"] = df[spending_columns].sum(axis=1)

    purchase_columns = [
        "NumDealsPurchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",
    ]
    df["Purchase_Activity_Sum"] = df[purchase_columns].sum(axis=1)
    df["Children_at_Home"] = df["Kidhome"] + df["Teenhome"]

    before_outlier_filter = df.copy()
    filtered = df[
        (df["Age"] <= 100)
        & (df["Income"] < 200_000)
    ].reset_index(drop=True)

    scaled = StandardScaler().fit_transform(filtered[FEATURES])
    return before_outlier_filter, filtered, scaled


def make_center_movement() -> None:
    points = np.array([1, 2, 3, 20, 21, 22], dtype=float)
    initial_centers = np.array([1.0, 20.0])

    distances = np.abs(points[:, None] - initial_centers[None, :])
    initial_labels = distances.argmin(axis=1)
    updated_centers = np.array(
        [points[initial_labels == label].mean() for label in range(2)]
    )
    updated_distances = np.abs(points[:, None] - updated_centers[None, :])
    updated_labels = updated_distances.argmin(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharex=True, sharey=True)
    frames = [
        ("Frame 1: assign to initial centers", initial_centers, initial_labels),
        ("Frame 2: move centers to means", updated_centers, updated_labels),
    ]

    for ax, (title, centers, labels) in zip(axes, frames):
        colors = np.where(labels == 0, BLUE, ORANGE)
        ax.scatter(points, np.zeros_like(points), c=colors, s=85, zorder=3)
        ax.scatter(
            centers,
            np.full_like(centers, 0.35),
            marker="X",
            s=180,
            c=[BLUE, ORANGE],
            edgecolor="white",
            linewidth=1.2,
            zorder=4,
        )

        boundary = centers.mean()
        ax.axvline(boundary, color=GRAY, linestyle="--", alpha=0.7)
        for point in points:
            ax.text(point, -0.13, f"{point:g}", ha="center", fontsize=9)
        for center_index, center in enumerate(centers):
            ax.text(
                center,
                0.58,
                f"c{center_index + 1}={center:g}",
                ha="center",
                color=[BLUE, ORANGE][center_index],
                fontweight="bold",
            )

        ax.set(
            title=title,
            xlabel="Point value",
            ylim=(-0.28, 0.75),
            yticks=[],
        )
        ax.spines[["left", "right", "top"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)

    axes[1].annotate(
        "",
        xy=(2.0, 0.35),
        xytext=(1.0, 0.35),
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.8},
    )
    axes[1].annotate(
        "",
        xy=(21.0, 0.35),
        xytext=(20.0, 0.35),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.8},
    )

    fig.suptitle("K-means: Assignment and Centroid Update", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "kmeans_center_movement.png")


def make_scaling_comparison() -> None:
    toy = pd.DataFrame(
        {
            "Income": [30000, 30500, 31000, 30200, 30800, 30100],
            "Satisfaction": [2, 2, 3, 8, 9, 8],
        }
    )

    raw_model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(toy)
    scaled_values = StandardScaler().fit_transform(toy)
    scaled_model = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaled_values)

    raw_labels = canonicalize_labels(
        raw_model.labels_,
        toy["Income"].to_numpy(dtype=float),
    )
    scaled_labels = canonicalize_labels(
        scaled_model.labels_,
        toy["Satisfaction"].to_numpy(dtype=float),
    )
    colors = np.array([BLUE, ORANGE])

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))

    axes[0].scatter(
        toy["Income"],
        toy["Satisfaction"],
        c=colors[raw_labels],
        s=90,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].set(
        title="Without scaling: income dominates",
        xlabel="Income",
        ylabel="Satisfaction",
    )
    axes[0].ticklabel_format(axis="x", style="plain", useOffset=False)

    axes[1].scatter(
        scaled_values[:, 0],
        scaled_values[:, 1],
        c=colors[scaled_labels],
        s=90,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[1].set(
        title="After StandardScaler: signal separates",
        xlabel="Standardized income",
        ylabel="Standardized satisfaction",
    )

    for index, (income, satisfaction) in enumerate(toy.to_numpy()):
        axes[0].annotate(
            f"P{index + 1}",
            (income, satisfaction),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )
        axes[1].annotate(
            f"P{index + 1}",
            scaled_values[index],
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, label="Cluster A"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, label="Cluster B"),
    ]
    axes[1].legend(handles=legend_handles, frameon=False, loc="best")

    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Scaling Changes the Meaning of Distance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "scaling_before_after.png")


def make_outlier_diagnostics(before_filter: pd.DataFrame) -> None:
    age = before_filter["Age"].to_numpy(dtype=float)
    income = before_filter["Income"].to_numpy(dtype=float)
    age_candidates = age[age > 100]
    income_candidates = income[income >= 200_000]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))

    axes[0].boxplot(
        age,
        orientation="horizontal",
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#DBEAFE", "edgecolor": BLUE},
        medianprops={"color": NAVY, "linewidth": 1.8},
        whiskerprops={"color": BLUE},
        capprops={"color": BLUE},
    )
    axes[0].scatter(
        age_candidates,
        np.ones(len(age_candidates)),
        s=65,
        color=RED,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label=f"나이 > 100세 (n={len(age_candidates)})",
    )
    axes[0].axvline(100, color=RED, linestyle="--", linewidth=1.2)
    axes[0].set(
        title="나이 기준 이상치 후보",
        xlabel="나이(2014년 기준)",
        yticks=[],
    )
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].boxplot(
        income,
        orientation="horizontal",
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#FEF3C7", "edgecolor": ORANGE},
        medianprops={"color": NAVY, "linewidth": 1.8},
        whiskerprops={"color": ORANGE},
        capprops={"color": ORANGE},
    )
    axes[1].scatter(
        income_candidates,
        np.ones(len(income_candidates)),
        s=65,
        color=RED,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label=f"소득 ≥ 200,000 (n={len(income_candidates)})",
    )
    axes[1].axvline(200_000, color=RED, linestyle="--", linewidth=1.2)
    if len(income_candidates):
        axes[1].annotate(
            f"{int(income_candidates.max()):,}",
            xy=(income_candidates.max(), 1),
            xytext=(-8, 20),
            textcoords="offset points",
            ha="right",
            color=RED,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": RED},
        )
    axes[1].set(
        title="소득 기준 이상치 후보",
        xlabel="소득",
        yticks=[],
    )
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    axes[1].legend(frameon=False, loc="upper center")

    for ax in axes:
        ax.grid(axis="x", alpha=0.18)
        ax.spines[["top", "right", "left"]].set_visible(False)

    kept_count = int(
        ((before_filter["Age"] <= 100) & (before_filter["Income"] < 200_000)).sum()
    )
    fig.suptitle(
        f"K-means 적용 전 이상치 후보 점검: {len(before_filter):,}행 → {kept_count:,}행",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "age_income_outliers.png")


def fit_k_candidates(
    scaled: np.ndarray,
) -> tuple[list[int], list[float], list[float]]:
    k_values = list(range(2, 9))
    inertias: list[float] = []
    silhouettes: list[float] = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(scaled, model.labels_)))

    return k_values, inertias, silhouettes


def make_k_selection(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))

    axes[0].plot(k_values, inertias, marker="o", color=BLUE, linewidth=2)
    axes[0].set(
        title="Elbow Method",
        xlabel="Number of clusters K",
        ylabel="Inertia",
        xticks=k_values,
    )

    axes[1].plot(k_values, silhouettes, marker="o", color=ORANGE, linewidth=2)
    axes[1].set(
        title="Silhouette Score",
        xlabel="Number of clusters K",
        ylabel="Average silhouette",
        xticks=k_values,
    )

    for ax in axes:
        ax.axvline(2, color=GREEN, linestyle="--", alpha=0.8, label="K=2: best silhouette")
        ax.axvline(4, color=RED, linestyle=":", alpha=0.85, label="K=4: chapter candidate")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].annotate(
        f"{inertias[0]:,.0f}",
        (2, inertias[0]),
        xytext=(5, -18),
        textcoords="offset points",
        color=GREEN,
        fontweight="bold",
    )
    axes[0].annotate(
        f"{inertias[2]:,.0f}",
        (4, inertias[2]),
        xytext=(5, 8),
        textcoords="offset points",
        color=RED,
        fontweight="bold",
    )
    axes[1].annotate(
        f"{silhouettes[0]:.3f}",
        (2, silhouettes[0]),
        xytext=(5, -18),
        textcoords="offset points",
        color=GREEN,
        fontweight="bold",
    )
    axes[1].annotate(
        f"{silhouettes[2]:.3f}",
        (4, silhouettes[2]),
        xytext=(5, 8),
        textcoords="offset points",
        color=RED,
        fontweight="bold",
    )
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle("K Selection Needs More Than One Metric", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "k_selection_metrics.png")


def semantic_cluster_order(summary: pd.DataFrame) -> list[tuple[int, str]]:
    """Assign stable, human-readable names from profile characteristics."""

    remaining = set(int(index) for index in summary.index)

    vip = int(summary["Total_Spending"].idxmax())
    remaining.remove(vip)

    children_saver = int(summary.loc[list(remaining), "Children_at_Home"].idxmax())
    remaining.remove(children_saver)

    young_low_engagement = int(summary.loc[list(remaining), "Age"].idxmin())
    remaining.remove(young_low_engagement)

    premium = remaining.pop()

    return [
        (vip, "VIP"),
        (premium, "Premium"),
        (children_saver, "Children saver"),
        (young_low_engagement, "Young low-eng."),
    ]


def make_cluster_profiles(filtered: pd.DataFrame, scaled: np.ndarray) -> None:
    model = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled)
    profiled = filtered.copy()
    profiled["Cluster"] = model.labels_

    summary = profiled.groupby("Cluster")[FEATURES].mean()
    counts = profiled["Cluster"].value_counts()
    ordered_clusters = semantic_cluster_order(summary)

    cluster_ids = [cluster for cluster, _ in ordered_clusters]
    names = [name for _, name in ordered_clusters]
    labels = [
        f"{name}\n(n={int(counts.loc[cluster])})"
        for cluster, name in ordered_clusters
    ]
    colors = [BLUE, GREEN, ORANGE, GRAY]

    panels = [
        ("Income", "Mean income", "{:,.0f}"),
        ("Total_Spending", "Mean total spending", "{:,.0f}"),
        ("Children_at_Home", "Mean children at home", "{:.2f}"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))

    for ax, (column, title, value_format) in zip(axes, panels):
        values = summary.loc[cluster_ids, column].to_numpy(dtype=float)
        bars = ax.bar(labels, values, color=colors, width=0.68)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.035,
                value_format.format(value),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

        ax.set(title=title, ylim=(0, max(values) * 1.18))
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "K=4 Customer Profiles in Original Units",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "cluster_profiles.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    make_center_movement()
    make_scaling_comparison()

    before_filter, filtered, scaled = load_customer_data()
    make_outlier_diagnostics(before_filter)

    k_values, inertias, silhouettes = fit_k_candidates(scaled)
    make_k_selection(k_values, inertias, silhouettes)
    make_cluster_profiles(filtered, scaled)

    generated = sorted(path.name for path in ASSET_DIR.glob("*.png"))
    print("Generated:", ", ".join(generated))


if __name__ == "__main__":
    main()
