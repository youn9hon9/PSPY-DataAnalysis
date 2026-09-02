"""Generate deterministic visual assets for the week 5 classification chapter.

Run from any directory:
    python 05주차/assets/generate_assets.py
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
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ASSET_DIR = Path(__file__).resolve().parent
REPO_ROOT = ASSET_DIR.parents[1]
DATA_PATH = (
    REPO_ROOT
    / "dataset"
    / "extracted"
    / "Loan Prediction Problem Dataset"
    / "train_u6lujuX_CVtuZ9i.csv"
)

DPI = 160
BLUE = "#2563EB"
ORANGE = "#F59E0B"
RED = "#DC2626"
NAVY = "#0F172A"
GRAY = "#64748B"
LIGHT_BLUE = "#DBEAFE"

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


def draw_confusion_matrix(
    matrix: np.ndarray,
    class_labels: list[str],
    cell_names: np.ndarray,
    title: str,
    filename: str,
    emphasize: tuple[int, int] | None = None,
) -> None:
    """Render a 2x2 confusion matrix with semantic cell names."""

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="개수")

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=class_labels,
        yticklabels=class_labels,
        xlabel="예측 클래스",
        ylabel="실제 클래스",
        title=title,
    )

    midpoint = matrix.max() / 2
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col])
            color = "white" if value > midpoint else NAVY
            ax.text(
                col,
                row,
                f"{cell_names[row, col]}\n{value}",
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=color,
            )

    if emphasize is not None:
        row, col = emphasize
        rectangle = plt.Rectangle(
            (col - 0.49, row - 0.49),
            0.98,
            0.98,
            fill=False,
            edgecolor=RED,
            linewidth=3,
        )
        ax.add_patch(rectangle)
        ax.text(
            0.50,
            -0.16,
            "Red box: operational risk to inspect",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=RED,
            fontweight="bold",
        )

    fig.tight_layout()
    save_figure(fig, filename)


def load_and_fit_model() -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Reproduce Parts 2-8 of the week 5 notebook."""

    df = pd.read_csv(DATA_PATH)

    df["target"] = (df["Loan_Status"] == "Y").astype(int)

    categorical_features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]
    numeric_features = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    features = df[categorical_features + numeric_features].copy()
    target = df["target"]

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(max_iter=5000)),
        ]
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_valid)

    return df, y_valid, prediction, np.ones(len(y_valid), dtype=int)


def make_toy_confusion_matrix() -> None:
    actual = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    predicted = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    matrix = confusion_matrix(actual, predicted)

    draw_confusion_matrix(
        matrix=matrix,
        class_labels=["불합격 (0)", "합격 (1)"],
        cell_names=np.array([["TN", "FP"], ["FN", "TP"]]),
        title="합격 예측 예제의 혼동행렬",
        filename="toy_confusion_matrix.png",
    )


def make_sigmoid_thresholds() -> None:
    z = np.linspace(-8, 8, 800)
    probability = 1 / (1 + np.exp(-z))
    thresholds = [0.3, 0.5, 0.7]
    colors = [BLUE, GRAY, ORANGE]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(z, probability, color=NAVY, linewidth=2.5, label="Sigmoid probability")

    for threshold, color in zip(thresholds, colors):
        boundary = np.log(threshold / (1 - threshold))
        ax.axhline(threshold, color=color, linestyle="--", alpha=0.75)
        ax.axvline(boundary, color=color, linestyle=":", alpha=0.75)
        ax.scatter([boundary], [threshold], s=55, color=color, zorder=3)
        ax.text(
            boundary + 0.12,
            threshold + 0.035,
            f"threshold={threshold:.1f}\nz={boundary:.2f}",
            color=color,
            fontsize=9,
        )

    ax.fill_between(
        z,
        0.5,
        probability,
        where=probability >= 0.5,
        color=LIGHT_BLUE,
        alpha=0.45,
        label="Positive region at threshold 0.5",
    )
    ax.set(
        xlabel="Linear score z",
        ylabel="P(y=1 | X)",
        title="Sigmoid Probability and Classification Thresholds",
        xlim=(-8, 8),
        ylim=(-0.03, 1.05),
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, "sigmoid_thresholds.png")


def make_target_distribution(df: pd.DataFrame) -> None:
    order = ["N", "Y"]
    counts = df["Loan_Status"].value_counts().reindex(order)
    proportions = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    bars = ax.bar(
        ["Rejected (N)", "Approved (Y)"],
        counts.to_numpy(),
        color=[ORANGE, BLUE],
        width=0.58,
    )

    for bar, count, proportion in zip(bars, counts, proportions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{int(count)}\n({proportion:.1%})",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    majority_count = int(counts.max())
    ax.axhline(
        majority_count,
        color=GRAY,
        linestyle="--",
        linewidth=1.4,
        label=f"Majority baseline: {proportions.max():.1%}",
    )
    ax.set(
        ylabel="Applications",
        title="Loan Approval Target Distribution",
        ylim=(0, majority_count * 1.18),
    )
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "loan_target_distribution.png")


def make_model_confusion_matrix(y_valid: pd.Series, prediction: np.ndarray) -> None:
    matrix = confusion_matrix(y_valid, prediction)

    draw_confusion_matrix(
        matrix=matrix,
        class_labels=["Rejected (N)", "Approved (Y)"],
        cell_names=np.array([["TN", "FP"], ["FN", "TP"]]),
        title="Logistic Regression Confusion Matrix",
        filename="model_confusion_matrix.png",
        emphasize=(0, 1),
    )


def make_model_metric_comparison(
    y_valid: pd.Series,
    prediction: np.ndarray,
    dummy_prediction: np.ndarray,
) -> None:
    metric_names = ["Accuracy", "Y precision", "Y recall", "N recall"]

    def calculate_metrics(predicted: np.ndarray) -> list[float]:
        return [
            accuracy_score(y_valid, predicted),
            precision_score(y_valid, predicted, zero_division=0),
            recall_score(y_valid, predicted, zero_division=0),
            recall_score(y_valid, predicted, pos_label=0, zero_division=0),
        ]

    dummy_metrics = calculate_metrics(dummy_prediction)
    model_metrics = calculate_metrics(prediction)

    x_positions = np.arange(len(metric_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    dummy_bars = ax.bar(
        x_positions - width / 2,
        dummy_metrics,
        width,
        label="Always approve",
        color="#CBD5E1",
        edgecolor=GRAY,
    )
    model_bars = ax.bar(
        x_positions + width / 2,
        model_metrics,
        width,
        label="Logistic regression",
        color=BLUE,
    )

    for bars in (dummy_bars, model_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set(
        xticks=x_positions,
        xticklabels=metric_names,
        ylabel="Score",
        title="Majority Baseline vs Logistic Regression",
        ylim=(0, 1.13),
    )
    ax.axhline(1.0, color="#E2E8F0", linewidth=1)
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "model_metric_comparison.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    make_toy_confusion_matrix()
    make_sigmoid_thresholds()

    dataframe, y_valid, prediction, dummy_prediction = load_and_fit_model()
    make_target_distribution(dataframe)
    make_model_confusion_matrix(y_valid, prediction)
    make_model_metric_comparison(y_valid, prediction, dummy_prediction)

    generated = sorted(path.name for path in ASSET_DIR.glob("*.png"))
    print("Generated:", ", ".join(generated))


if __name__ == "__main__":
    main()
