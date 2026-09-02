"""7주차의 데이터 비의존 개념도를 PNG로 생성합니다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ASSET_DIR = Path(__file__).resolve().parent
NAVY = "#183B56"
BLUE = "#3A7BD5"
ORANGE = "#F59E0B"
RED = "#D64550"
GREEN = "#2AA198"
GRAY = "#667085"
LIGHT = "#F4F7FB"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(ASSET_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def facility_location_concept() -> None:
    demand = np.array([
        [0.9, 1.0], [1.5, 2.1], [2.0, 3.7], [2.9, 1.4], [3.3, 3.0],
        [4.2, 4.1], [4.7, 1.0], [5.2, 2.4], [6.2, 3.8], [6.7, 1.5],
        [7.6, 2.7], [8.2, 4.0], [8.6, 1.0],
    ])
    existing = np.array([[2.3, 2.4], [6.2, 2.2]])
    candidates = np.array([[4.5, 2.7], [7.8, 1.6], [8.0, 3.7]])
    radius = 1.65
    covered = np.zeros(len(demand), dtype=bool)
    for site in existing:
        covered |= np.linalg.norm(demand - site, axis=1) <= radius

    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0.2, 5.1)
    ax.set_aspect("equal")
    ax.set_facecolor("#FAFCFE")
    ax.grid(color="#E5EAF0", linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_title("수요가 있으면서 기존 시설 반경 밖인 곳이 ‘사각지대’입니다", fontsize=20, weight="bold", color=NAVY, pad=16)

    for site in existing:
        ax.add_patch(Circle(site, radius, facecolor=BLUE, edgecolor=BLUE, alpha=0.12, linewidth=2))
    ax.scatter(demand[covered, 0], demand[covered, 1], s=95, c="#6FB1E5", edgecolor="white", linewidth=1.2, zorder=4)
    ax.scatter(demand[~covered, 0], demand[~covered, 1], s=115, c=RED, marker="o", edgecolor="white", linewidth=1.2, zorder=4)
    ax.scatter(existing[:, 0], existing[:, 1], s=180, c=BLUE, marker="s", edgecolor="white", linewidth=1.5, zorder=5)
    ax.scatter(candidates[:, 0], candidates[:, 1], s=190, c=ORANGE, marker="^", edgecolor="white", linewidth=1.5, zorder=5)
    ax.annotate("사람이 있지만\n기존 반경 밖", xy=(8.2, 4.0), xytext=(6.9, 4.65), color=RED, fontsize=12, weight="bold", arrowprops={"arrowstyle": "->", "color": RED, "linewidth": 1.8})

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6FB1E5", markeredgecolor="white", markersize=10, label="커버된 수요"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markeredgecolor="white", markersize=10, label="미커버 수요"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=BLUE, markersize=11, label="기존 시설"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE, markersize=11, label="신규 후보지"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=11)
    ax.text(0.15, 0.35, "빈 공간 ≠ 사각지대\n수요가 있는 미커버 공간이 분석 대상입니다.", fontsize=12, color=NAVY, bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#D0D5DD"})

    save(fig, "facility-location-concept.png")


def haversine_concept() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.2, 1]})
    fig.suptitle("하버사인은 구면 위 두 좌표 사이의 대권거리입니다", fontsize=20, weight="bold", color=NAVY)

    ax = axes[0]
    ax.set_aspect("equal")
    ax.axis("off")
    earth = Circle((0, 0), 2.3, facecolor="#EAF4FB", edgecolor=BLUE, linewidth=2)
    ax.add_patch(earth)
    theta = np.linspace(np.deg2rad(40), np.deg2rad(105), 120)
    ax.plot(2.3 * np.cos(theta), 2.3 * np.sin(theta), color=ORANGE, linewidth=4)
    start = np.array([2.3 * np.cos(theta[0]), 2.3 * np.sin(theta[0])])
    end = np.array([2.3 * np.cos(theta[-1]), 2.3 * np.sin(theta[-1])])
    ax.scatter([start[0], end[0]], [start[1], end[1]], s=120, c=[BLUE, RED], edgecolor="white", linewidth=1.5, zorder=5)
    ax.text(start[0] + 0.15, start[1] - 0.05, "서울시청\n37.5665, 126.9780", fontsize=11, color=NAVY)
    ax.text(end[0] - 1.0, end[1] + 0.18, "광화문\n37.5716, 126.9769", fontsize=11, color=NAVY)
    ax.text(0.15, 2.18, "대권 경로(개념도)", fontsize=12, color=ORANGE, weight="bold")
    ax.text(0, -0.3, "지구 반지름 R\n위도·경도의 각도 차이", ha="center", va="center", fontsize=12, color=GRAY)
    ax.set_xlim(-3.1, 3.2)
    ax.set_ylim(-2.8, 3.0)

    ax = axes[1]
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.08, 0.58), 0.84, 0.28, boxstyle="round,pad=0.03", facecolor="#FFF7E8", edgecolor=ORANGE, linewidth=2, transform=ax.transAxes))
    ax.text(0.5, 0.76, "하버사인 거리", ha="center", va="center", fontsize=15, color=NAVY, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.65, "575.3 m", ha="center", va="center", fontsize=28, color=ORANGE, weight="bold", transform=ax.transAxes)
    ax.add_patch(FancyArrowPatch((0.5, 0.56), (0.5, 0.43), arrowstyle="-|>", mutation_scale=18, color="#98A2B3", transform=ax.transAxes))
    ax.add_patch(FancyBboxPatch((0.08, 0.12), 0.84, 0.28, boxstyle="round,pad=0.03", facecolor=LIGHT, edgecolor="#98A2B3", linewidth=1.5, transform=ax.transAxes))
    ax.text(0.5, 0.30, "실제 보행 거리·시간", ha="center", va="center", fontsize=15, color=NAVY, weight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.20, "도로·횡단보도·출입구·장애물을 반영해야 합니다", ha="center", va="center", fontsize=11, color=GRAY, transform=ax.transAxes)
    ax.text(0.5, 0.02, "※ 거리와 지구 크기는 설명을 위해 과장했으며 실제 축척이 아닙니다.", ha="center", fontsize=9.5, color=GRAY, transform=ax.transAxes)

    save(fig, "haversine-distance-concept.png")


def greedy_coverage_toy() -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0.3, 5.7)
    ax.set_ylim(0.1, 4.7)
    ax.axis("off")
    ax.set_title("전체 커버 수보다 ‘아직 커버되지 않은 수요’를 얼마나 추가하는지가 중요합니다", fontsize=19, weight="bold", color=NAVY, pad=18)

    demand_x = np.arange(1, 6)
    for x, label in zip(demand_x, list("ABCDE")):
        ax.scatter(x, 3.45, s=180, color=NAVY, edgecolor="white", linewidth=1.5, zorder=5)
        ax.text(x, 3.45, label, ha="center", va="center", color="white", fontsize=12, weight="bold", zorder=6)

    rows = [
        ("후보지 1", 0.72, 3.0, 2.55, BLUE, "A · B · C", True),
        ("후보지 2", 1.72, 3.0, 1.65, "#98A2B3", "B · C · D", False),
        ("후보지 3", 3.72, 2.0, 0.75, GREEN, "D · E", True),
    ]
    for name, x, width, y, color, covered_text, selected in rows:
        patch = FancyBboxPatch(
            (x, y),
            width,
            0.55,
            boxstyle="round,pad=0.03,rounding_size=0.20",
            facecolor=color,
            alpha=0.20 if selected else 0.10,
            edgecolor=color,
            linewidth=3 if selected else 2,
            linestyle="-" if selected else "--",
        )
        ax.add_patch(patch)
        ax.text(0.42, y + 0.28, name, va="center", fontsize=12, color=color, weight="bold")
        ax.text(5.5, y + 0.28, covered_text, ha="right", va="center", fontsize=12, color=GRAY)
        if selected:
            ax.text(4.95, y + 0.28, "선택", ha="right", va="center", fontsize=10, color=color, weight="bold")

    ax.text(0.55, 4.1, "수요 지점", fontsize=12, color=NAVY, weight="bold")
    ax.add_patch(FancyBboxPatch((0.75, 0.12), 4.5, 0.38, boxstyle="round,pad=0.04", facecolor=LIGHT, edgecolor="none"))
    ax.text(3.0, 0.31, "후보지 1 선택 후: 후보지 2는 D만 추가 · 후보지 3은 D와 E를 추가 → 후보지 3 선택", ha="center", va="center", fontsize=12, color=NAVY, weight="bold")
    save(fig, "greedy-coverage-toy.png")


if __name__ == "__main__":
    facility_location_concept()
    haversine_concept()
    greedy_coverage_toy()
    print("Week 7 assets generated: 3")
