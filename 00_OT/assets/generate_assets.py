"""OT의 정적 교육용 시각 자료를 PNG로 생성합니다."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ASSET_DIR = Path(__file__).resolve().parent
NAVY = "#183B56"
BLUE = "#3A7BD5"
TEAL = "#2AA198"
ORANGE = "#F59E0B"
LIGHT = "#F4F7FB"
MUTED = "#667085"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(ASSET_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def curriculum_roadmap() -> None:
    labels = [
        ("1주차", "EDA"),
        ("2주차", "회귀"),
        ("3주차", "트리"),
        ("4주차", "랜덤포레스트"),
        ("5주차", "분류"),
        ("6주차", "군집"),
        ("7주차", "공간분석"),
        ("8주차", "프로젝트"),
    ]
    colors = [BLUE, "#557FDB", "#6B73D6", "#7A69CA", "#925CB7", "#A8519E", ORANGE, TEAL]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(-0.7, 7.7)
    ax.set_ylim(-1.8, 2.2)
    ax.axis("off")
    ax.set_title("데이터 점검에서 프로젝트 발표까지 이어지는 8주 학습 여정", fontsize=22, weight="bold", color=NAVY, pad=18)

    for index, ((week, topic), color) in enumerate(zip(labels, colors)):
        if index < len(labels) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (index + 0.42, 0.65),
                    (index + 0.58, 0.65),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=2,
                    color="#A9B4C4",
                )
            )
        box = FancyBboxPatch(
            (index - 0.42, 0.15),
            0.84,
            1.0,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            facecolor=color,
            edgecolor="none",
        )
        ax.add_patch(box)
        ax.text(index, 0.82, week, ha="center", va="center", color="white", fontsize=11, weight="bold")
        ax.text(index, 0.48, topic, ha="center", va="center", color="white", fontsize=12, weight="bold")

    band = FancyBboxPatch(
        (-0.45, -1.25),
        7.9,
        0.72,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=LIGHT,
        edgecolor="#CCD6E2",
        linewidth=1.5,
    )
    ax.add_patch(band)
    ax.text(
        3.5,
        -0.89,
        "매주 반복되는 공통 축  ·  문제 정의  →  데이터 구조·품질 점검  →  검증  →  해석과 한계",
        ha="center",
        va="center",
        fontsize=14,
        color=NAVY,
        weight="bold",
    )
    for index in range(8):
        ax.plot([index, index], [-0.52, 0.12], color="#CCD6E2", linewidth=1.2, linestyle="--")

    save(fig, "curriculum-roadmap.png")


def presentation_before_after() -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("같은 그래프도 질문·근거·한계를 연결하면 설명력이 달라집니다", fontsize=21, weight="bold", color=NAVY, pad=18)

    panels = [(0.5, "Before", "#E76F51"), (8.25, "After", TEAL)]
    for x, label, color in panels:
        ax.add_patch(Rectangle((x, 0.55), 7.25, 7.55, facecolor="white", edgecolor="#D0D5DD", linewidth=2))
        ax.text(x + 0.25, 7.72, label, fontsize=13, weight="bold", color=color)

    ax.text(0.9, 7.1, "분석 결과", fontsize=18, weight="bold", color=NAVY)
    bars = [2.0, 3.3, 2.5, 4.0, 3.0]
    for i, height in enumerate(bars):
        ax.add_patch(Rectangle((1.25 + i * 0.9, 1.55), 0.55, height, facecolor="#B8C4D6", edgecolor="none"))
    ax.plot([1.0, 6.6], [1.55, 1.55], color="#98A2B3", linewidth=1)
    ax.text(1.0, 0.95, "축·단위·질문·해석이 없어 무엇을 봐야 할지 모릅니다.", fontsize=11, color="#B54708")

    ax.text(8.65, 7.05, "주말 오후 평균 대여량이 평일보다 28% 높습니다", fontsize=16, weight="bold", color=NAVY)
    right_heights = [2.1, 3.0, 2.4, 4.3, 3.2]
    labels = ["평일 오전", "평일 오후", "주말 오전", "주말 오후", "전체 평균"]
    for i, (height, label) in enumerate(zip(right_heights, labels)):
        color = ORANGE if label == "주말 오후" else BLUE
        ax.add_patch(Rectangle((8.85 + i * 1.0, 2.4), 0.62, height, facecolor=color, edgecolor="none"))
        ax.text(9.16 + i * 1.0, 2.15, label.replace(" ", "\n"), ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.plot([8.65, 14.25], [2.4, 2.4], color="#667085", linewidth=1)
    ax.text(8.55, 4.85, "평균 대여건수(천 건)", rotation=90, ha="center", va="center", fontsize=10, color=MUTED)
    ax.annotate(
        "+28%",
        xy=(12.15, 6.7),
        xytext=(13.4, 6.3),
        arrowprops={"arrowstyle": "->", "color": ORANGE, "linewidth": 1.8},
        color=ORANGE,
        fontsize=13,
        weight="bold",
    )
    ax.add_patch(FancyBboxPatch((8.7, 0.75), 6.45, 0.9, boxstyle="round,pad=0.05", facecolor=LIGHT, edgecolor="none"))
    ax.text(8.95, 1.32, "관찰  주말 오후 막대가 가장 높습니다.", fontsize=10.5, color=NAVY, weight="bold")
    ax.text(8.95, 0.96, "한계  2월 한 달의 연관성이며 이용 목적을 증명하지 않습니다.", fontsize=10, color=MUTED)

    save(fig, "presentation-before-after.png")


if __name__ == "__main__":
    curriculum_roadmap()
    presentation_before_after()
    print("OT assets generated: 2")
