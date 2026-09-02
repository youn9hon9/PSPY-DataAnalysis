"""8주차의 프로젝트 설계·검산 시각 자료를 PNG로 생성합니다."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ASSET_DIR = Path(__file__).resolve().parent
NAVY = "#183B56"
BLUE = "#3A7BD5"
TEAL = "#2AA198"
ORANGE = "#F59E0B"
RED = "#D64550"
GREEN = "#32936F"
GRAY = "#667085"
LIGHT = "#F4F7FB"

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(ASSET_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def question_funnel() -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("넓은 관심사를 대상·기간·비교값·완료 기준이 있는 질문으로 좁힙니다", fontsize=20, weight="bold", color=NAVY, pad=18)

    levels = [
        (1.1, 8.1, 8.8, 1.25, BLUE, "넓은 관심사", "제주 교통"),
        (1.9, 6.2, 7.2, 1.25, "#5D8FD8", "관찰 질문", "평일과 주말의 시간대별 평균 통행속도는 어떻게 다릅니까?"),
        (2.7, 4.3, 5.6, 1.25, "#6F7FCB", "예측 질문", "도로·시간·요일 정보로 통행속도를 어느 정도 예측할 수 있습니까?"),
        (3.5, 2.4, 4.0, 1.25, TEAL, "의사결정 질문", "우선 점검할 도로·시간대는\n어디입니까?"),
    ]
    for x, y, width, height, color, label, question in levels:
        polygon = Polygon(
            [[x, y + height], [x + width, y + height], [x + width - 0.45, y], [x + 0.45, y]],
            closed=True,
            facecolor=color,
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(polygon)
        ax.text(5, y + 0.83, label, ha="center", va="center", fontsize=12, color="white", weight="bold")
        ax.text(5, y + 0.38, question, ha="center", va="center", fontsize=10.5, color="white", wrap=True)

    ax.add_patch(FancyArrowPatch((5, 2.25), (5, 1.25), arrowstyle="-|>", mutation_scale=22, linewidth=2, color=ORANGE))
    ax.add_patch(FancyBboxPatch((2.55, 0.35), 4.9, 0.85, boxstyle="round,pad=0.05", facecolor="#FFF7E8", edgecolor=ORANGE, linewidth=1.7))
    ax.text(5, 0.78, "완료 기준: 우선 점검할 도로·시간대 목록과 근거 그래프", ha="center", va="center", fontsize=12, color=NAVY, weight="bold")
    save(fig, "question-funnel.png")


def join_before_after() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("코드가 실행됐다는 사실과 조인이 성공했다는 사실은 다릅니다", fontsize=20, weight="bold", color=NAVY, y=0.96)
    fig.subplots_adjust(top=0.78, bottom=0.14, wspace=0.28)
    columns = ["고객ID", "이름", "금액"]
    before = [["A001", "김민준", "NaN"], ["A002", "이서연", "20,000"], ["A003", "박도윤", "NaN"]]
    after = [["A001", "김민준", "10,000"], ["A002", "이서연", "20,000"], ["A003", "박도윤", "15,000"]]
    for ax, data, title, subtitle, good in [
        (axes[0], before, "표준화 전", "3명 중 1명만 연결", False),
        (axes[1], after, "표준화 후", "3명 모두 연결", True),
    ]:
        ax.axis("off")
        ax.text(0.5, 0.88, f"{title}\n{subtitle}", transform=ax.transAxes, ha="center", va="center", fontsize=15, color=GREEN if good else RED, weight="bold")
        table = ax.table(cellText=data, colLabels=columns, cellLoc="center", colLoc="center", loc="center", colWidths=[0.26, 0.26, 0.30])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2.1)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            if row == 0:
                cell.set_facecolor(NAVY)
                cell.get_text().set_color("white")
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(LIGHT)
                if col == 2 and data[row - 1][2] == "NaN":
                    cell.set_facecolor("#FDECEC")
                    cell.get_text().set_color(RED)
                    cell.get_text().set_weight("bold")
                elif col == 2 and good:
                    cell.set_facecolor("#E8F5EF")
                    cell.get_text().set_color(GREEN)
                    cell.get_text().set_weight("bold")
    fig.text(0.5, 0.055, "upper() + 하이픈 제거로 양쪽 키를 같은 규칙으로 표준화합니다", ha="center", fontsize=12, color=GRAY)
    save(fig, "join-before-after.png")


def unit_error() -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("단위를 맞추지 않으면 코드는 성공해도 결과가 10,000배 틀립니다", fontsize=20, weight="bold", color=NAVY, pad=18)

    cards = [
        (0.8, "단위 통일 전", "120 만원 ÷ 1,500,000 원", "0.000080", RED, "잘못된 비율"),
        (8.0, "단위 통일 후", "1,200,000 원 ÷ 1,500,000 원", "0.800000", GREEN, "올바른 비율"),
    ]
    for x, title, equation, value, color, badge in cards:
        ax.add_patch(FancyBboxPatch((x, 1.25), 5.2, 4.4, boxstyle="round,pad=0.08", facecolor="white", edgecolor=color, linewidth=2.5))
        ax.text(x + 2.6, 4.95, title, ha="center", fontsize=15, color=NAVY, weight="bold")
        ax.text(x + 2.6, 4.1, equation, ha="center", fontsize=12, color=GRAY)
        ax.text(x + 2.6, 2.85, value, ha="center", fontsize=29, color=color, weight="bold")
        ax.text(x + 2.6, 1.75, badge, ha="center", fontsize=11, color="white", weight="bold", bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": "none"})
    ax.add_patch(FancyArrowPatch((6.2, 3.45), (7.8, 3.45), arrowstyle="-|>", mutation_scale=24, linewidth=2.5, color=ORANGE))
    ax.text(7.0, 4.05, "× 10,000", ha="center", fontsize=15, color=ORANGE, weight="bold")
    ax.text(7.0, 0.55, "계산 전에 모든 금액을 원 단위로 변환하고, 결과의 자릿수를 상식적으로 검산합니다.", ha="center", fontsize=12, color=NAVY)
    save(fig, "unit-error-10000x.png")


def join_audit() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.axis("off")
    ax.set_title("조인 전후 검산표 — 행 수·키 관계·출처·합계를 함께 확인합니다", fontsize=20, weight="bold", color=NAVY, pad=18)
    columns = ["검산 항목", "왼쪽", "오른쪽", "조인 결과", "판정 질문"]
    rows = [
        ["행 수", "3", "3", "3", "예상한 만큼 유지됐습니까?"],
        ["키 고유값", "3", "3", "3", "중복 키가 있습니까?"],
        ["기대 관계", "1", "1", "1:1", "validate='one_to_one'을 통과합니까?"],
        ["_merge 분포", "—", "—", "both 3", "left_only/right_only가 있습니까?"],
        ["새 결측", "0", "0", "0", "조인 때문에 생긴 결측입니까?"],
        ["금액 합계", "—", "45,000", "45,000", "중복 조인으로 부풀지 않았습니까?"],
    ]
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center", colWidths=[0.17, 0.12, 0.12, 0.15, 0.36])
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(NAVY)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F8FAFC" if row % 2 else "#EEF4FA")
            if col == 0:
                cell.get_text().set_weight("bold")
                cell.get_text().set_color(BLUE)
    fig.text(0.5, 0.055, "행 수가 맞아도 합계가 부풀 수 있고, 결측이 없어도 N:M 조인일 수 있습니다.", ha="center", fontsize=12, color=GRAY)
    save(fig, "join-audit-table.png")


def final_slide_wireframe() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("질문·근거·결론·한계를 한 장에서 연결하는 발표 구조", fontsize=20, weight="bold", color=NAVY, pad=16)
    ax.add_patch(Rectangle((0.55, 0.45), 14.9, 7.75, facecolor="white", edgecolor="#98A2B3", linewidth=2))
    ax.add_patch(Rectangle((0.85, 6.95), 13.95, 0.78, facecolor=NAVY, edgecolor="none"))
    ax.text(1.15, 7.34, "질문  주말 오후에 혼잡이 반복되는 도로는 어디입니까?", va="center", fontsize=16, color="white", weight="bold")

    ax.add_patch(FancyBboxPatch((0.95, 2.25), 9.3, 4.35, boxstyle="round,pad=0.04", facecolor="#F8FAFC", edgecolor="#D0D5DD"))
    ax.text(1.25, 6.2, "핵심 근거 그래프", fontsize=12, color=BLUE, weight="bold")
    heights = [2.0, 2.7, 2.3, 3.7, 2.9, 4.0]
    for i, height in enumerate(heights):
        color = ORANGE if i in (3, 5) else BLUE
        ax.add_patch(Rectangle((1.5 + i * 1.25, 2.75), 0.75, height * 0.72, facecolor=color, edgecolor="none"))
    ax.plot([1.25, 9.75], [2.75, 2.75], color="#667085", linewidth=1)
    ax.text(1.2, 2.43, "도로·시간대", fontsize=10, color=GRAY)
    ax.annotate("반복 혼잡", xy=(8.0, 5.65), xytext=(8.9, 6.0), arrowprops={"arrowstyle": "->", "color": ORANGE}, color=ORANGE, weight="bold")

    ax.add_patch(FancyBboxPatch((10.55, 4.55), 4.05, 2.05, boxstyle="round,pad=0.04", facecolor="#FFF7E8", edgecolor=ORANGE, linewidth=1.8))
    ax.text(10.9, 6.15, "기준선·핵심 지표", fontsize=12, color=ORANGE, weight="bold")
    ax.text(10.9, 5.65, "평일 평균 대비  +24%\n반복 횟수  6/8주\n표본 수  n=12,480", fontsize=11.5, color=NAVY, linespacing=1.45, va="top")

    ax.add_patch(FancyBboxPatch((10.55, 2.25), 4.05, 1.95, boxstyle="round,pad=0.04", facecolor=LIGHT, edgecolor="#98A2B3"))
    ax.text(10.9, 3.75, "해석 조건", fontsize=12, color=GRAY, weight="bold")
    ax.text(10.9, 3.2, "샘플 데이터 기반\n사고·공사 정보 미포함", fontsize=11, color=NAVY, linespacing=1.45)

    ax.add_patch(FancyBboxPatch((0.95, 0.78), 13.65, 1.05, boxstyle="round,pad=0.04", facecolor="#E8F5EF", edgecolor=TEAL, linewidth=1.5))
    ax.text(1.25, 1.45, "결론", fontsize=12, color=TEAL, weight="bold")
    ax.text(2.1, 1.45, "A도로의 주말 16~18시를 우선 점검합니다.", fontsize=13, color=NAVY, weight="bold")
    ax.text(1.25, 1.05, "한계", fontsize=11, color=GRAY, weight="bold")
    ax.text(2.1, 1.05, "관측 연관성이며 원인을 증명하지 않습니다. 전체 원본으로 재검증합니다.", fontsize=11, color=GRAY)
    save(fig, "final-slide-wireframe.png")


if __name__ == "__main__":
    question_funnel()
    join_before_after()
    unit_error()
    join_audit()
    final_slide_wireframe()
    print("Week 8 assets generated: 5")
