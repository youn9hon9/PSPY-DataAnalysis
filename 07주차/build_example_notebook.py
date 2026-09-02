"""7주차 실제 CCTV 분석 예제 노트북을 생성하고 코드 셀을 순서대로 검증합니다."""

from __future__ import annotations

import contextlib
import io
import json
import textwrap
import traceback
from pathlib import Path


WEEK_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = WEEK_DIR / "example.ipynb"


def lines(text: str) -> list[str]:
    normalized = textwrap.dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(text),
    }


cells = [
    markdown(
        '''
        # 7주차 예제 — 실제 CCTV·S-DoT 데이터로 공간 커버리지 분석

        이 노트북은 서울시 **전용차로 위반 단속 CCTV 원본 60행에서 정리한 고유 위치 56곳**을 기존 시설로, 파일명 기준 2026년 7월 27일~8월 16일인 S-DoT 주간 CSV 3개의 측정 지점을 수요로 사용합니다. 실제 측정 시작·종료 시각은 Part 2에서 확인합니다. 좌표 품질 점검, 하버사인 거리, 현재 커버리지, 그리디 후보지 추천, 민감도 분석을 처음부터 끝까지 실행합니다.

        - [서울시 불법주정차·전용차로 단속 CCTV 위치정보](https://data.seoul.go.kr/dataList/OA-20471/S/1/datasetView.do)
        - [스마트서울 도시데이터 센서(S-DoT) 유동인구 측정 정보](https://data.seoul.go.kr/dataList/OA-15964/S/1/datasetView.do)
        - [로컬 데이터 폴더 안내](<../dataset/extracted/서울시 CCTV 공공데이터/README.md>)
        - [교재 설명과 결과 해석](notion.md)

        > **분석 범위:** 전용차로 단속 CCTV는 범죄예방 CCTV가 아닙니다. S-DoT 지점도 실제 CCTV 설치 가능 후보지가 아닙니다. 이 결과는 공개 좌표로 공간 커버리지와 그리디 MCLP 절차를 익히는 교육용 시나리오이며, 안전 효과나 실제 설치 우선순위를 뜻하지 않습니다.
        '''
    ),
    markdown(
        '''
        ## Part 0 — 실행 경로와 입력 파일 확인

        노트북을 프로젝트 루트에서 열어도, `07주차` 폴더에서 열어도 작동하도록 `README.md`가 있는 상위 폴더를 찾습니다. 입력 파일이 없을 때는 임의의 대체 데이터를 만들지 않고 어떤 파일이 필요한지 오류로 알려 줍니다.
        '''
    ),
    code(
        r'''
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse
        import numpy as np
        import pandas as pd


        def find_project_root(start: Path) -> Path:
            for candidate in (start, *start.parents):
                if (candidate / "README.md").is_file() and (candidate / "07주차").is_dir():
                    return candidate
            raise FileNotFoundError("README.md가 있는 프로젝트 루트를 찾을 수 없습니다.")


        PROJECT_ROOT = find_project_root(Path.cwd().resolve())
        DATA_ROOT = PROJECT_ROOT / "dataset" / "extracted" / "서울시 CCTV 공공데이터"
        ASSET_DIR = PROJECT_ROOT / "07주차" / "assets"
        OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "서울시 CCTV 공공데이터"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        PARKING_PATH = DATA_ROOT / "02_교통_단속_CCTV" / "02_불법주정차" / "서울시 불법주정차 단속 CCTV 위치정보.csv"
        CCTV_PATH = DATA_ROOT / "02_교통_단속_CCTV" / "03_전용차로" / "서울시 전용차로 위반 단속 CCTV 위치정보.csv"
        SENSOR_PATH = DATA_ROOT / "03_S-DoT_유동인구" / "01_설치위치" / "서울시 도시데이터 센서(S-DoT) 유동인구 설치 위치정보_251113.xlsx"
        WALK_DIR = DATA_ROOT / "03_S-DoT_유동인구" / "02_측정데이터" / "2026"
        WALK_FILENAMES = (
            "S-DoT_WALK_2026.07.27-08.02.csv",
            "S-DoT_WALK_2026.08.03-08.09.csv",
            "S-DoT_WALK_2026.08.10-08.16.csv",
        )
        WALK_PATHS = [WALK_DIR / filename for filename in WALK_FILENAMES]

        required = [PARKING_PATH, CCTV_PATH, SENSOR_PATH, *WALK_PATHS]
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("필요한 입력 파일이 없습니다:\n" + "\n".join(map(str, missing)))
        if len(WALK_PATHS) != 3:
            raise ValueError(f"예제는 S-DoT 주간 파일 3개를 기대하지만 {len(WALK_PATHS)}개를 찾았습니다.")

        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        for font_name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"):
            if font_name in available_fonts:
                plt.rcParams["font.family"] = font_name
                break
        plt.rcParams["axes.unicode_minus"] = False

        print("프로젝트 루트 확인:", PROJECT_ROOT.name)
        print("입력 파일 확인:", len(required), "개")
        print("S-DoT 주간 파일:", len(WALK_PATHS), "개")
        print("산출물 폴더:", OUTPUT_DIR.relative_to(PROJECT_ROOT))
        '''
    ),
    markdown(
        '''
        ## Part 1 — 분석 대상을 정의합니다

        | 역할 | 이 노트북의 정의 | 선택 이유 |
        |---|---|---|
        | 기존 시설 | 전용차로 위반 단속 CCTV 60행을 정리한 56개 고유 위치 | 좌표가 모두 서울 범위에 있고, 작은 시설 집합이라 입지추천 변화가 드러납니다. |
        | 수요 지점 | 센서별 최대 관측 수의 80% 이상이 있는 S-DoT 센서 | 관측이 충분한 지점만 남겨 적은 표본으로 계산한 평균이 가중치를 좌우하지 않게 합니다. |
        | 수요 가중치 | 센서별 관측 1회당 평균 방문자 수 | 지점 수만 세는 결과와 유동량을 반영한 결과를 비교할 수 있습니다. |
        | 후보지 | 서울특별시 주소가 있는 S-DoT 설치 위치 126곳 | 공개 좌표를 가진 계산 후보입니다. 서울대공원 8곳은 서울 행정구역 밖이라 제외하며, 실제 설치 가능 여부는 검증하지 않았습니다. |
        | 기준 시나리오 | 반경 `R=500m`, 신규 시설 `p=5` | 알고리즘을 비교하기 위한 가정이며 카메라의 실제 촬영거리나 정책 기준이 아닙니다. |

        불법주정차 CCTV도 함께 읽어 좌표 품질과 시설 밀도를 비교합니다. 더 많은 시설을 무조건 사용하는 대신, 질문과 실습 목적에 맞는 자료를 선택하는 과정도 분석의 일부입니다.
        '''
    ),
    code(
        r'''
        parking_raw = pd.read_csv(PARKING_PATH, encoding="cp949")
        cctv_raw = pd.read_csv(CCTV_PATH, encoding="cp949")
        sensors_raw = pd.read_excel(SENSOR_PATH)

        walk_frames = []
        for path in WALK_PATHS:
            frame = pd.read_csv(path, encoding="cp949")
            frame["원본파일"] = path.name
            walk_frames.append(frame)
        walk_raw = pd.concat(walk_frames, ignore_index=True)

        print(f"불법주정차 CCTV: {len(parking_raw):,}행")
        print(f"전용차로 단속 CCTV: {len(cctv_raw):,}행")
        print(f"S-DoT 설치 위치: {len(sensors_raw):,}행")
        print(f"S-DoT 측정 원본: {len(walk_raw):,}행")
        print("전용차로 CCTV 유형:", cctv_raw["현장구분"].value_counts().to_dict())
        '''
    ),
    markdown(
        '''
        ## Part 2 — 좌표·중복·관측량을 점검합니다

        서울 범위를 넉넉하게 `위도 37.40~37.75`, `경도 126.75~127.20`으로 두고 좌표를 검사합니다. 범위 밖 좌표는 자동으로 고치지 않습니다. 원본 주소와 공식 자료를 다시 확인해야 하므로 분석에서는 제외하고 개수를 기록합니다.

        S-DoT는 같은 센서·측정시간 조합이 중복된 경우 등록일이 가장 늦은 행을 남깁니다. 이후 각 센서의 관측 수를 가장 많이 관측된 센서와 비교하고, 관측 완전성이 80% 이상인 지점만 수요로 사용합니다.
        '''
    ),
    code(
        r'''
        SEOUL_LAT = (37.40, 37.75)
        SEOUL_LON = (126.75, 127.20)
        MIN_COMPLETENESS = 0.80


        def clean_coordinate_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
            work = frame.copy()
            work["위도"] = pd.to_numeric(work["위도"], errors="coerce")
            work["경도"] = pd.to_numeric(work["경도"], errors="coerce")
            missing_coordinate = work[["위도", "경도"]].isna().any(axis=1)
            out_of_range = ~(
                work["위도"].between(*SEOUL_LAT)
                & work["경도"].between(*SEOUL_LON)
            )
            valid = ~(missing_coordinate | out_of_range)
            duplicate_coordinates = work.loc[valid].duplicated(["위도", "경도"])
            cleaned = work.loc[valid].drop_duplicates(["위도", "경도"]).reset_index(drop=True)
            report = {
                "원본행": len(work),
                "좌표결측": int(missing_coordinate.sum()),
                "범위밖좌표": int((out_of_range & ~missing_coordinate).sum()),
                "중복좌표": int(duplicate_coordinates.sum()),
                "분석행": len(cleaned),
            }
            return cleaned, report


        parking, parking_report = clean_coordinate_rows(parking_raw)
        cctv, cctv_report = clean_coordinate_rows(cctv_raw)
        sensor_locations, sensor_report = clean_coordinate_rows(sensors_raw)

        parking_valid_mask = (
            pd.to_numeric(parking_raw["위도"], errors="coerce").between(*SEOUL_LAT)
            & pd.to_numeric(parking_raw["경도"], errors="coerce").between(*SEOUL_LON)
        )
        parking_invalid_preview = parking_raw.loc[
            ~parking_valid_mask,
            ["고정형CCTV지번주소", "위도", "경도", "자치구", "단속지점명"],
        ]

        walk = walk_raw.copy()
        walk["측정시간"] = pd.to_datetime(
            walk["측정시간"], format="%Y-%m-%d_%H:%M:%S", errors="coerce"
        )
        walk["등록일시"] = pd.to_datetime(walk["등록일"], errors="coerce")
        invalid_time = int(walk["측정시간"].isna().sum())
        negative_visitors = int((walk["방문자수"] < 0).sum())
        duplicate_sensor_time = int(walk.duplicated(["시리얼", "측정시간"]).sum())
        walk = (
            walk.sort_values(["시리얼", "측정시간", "등록일시"])
            .drop_duplicates(["시리얼", "측정시간"], keep="last")
            .reset_index(drop=True)
        )

        sensor_summary = (
            walk.groupby("시리얼", as_index=False)
            .agg(
                평균방문자수=("방문자수", "mean"),
                중앙방문자수=("방문자수", "median"),
                관측수=("방문자수", "size"),
            )
        )
        measured_sensor_count = len(sensor_summary)
        max_observations = int(sensor_summary["관측수"].max())
        minimum_observations = int(np.ceil(max_observations * MIN_COMPLETENESS))
        sensor_summary["관측완전성"] = sensor_summary["관측수"] / max_observations
        sensor_summary_all = sensor_summary.copy()
        completeness_table = pd.DataFrame(
            {
                "완전성 기준": ["50% 이상", "80% 이상", "90% 이상"],
                "센서 수": [
                    int((sensor_summary_all["관측완전성"] >= threshold).sum())
                    for threshold in (0.50, 0.80, 0.90)
                ],
            }
        )
        sensor_summary = sensor_summary.loc[
            sensor_summary["관측완전성"] >= MIN_COMPLETENESS
        ].copy()
        demand = sensor_summary.merge(
            sensor_locations[["방문자 센서코드", "주소", "위도", "경도"]],
            left_on="시리얼",
            right_on="방문자 센서코드",
            how="inner",
            validate="one_to_one",
        )
        if len(demand) != len(sensor_summary):
            raise ValueError("품질 기준을 통과한 센서 중 설치 위치와 연결되지 않은 센서가 있습니다.")
        demand = demand.loc[demand["주소"].str.startswith("서울특별시")].reset_index(drop=True)
        candidates = sensor_locations.loc[
            sensor_locations["주소"].str.startswith("서울특별시")
        ].reset_index(drop=True)

        quality_table = pd.DataFrame(
            [
                {"자료": "불법주정차 CCTV", **parking_report},
                {"자료": "전용차로 단속 CCTV", **cctv_report},
                {"자료": "S-DoT 설치 위치", **sensor_report},
            ]
        )
        print(quality_table.to_string(index=False))
        print("\n[서울 범위 밖으로 제외한 불법주정차 CCTV 좌표]")
        print(parking_invalid_preview.to_string(index=False))
        print()
        print(f"S-DoT 측정시간: {walk['측정시간'].min()} ~ {walk['측정시간'].max()}")
        print(f"잘못된 측정시간: {invalid_time:,}행 / 음수 방문자수: {negative_visitors:,}행")
        print(f"중복 센서·시간: {duplicate_sensor_time:,}행 → 제거 후 {len(walk):,}행")
        print(
            f"측정 센서: {measured_sensor_count}곳 → "
            f"최대 {max_observations:,}건의 80%({minimum_observations:,}건) 이상 수요 지점: {len(demand)}곳"
        )
        print("\n[관측 완전성 기준별 센서 수]")
        print(completeness_table.to_string(index=False))
        print(f"후보지: {len(candidates)}곳 / 수요 좌표 결측: {int(demand[['위도', '경도']].isna().any(axis=1).sum())}곳")

        completeness_sorted = np.sort(sensor_summary_all["관측완전성"].to_numpy())
        fig, ax = plt.subplots(figsize=(10, 5.5))
        colors = np.where(completeness_sorted >= MIN_COMPLETENESS, "#2AA198", "#D0D5DD")
        ax.bar(np.arange(1, len(completeness_sorted) + 1), completeness_sorted * 100, color=colors, width=0.9)
        ax.axhline(80, color="#D64550", linestyle="--", linewidth=2, label="기준 80%")
        ax.set_title("S-DoT 센서별 관측 완전성", fontsize=16, weight="bold", color="#183B56")
        ax.set_xlabel("관측 완전성이 낮은 순서의 센서")
        ax.set_ylabel("최대 관측 수 대비 비율(%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        completeness_asset = ASSET_DIR / "sdot-observation-completeness.png"
        fig.savefig(completeness_asset, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        '''
    ),
    markdown(
        '''
        좌표 컬럼에 결측치가 없더라도 범위를 확인해야 합니다. 불법주정차 자료에서는 범위 밖 좌표가 발견되지만, 선택한 전용차로 자료는 원본 60행 모두 서울 범위 안에 있으며 중복 좌표 4행을 제거해 56개 위치로 정리됩니다. S-DoT 원본 265,146행 중 같은 센서·시간 조합 7행을 정리하고, 최대 관측 수의 80% 이상인 62곳을 수요 지점으로 사용합니다.

        ![S-DoT 센서별 관측 완전성과 80% 품질 기준](assets/sdot-observation-completeness.png)

        *캡션: 전체 측정 센서 112곳을 관측 완전성이 낮은 순서로 놓고, 기준을 통과한 62곳을 청록색으로 표시했습니다.*

        > **그림 읽기:** 80% 선 아래의 회색 센서까지 포함하면 수요 지점 수는 늘지만, 적은 관측으로 계산한 평균의 불확실성도 함께 커집니다.

        > **확인 질문:** 관측이 16번뿐인 센서와 약 3,000번인 센서의 평균을 같은 신뢰도로 비교해도 되는지 생각해 봅니다. 80% 기준을 50%나 90%로 바꾸면 결과가 어떻게 달라질지도 확인해 봅니다.
        '''
    ),
    markdown(
        '''
        ## Part 3 — 하버사인 거리와 현재 커버리지를 계산합니다

        먼저 서울시청과 광화문의 거리로 함수 단위를 검산합니다. 그다음 수요 62곳×정제된 기존 시설 56곳, 수요 62곳×후보지 126곳의 거리 행렬을 만듭니다.
        '''
    ),
    code(
        r'''
        EARTH_RADIUS_M = 6_371_000


        def haversine_matrix(origin_latlon, destination_latlon):
            """두 좌표 배열의 모든 조합에 대한 하버사인 거리(m)를 반환합니다."""
            origin = np.radians(np.asarray(origin_latlon, dtype=float))
            destination = np.radians(np.asarray(destination_latlon, dtype=float))
            lat1 = origin[:, 0][:, None]
            lon1 = origin[:, 1][:, None]
            lat2 = destination[:, 0][None, :]
            lon2 = destination[:, 1][None, :]
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            a = np.clip(a, 0, 1)
            return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


        unit_check = haversine_matrix(
            [[37.5665, 126.9780]],
            [[37.5716, 126.9769]],
        )[0, 0]
        print(f"서울시청↔광화문: {unit_check:.1f}m")

        demand_coordinates = demand[["위도", "경도"]].to_numpy()
        existing_coordinates = cctv[["위도", "경도"]].to_numpy()
        candidate_coordinates = candidates[["위도", "경도"]].to_numpy()

        distance_existing = haversine_matrix(demand_coordinates, existing_coordinates)
        distance_candidates = haversine_matrix(demand_coordinates, candidate_coordinates)
        print("수요×기존 시설 거리 행렬:", distance_existing.shape)
        print("수요×후보지 거리 행렬:", distance_candidates.shape)
        '''
    ),
    code(
        r'''
        RADIUS_M = 500
        weights = demand["평균방문자수"].to_numpy()


        def coverage_row(label, distances, radius, weights):
            nearest = distances.min(axis=1)
            covered = nearest <= radius
            return {
                "시설 자료": label,
                "시설 수": distances.shape[1],
                "커버 지점": int(covered.sum()),
                "지점 커버율(%)": covered.mean() * 100,
                "가중 커버율(%)": weights[covered].sum() / weights.sum() * 100,
            }


        distance_parking = haversine_matrix(
            demand_coordinates,
            parking[["위도", "경도"]].to_numpy(),
        )
        facility_comparison = pd.DataFrame(
            [
                coverage_row("불법주정차 CCTV", distance_parking, RADIUS_M, weights),
                coverage_row("전용차로 단속 CCTV", distance_existing, RADIUS_M, weights),
            ]
        )
        print(facility_comparison.round(1).to_string(index=False))
        '''
    ),
    markdown(
        '''
        불법주정차 CCTV는 4,569개 고유 좌표로 매우 촘촘해 500m 시나리오에서 추천할 미커버 수요가 거의 남지 않습니다. 이 장에서는 알고리즘의 선택 과정과 개선 폭을 확인할 수 있도록 전용차로 단속 CCTV 56개 고유 위치를 기존 시설로 사용합니다. 이는 분석 난이도를 위한 선택이며 두 CCTV의 정책 목적이나 성능을 비교한 결과가 아닙니다.
        '''
    ),
    markdown(
        '''
        ## Part 4·5 — 후보지 커버 집합과 그리디 추천을 만듭니다

        각 후보지가 500m 안에서 아직 커버되지 않은 수요 가중치를 얼마나 추가하는지 계산합니다. 동률이면 새로 커버하는 지점 수가 많은 후보, 그마저 같으면 센서코드가 작은 후보를 선택해 실행 순서에 따라 결과가 달라지지 않게 합니다.
        '''
    ),
    code(
        r'''
        def greedy_max_coverage(
            initial_covered,
            candidate_cover,
            weights,
            candidate_frame,
            p,
        ):
            covered = initial_covered.copy()
            selected = []
            rows = []

            for step in range(1, p + 1):
                uncovered = ~covered
                new_cover = uncovered[:, None] & candidate_cover
                gains = (new_cover * weights[:, None]).sum(axis=0)
                new_counts = new_cover.sum(axis=0)
                available = [idx for idx in range(candidate_cover.shape[1]) if idx not in selected]
                best = max(
                    available,
                    key=lambda idx: (
                        round(float(gains[idx]), 12),
                        int(new_counts[idx]),
                        -int(candidate_frame.iloc[idx]["방문자 센서코드"]),
                    ),
                )
                if gains[best] <= 0:
                    break

                selected.append(best)
                covered |= candidate_cover[:, best]
                site = candidate_frame.iloc[best]
                rows.append(
                    {
                        "단계": step,
                        "후보_인덱스": best,
                        "센서코드": int(site["방문자 센서코드"]),
                        "주소": site["주소"],
                        "위도": float(site["위도"]),
                        "경도": float(site["경도"]),
                        "신규커버지점": int(new_counts[best]),
                        "추가평균방문자수합": float(gains[best]),
                        "누적지점커버율": float(covered.mean()),
                        "누적가중커버율": float(weights[covered].sum() / weights.sum()),
                    }
                )

            return selected, covered, pd.DataFrame(rows)


        def metrics_row(label, covered, nearest, weights):
            return {
                "구분": label,
                "커버 지점": int(covered.sum()),
                "지점 커버율(%)": covered.mean() * 100,
                "가중 커버율(%)": weights[covered].sum() / weights.sum() * 100,
                "최근접 중앙값(m)": np.median(nearest),
                "최근접 상위90%(m)": np.quantile(nearest, 0.90),
            }


        initial_covered = (distance_existing <= RADIUS_M).any(axis=1)
        candidate_cover = distance_candidates <= RADIUS_M
        selected, covered_after, recommendations = greedy_max_coverage(
            initial_covered,
            candidate_cover,
            weights,
            candidates,
            p=5,
        )

        nearest_before = distance_existing.min(axis=1)
        nearest_after = np.minimum(
            nearest_before,
            distance_candidates[:, selected].min(axis=1),
        )
        result_metrics = pd.DataFrame(
            [
                metrics_row("신규 설치 전", initial_covered, nearest_before, weights),
                metrics_row("후보 5곳 추가 후", covered_after, nearest_after, weights),
            ]
        )

        print("[단계별 추천]")
        print(
            recommendations[
                ["단계", "센서코드", "주소", "신규커버지점", "추가평균방문자수합", "누적가중커버율"]
            ].round({"추가평균방문자수합": 1, "누적가중커버율": 3}).to_string(index=False)
        )
        print("\n[설치 전후 지표]")
        print(result_metrics.round(1).to_string(index=False))
        '''
    ),
    markdown(
        '''
        추천 5곳은 “유동인구가 많은 순서”가 아니라 **기존 시설과 앞서 선택한 후보가 아직 커버하지 못한 수요를 추가하는 순서**입니다. 결과를 실제 설치안으로 사용하려면 전용차로 존재 여부, 시야, 전력·통신, 토지 소유, 민원과 사생활 영향을 별도로 검토해야 합니다.
        '''
    ),
    markdown(
        '''
        ## Part 6 — 반경과 설치 수를 바꿔 민감도를 확인합니다

        반경을 300m, 500m, 700m로 바꾸고 신규 후보를 0개, 3개, 5개, 10개 선택합니다. 지점 수가 아니라 평균 방문자 수 가중 커버율을 비교합니다. 또한 동일 가중치를 썼을 때 추천 후보가 달라지는지도 확인합니다.
        '''
    ),
    code(
        r'''
        def run_scenario(radius, p, scenario_weights):
            base = (distance_existing <= radius).any(axis=1)
            if p == 0:
                return [], base, pd.DataFrame()
            return greedy_max_coverage(
                base,
                distance_candidates <= radius,
                scenario_weights,
                candidates,
                p,
            )


        sensitivity_rows = []
        for radius in (300, 500, 700):
            for p in (0, 3, 5, 10):
                chosen, covered, _ = run_scenario(radius, p, weights)
                sensitivity_rows.append(
                    {
                        "반경(m)": radius,
                        "요청 설치 수": p,
                        "실제 선택 수": len(chosen),
                        "지점 커버율(%)": covered.mean() * 100,
                        "가중 커버율(%)": weights[covered].sum() / weights.sum() * 100,
                    }
                )
        sensitivity = pd.DataFrame(sensitivity_rows)
        sensitivity_pivot = sensitivity.pivot(
            index="반경(m)", columns="요청 설치 수", values="가중 커버율(%)"
        )
        print("[반경·설치 수별 가중 커버율(%)]")
        print(sensitivity_pivot.round(1).to_string())

        equal_weights = np.ones(len(demand))
        equal_selected, equal_covered, _ = run_scenario(500, 5, equal_weights)
        weight_comparison = pd.DataFrame(
            {
                "가중치": ["평균 방문자 수", "모든 지점 동일"],
                "선택 센서코드": [
                    ", ".join(map(str, recommendations["센서코드"].tolist())),
                    ", ".join(str(int(candidates.iloc[idx]["방문자 센서코드"])) for idx in equal_selected),
                ],
                "실제 방문자 가중 커버율(%)": [
                    weights[covered_after].sum() / weights.sum() * 100,
                    weights[equal_covered].sum() / weights.sum() * 100,
                ],
            }
        )
        print("\n[가중치 선택 비교]")
        print(weight_comparison.round(1).to_string(index=False))
        '''
    ),
    code(
        r'''
        NAVY = "#183B56"
        BLUE = "#3A7BD5"
        ORANGE = "#F59E0B"
        RED = "#D64550"
        GREEN = "#2AA198"
        GRAY = "#667085"

        fig, ax = plt.subplots(figsize=(10, 6))
        for radius, group in sensitivity.groupby("반경(m)"):
            ax.plot(
                group["요청 설치 수"],
                group["가중 커버율(%)"],
                marker="o",
                linewidth=2.5,
                label=f"반경 {radius}m",
            )
        ax.set_title("반경과 설치 수에 따른 S-DoT 가중 커버율", fontsize=16, weight="bold", color=NAVY)
        ax.set_xlabel("신규 후보지 수 p")
        ax.set_ylabel("평균 방문자 수 가중 커버율(%)")
        ax.set_xticks([0, 3, 5, 10])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        sensitivity_asset = ASSET_DIR / "cctv-sensitivity.png"
        fig.savefig(sensitivity_asset, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        max_weight = weights.max()
        point_sizes = 24 + 120 * np.sqrt(weights / max_weight)
        mean_latitude = demand["위도"].mean()


        def draw_coverage_panel(ax, covered, title, selected_indices=None):
            uncovered = ~covered
            ax.scatter(
                demand.loc[covered, "경도"], demand.loc[covered, "위도"],
                s=point_sizes[covered], color=GREEN, alpha=0.70,
                edgecolor="white", linewidth=0.6, label="커버 수요",
            )
            ax.scatter(
                demand.loc[uncovered, "경도"], demand.loc[uncovered, "위도"],
                s=point_sizes[uncovered], color=RED, alpha=0.78,
                edgecolor="white", linewidth=0.6, label="미커버 수요",
            )
            ax.scatter(
                cctv["경도"], cctv["위도"], s=32, color=BLUE,
                marker="x", linewidth=1.4, label="기존 전용차로 CCTV",
            )
            if selected_indices:
                selected_sites = candidates.iloc[selected_indices]
                ax.scatter(
                    selected_sites["경도"], selected_sites["위도"],
                    s=180, color=ORANGE, marker="*", edgecolor=NAVY,
                    linewidth=0.8, label="그리디 추천 후보",
                )
                lat_radius = RADIUS_M / 110_574
                for _, site in selected_sites.iterrows():
                    lon_radius = RADIUS_M / (111_320 * np.cos(np.radians(site["위도"])))
                    ax.add_patch(
                        Ellipse(
                            (site["경도"], site["위도"]),
                            width=2 * lon_radius,
                            height=2 * lat_radius,
                            facecolor=ORANGE,
                            edgecolor=ORANGE,
                            alpha=0.10,
                            linewidth=1.2,
                        )
                    )
                    ax.annotate(
                        str(int(site["방문자 센서코드"])),
                        (site["경도"], site["위도"]),
                        xytext=(4, 5), textcoords="offset points",
                        fontsize=8, color=NAVY, weight="bold",
                    )
            ax.set_title(title, fontsize=14, weight="bold", color=NAVY)
            ax.set_xlabel("경도")
            ax.set_ylabel("위도")
            ax.grid(alpha=0.18)
            ax.set_aspect(1 / np.cos(np.radians(mean_latitude)))


        fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
        draw_coverage_panel(
            axes[0], initial_covered,
            f"추가 전: {initial_covered.sum()}/{len(demand)}곳 커버",
        )
        draw_coverage_panel(
            axes[1], covered_after,
            f"후보 5곳 추가 후: {covered_after.sum()}/{len(demand)}곳 커버",
            selected,
        )
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markersize=9, label="커버 수요"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=RED, markersize=9, label="미커버 수요"),
            Line2D([0], [0], marker="x", color=BLUE, markersize=9, label="기존 전용차로 CCTV"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=ORANGE, markeredgecolor=NAVY, markersize=13, label="그리디 추천 후보"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(
            "500m 공간 커버리지 시나리오 — 실제 지도 축척이 아닌 위경도 산점도",
            fontsize=17, weight="bold", color=NAVY,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
        coverage_asset = ASSET_DIR / "cctv-coverage-before-after.png"
        fig.savefig(coverage_asset, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print("생성한 시각자료:", sensitivity_asset.name)
        print("생성한 시각자료:", coverage_asset.name)
        '''
    ),
    markdown(
        '''
        ![반경과 신규 후보지 수에 따른 가중 커버율 변화](assets/cctv-sensitivity.png)

        *캡션: 같은 후보지 수라도 반경 가정에 따라 가중 커버율이 달라지며, 후보를 더 추가할수록 증가 폭이 작아지는 구간이 나타납니다.*

        > **그림 읽기:** `p=0`에서 반경별 시작점이 다르고, `p=5` 이후 각 선의 기울기가 어떻게 바뀌는지 살펴봅니다.

        ![전용차로 CCTV 공간 커버리지와 그리디 후보 5곳 추가 전후](assets/cctv-coverage-before-after.png)

        *캡션: 점의 크기는 센서별 평균 방문자 수, 색은 500m 반경 커버 여부입니다. 별표와 반투명 원은 선택된 계산 후보와 500m 분석 반경을 나타냅니다.*

        > **그림 읽기:** 추천 후보가 기존 CCTV와 멀면서 크기가 큰 빨간 수요 지점을 우선 포함하는지 확인합니다. 배경 도로·행정경계가 없는 좌표 산점도이므로 실제 설치 가능성은 판단할 수 없습니다.
        '''
    ),
    markdown(
        '''
        ## Part 7 — 결과 파일을 저장하고 선택적으로 대화형 지도를 만듭니다

        정제한 수요 지점, 추천 후보, 민감도 표를 `dataset/processed/서울시 CCTV 공공데이터/`에 저장합니다. 원본과 분석 산출물을 분리하면 같은 분석을 다시 실행하기 쉽습니다.
        '''
    ),
    code(
        r'''
        demand_result = demand[
            ["시리얼", "주소", "위도", "경도", "평균방문자수", "중앙방문자수", "관측수", "관측완전성"]
        ].copy()
        demand_result["기존최근접거리_m"] = nearest_before
        demand_result["기존500m커버"] = initial_covered
        demand_result["추천후최근접거리_m"] = nearest_after
        demand_result["추천후500m커버"] = covered_after

        demand_output = OUTPUT_DIR / "week7_sdot_demand_coverage.csv"
        recommendation_output = OUTPUT_DIR / "week7_greedy_recommendations.csv"
        sensitivity_output = OUTPUT_DIR / "week7_sensitivity.csv"
        demand_result.to_csv(demand_output, index=False, encoding="utf-8-sig")
        recommendations.drop(columns="후보_인덱스").to_csv(
            recommendation_output, index=False, encoding="utf-8-sig"
        )
        sensitivity.to_csv(sensitivity_output, index=False, encoding="utf-8-sig")

        print("저장 완료:")
        for output_path in (demand_output, recommendation_output, sensitivity_output):
            print("-", output_path.relative_to(PROJECT_ROOT))
        '''
    ),
    code(
        r'''
        try:
            import folium
        except ModuleNotFoundError:
            print("선택 실습: `pip install folium` 후 이 셀을 다시 실행하면 HTML 지도를 저장합니다.")
        else:
            center = [float(demand["위도"].mean()), float(demand["경도"].mean())]
            map_object = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
            for _, row in cctv.iterrows():
                folium.CircleMarker(
                    [row["위도"], row["경도"]], radius=2,
                    color="#3A7BD5", fill=True, tooltip="기존 전용차로 CCTV",
                ).add_to(map_object)
            for idx, row in demand.iterrows():
                color = "#2AA198" if covered_after[idx] else "#D64550"
                folium.CircleMarker(
                    [row["위도"], row["경도"]], radius=4,
                    color=color, fill=True,
                    tooltip=f"S-DoT {int(row['시리얼'])} / 평균 {row['평균방문자수']:.1f}",
                ).add_to(map_object)
            for _, row in recommendations.iterrows():
                folium.Marker(
                    [row["위도"], row["경도"]],
                    tooltip=f"추천 {int(row['단계'])}: 센서 {int(row['센서코드'])}",
                    icon=folium.Icon(color="orange", icon="star"),
                ).add_to(map_object)
                folium.Circle(
                    [row["위도"], row["경도"]], radius=RADIUS_M,
                    color="#F59E0B", fill=True, fill_opacity=0.08,
                ).add_to(map_object)
            html_output = ASSET_DIR / "cctv-coverage-map.html"
            map_object.save(html_output)
            print("대화형 지도 저장:", html_output.relative_to(PROJECT_ROOT))
        '''
    ),
    markdown(
        '''
        ## 분석 결론과 한계

        - 이 노트북의 수치는 **전용차로 단속 CCTV 56개 고유 위치, 관측 완전성 80%를 통과한 S-DoT 수요 62곳, 서울 주소의 계산 후보 126곳, 500m 반경, 후보 5곳**이라는 조건에서만 성립합니다.
        - S-DoT 공식 안내에 따르면 방문자 수는 Wi-Fi 단말 신호 집계 또는 CCTV 피플카운팅 방식의 센서 관측값이며 정확한 보행자 총인구가 아닙니다. 평균값은 수요 가중치의 한 가지 대리변수입니다.
        - 500m는 카메라 촬영범위가 아니라 알고리즘 비교를 위한 공간 반경입니다.
        - 후보지는 공개된 S-DoT 설치 좌표일 뿐, 전용차로·전력·통신·시야·토지·사생활 조건을 검증한 CCTV 설치 후보가 아닙니다.
        - 그리디 선택은 설명 가능한 휴리스틱이며 전역 최적해를 보장하지 않습니다.
        - 이 결과로 범죄예방 효과나 지역의 안전 수준을 판단할 수 없습니다.

        다음에는 동일 가중치 결과와 평균 방문자 수 가중 결과가 다른 이유를 설명하고, 후보지 제약이나 행정경계를 추가해 분석이 어떻게 달라지는지 확인해 봅니다.
        '''
    ),
]


def execute_cells(notebook_cells: list[dict]) -> None:
    namespace: dict = {"__name__": "__main__"}
    execution_count = 0
    failures: list[str] = []

    for index, cell in enumerate(notebook_cells):
        if cell["cell_type"] != "code":
            continue

        execution_count += 1
        stdout = io.StringIO()
        stderr = io.StringIO()
        source = "".join(cell["source"])

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{NOTEBOOK_PATH.name}:cell-{index}", "exec"), namespace)
        except Exception:
            traceback.print_exc(file=stderr)
            failures.append(f"cell {index}")

        outputs = []
        if stdout.getvalue():
            outputs.append({"name": "stdout", "output_type": "stream", "text": lines(stdout.getvalue())})
        if stderr.getvalue():
            outputs.append({"name": "stderr", "output_type": "stream", "text": lines(stderr.getvalue())})
        cell["execution_count"] = execution_count
        cell["outputs"] = outputs

        if failures:
            print(stderr.getvalue())
            break

    if failures:
        raise RuntimeError("노트북 실행 중 오류가 발생했습니다: " + ", ".join(failures))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10+"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

execute_cells(cells)
NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"생성 완료: {NOTEBOOK_PATH}")
print(f"코드 셀 실행 완료: {sum(cell['cell_type'] == 'code' for cell in cells)}개")
