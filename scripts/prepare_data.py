"""주차에서 사용하는 ZIP 항목만 extracted에 준비합니다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import zipfile


@dataclass(frozen=True)
class Dataset:
    archive: str
    folder: str
    files: tuple[str, ...]


DATASETS = {
    "bike-rentals": Dataset("따릉이 공공데이터.zip", "따릉이 공공데이터", (
        "03_대여이력/서울특별시 공공자전거 대여이력 정보_2602.csv",
    )),
    "bike-daily": Dataset("따릉이 공공데이터.zip", "따릉이 공공데이터", tuple(
        f"02_이용정보/서울특별시 공공자전거 일별 대여건수_{period}.csv"
        for period in ("24.1-6", "24.7-12", "25.1-6", "25.7-12", "26.1-6")
    )),
    "citrus": Dataset("감귤 착과량 예측 AI 경진대회.zip", "감귤 착과량 예측 AI 경진대회", ("train.csv", "test.csv")),
    "power": Dataset("2023 전력사용량 예측 AI 경진대회.zip", "2023 전력사용량 예측 AI 경진대회", ("train.csv", "building_info.csv")),
    "logistics": Dataset("물류 유통량 예측 경진대회.zip", "물류 유통량 예측 경진대회", ("train.csv", "test.csv")),
    "loan": Dataset("loan-prediction-problem-dataset.zip", "Loan Prediction Problem Dataset", ("train_u6lujuX_CVtuZ9i.csv",)),
    "flights": Dataset("월간 데이콘 항공편 지연 예측 AI 경진대회.zip", "월간 데이콘 항공편 지연 예측 AI 경진대회", ("train.csv",)),
    "customers": Dataset("customer-personality-analysis.zip", "Customer Personality Analysis", ("marketing_campaign.csv",)),
    "ecommerce": Dataset("이커머스 고객 세분화 분석 아이디어 경진대회.zip", "이커머스 고객 세분화 분석 아이디어 경진대회", (
        "Customer_info.csv", "Discount_info.csv", "Marketing_info.csv", "Onlinesales_info.csv", "Tax_info.csv",
    )),
    "jeju": Dataset("제주도 도로 교통량 예측 AI 경진대회.zip", "제주도 도로 교통량 예측 AI 경진대회", ("open/train.csv",)),
    "credit": Dataset("신용카드 고객 세그먼트 분류 AI 경진대회.zip", "신용카드 고객 세그먼트 분류 AI 경진대회", (
        "train/1.회원정보/201807_train_회원정보.parquet",
        "train/2.신용정보/201807_train_신용정보.parquet",
    )),
}

DEFAULT_DATASETS = {
    "example": {1: "bike-rentals", 2: "bike-daily", 3: "bike-daily", 4: "bike-daily", 5: "loan", 6: "customers"},
    "assignment": {2: "citrus", 3: "power", 4: "logistics", 5: "flights", 6: "ecommerce"},
}


def prepare_data(root: Path, week: int, kind: str = "example", data: str | None = None) -> str:
    """기존 파일은 보존하고 없는 파일만 추출합니다. 선택한 데이터 이름을 반환합니다."""
    if week not in range(1, 9) or kind not in DEFAULT_DATASETS:
        raise ValueError("week는 1~8, kind는 example 또는 assignment여야 합니다.")
    selected = data if data is not None else DEFAULT_DATASETS[kind].get(week, "none")
    if selected != "none" and selected not in DATASETS:
        raise ValueError(f"지원하지 않는 데이터: {selected}")
    extracted = (root / "dataset/open/extracted").resolve()
    extracted.mkdir(parents=True, exist_ok=True)
    if selected == "none":
        if week == 8 and kind == "assignment":
            print("[데이터 선택] 준비 셀에 --data jeju 또는 --data credit을 추가하면 해당 트랙 파일만 압축 해제합니다.")
        elif week == 7 and kind == "example":
            print("[데이터 안내] 7주차 CCTV·S-DoT는 cctv_data_guide.md에 따라 별도로 준비합니다.")
        return selected

    dataset = DATASETS[selected]
    targets = {name: extracted / dataset.folder / name for name in dataset.files}
    for target in targets.values():
        if not target.resolve().is_relative_to(extracted):
            raise ValueError(f"extracted 밖의 경로로는 추출하지 않습니다: {target}")
    pending = {name: target for name, target in targets.items() if not target.is_file()}
    if not pending:
        print(f"[데이터 재사용] {selected}: {len(targets)}개 파일이 이미 있습니다.")
        return selected

    archive = root / "dataset/open/raw" / dataset.archive
    if not archive.is_file():
        print(f"[데이터 미준비] {archive}가 없습니다. 필요한 파일 {len(pending)}개를 직접 준비하거나 ZIP을 raw에 넣고 이 셀을 다시 실행하세요.")
        return selected

    with zipfile.ZipFile(archive) as source:
        members = set(source.namelist())
        chosen = {}
        for name in pending:
            # 따릉이 ZIP에는 데이터 폴더가 포함되고, DACON/Kaggle ZIP은 파일부터 시작합니다.
            candidates = (name, f"{dataset.folder}/{name}")
            member = next((value for value in candidates if value in members), None)
            if member is None:
                raise FileNotFoundError(f"{archive.name} 안에 필요한 항목이 없습니다: {name}")
            chosen[name] = member
        total = sum(source.getinfo(member).file_size for member in chosen.values())
        print(f"[압축 해제] {archive.name}: 필요한 {len(chosen)}개 파일 ({total / 1_000_000:.1f} MB)", flush=True)
        for name, member in chosen.items():
            target = pending[name]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".extract-", delete=False) as output:
                    temporary = Path(output.name)
                    with source.open(member) as entry:
                        # 끝까지 읽으면 zipfile이 CRC도 검증합니다.
                        shutil.copyfileobj(entry, output, length=1024 * 1024)
                if not target.exists():
                    temporary.replace(target)
                print(f"  준비: {dataset.folder}/{name}", flush=True)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    return selected
