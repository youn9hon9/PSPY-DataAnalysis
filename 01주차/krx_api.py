"""1주차 KRX OPEN API 실습용 수집 도구입니다.

인증키는 ``KRX_API_KEY`` 환경 변수 또는 프로젝트 루트의 ``.env``에서
읽습니다. 키 자체는 로그, 파일명, 메타데이터에 기록하지 않습니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests


@dataclass(frozen=True)
class EndpointSpec:
    """KRX 데이터셋 한 종류의 고정 사양입니다."""

    label: str
    url: str
    file_stem: str
    identity_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]


ENDPOINTS: Mapping[str, EndpointSpec] = {
    "krx_index": EndpointSpec(
        label="KRX 시리즈 일별시세정보",
        url="https://data-dbg.krx.co.kr/svc/apis/idx/krx_dd_trd",
        file_stem="krx_index",
        identity_columns=("IDX_CLSS", "IDX_NM"),
        numeric_columns=(
            "CLSPRC_IDX",
            "CMPPREVDD_IDX",
            "FLUC_RT",
            "OPNPRC_IDX",
            "HGPRC_IDX",
            "LWPRC_IDX",
            "ACC_TRDVOL",
            "ACC_TRDVAL",
            "MKTCAP",
        ),
    ),
    "kosdaq_stocks": EndpointSpec(
        label="코스닥 일별매매정보",
        url="https://data-dbg.krx.co.kr/svc/apis/sto/ksq_bydd_trd",
        file_stem="kosdaq_stocks",
        identity_columns=("ISU_CD", "ISU_NM", "MKT_NM", "SECT_TP_NM"),
        numeric_columns=(
            "TDD_CLSPRC",
            "CMPPREVDD_PRC",
            "FLUC_RT",
            "TDD_OPNPRC",
            "TDD_HGPRC",
            "TDD_LWPRC",
            "ACC_TRDVOL",
            "ACC_TRDVAL",
            "MKTCAP",
            "LIST_SHRS",
        ),
    ),
}

MIN_DATE = datetime.strptime("20100104", "%Y%m%d").date()
DATE_PATTERN = re.compile(r"^\d{8}$")


class KRXAPIError(RuntimeError):
    """KRX 응답을 신뢰할 수 없을 때 발생합니다."""


def project_root() -> Path:
    """이 파일 위치를 기준으로 프로젝트 루트를 반환합니다."""

    return Path(__file__).resolve().parent.parent


def validate_bas_dd(value: str) -> str:
    """기준일을 API가 요구하는 YYYYMMDD 형식으로 검증합니다."""

    if not DATE_PATTERN.fullmatch(value):
        raise ValueError("기준일은 YYYYMMDD 형식의 8자리 문자열이어야 합니다.")
    parsed = datetime.strptime(value, "%Y%m%d").date()
    if parsed < MIN_DATE:
        raise ValueError("KRX OPEN API의 데이터 제공 시작일인 20100104 이후를 입력합니다.")
    if parsed > date.today():
        raise ValueError("미래 날짜는 조회할 수 없습니다.")
    return value


def _read_dotenv_value(path: Path, variable: str) -> str | None:
    """별도 패키지 없이 .env의 단일 값을 읽습니다."""

    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == variable:
            return value.strip().strip("'\"")
    return None


def load_api_key(env_path: Path | None = None) -> str:
    """환경 변수 또는 .env에서 KRX 인증키를 읽되 값을 출력하지 않습니다."""

    variable = "KRX_API_KEY"
    value = os.getenv(variable)
    if not value:
        value = _read_dotenv_value(env_path or project_root() / ".env", variable)
    if not value:
        raise KRXAPIError(
            "KRX_API_KEY가 없습니다. .env.example을 복사해 .env를 만들고 키를 설정합니다."
        )
    return value


def request_payload(
    dataset: str,
    bas_dd: str,
    *,
    api_key: str | None = None,
    timeout: tuple[int, int] = (5, 30),
    requester: Any = requests,
) -> dict[str, Any]:
    """KRX API를 한 번 호출하고 예상 JSON 스키마를 검증합니다."""

    if dataset not in ENDPOINTS:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
    bas_dd = validate_bas_dd(bas_dd)
    spec = ENDPOINTS[dataset]

    try:
        response = requester.get(
            spec.url,
            headers={"AUTH_KEY": api_key or load_api_key()},
            params={"basDd": bas_dd},
            timeout=timeout,
            allow_redirects=False,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise KRXAPIError(f"KRX API 요청에 실패했습니다: {type(exc).__name__}") from exc

    content_type = getattr(response, "headers", {}).get("Content-Type", "")
    if content_type and "json" not in content_type.lower():
        raise KRXAPIError(f"예상하지 못한 응답 형식입니다: {content_type.split(';', 1)[0]}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise KRXAPIError("KRX 응답이 JSON 형식이 아닙니다.") from exc

    if not isinstance(payload, dict):
        raise KRXAPIError("KRX 응답의 최상위 자료형이 객체가 아닙니다.")
    if "OutBlock_1" not in payload:
        keys = ", ".join(sorted(map(str, payload.keys()))) or "없음"
        raise KRXAPIError(f"예상한 OutBlock_1이 없습니다. 실제 최상위 키: {keys}")
    if not isinstance(payload["OutBlock_1"], list):
        raise KRXAPIError("OutBlock_1이 행 목록 형식이 아닙니다.")
    return payload


def payload_to_frame(dataset: str, payload: Mapping[str, Any]) -> pd.DataFrame:
    """원본 행 목록을 분석 가능한 자료형의 DataFrame으로 변환합니다."""

    if dataset not in ENDPOINTS:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
    rows = payload.get("OutBlock_1")
    if not isinstance(rows, list):
        raise KRXAPIError("OutBlock_1이 행 목록 형식이 아닙니다.")

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    if "BAS_DD" not in frame.columns:
        raise KRXAPIError("응답 행에 기준일 BAS_DD가 없습니다.")
    frame["BAS_DD"] = pd.to_datetime(frame["BAS_DD"], format="%Y%m%d", errors="raise")

    for column in ENDPOINTS[dataset].identity_columns:
        if column not in frame.columns:
            raise KRXAPIError(f"응답 행에 예상 식별 열 {column}이 없습니다.")

    for column in ENDPOINTS[dataset].numeric_columns:
        if column not in frame.columns:
            raise KRXAPIError(f"응답 행에 예상 숫자 열 {column}이 없습니다.")
        cleaned = frame[column].astype("string").str.replace(",", "", regex=False)
        frame[column] = pd.to_numeric(cleaned, errors="coerce")
    return frame


def _atomic_write_text(path: Path, text: str, *, encoding: str) -> None:
    """같은 폴더의 임시 파일을 거쳐 텍스트를 교체합니다."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    """같은 폴더의 임시 파일을 거쳐 CSV를 교체합니다."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def snapshot_paths(dataset: str, bas_dd: str, root: Path | None = None) -> dict[str, Path]:
    """원본 JSON, 분석용 CSV, 메타데이터 경로를 계산합니다."""

    if dataset not in ENDPOINTS:
        raise ValueError(f"지원하지 않는 데이터셋입니다: {dataset}")
    bas_dd = validate_bas_dd(bas_dd)
    root = (root or project_root()).resolve()
    stem = f"{ENDPOINTS[dataset].file_stem}_{bas_dd}"
    return {
        "raw": root / "dataset" / "raw" / "KRX" / f"{stem}.json",
        "csv": root / "dataset" / "extracted" / "KRX" / f"{stem}.csv",
        "metadata": root / "dataset" / "extracted" / "KRX" / f"{stem}.metadata.json",
    }


def collect_dataset(
    dataset: str,
    bas_dd: str,
    *,
    root: Path | None = None,
    api_key: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """한 날짜의 데이터를 호출하고 원본·가공본·메타데이터로 저장합니다."""

    root = (root or project_root()).resolve()
    paths = snapshot_paths(dataset, bas_dd, root)
    if not overwrite and all(path.exists() for path in paths.values()):
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        return {**metadata, "cached": True}
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(
            "일부 캐시 파일만 존재합니다. 상태를 확인한 뒤 --force로 다시 수집합니다."
        )

    payload = request_payload(dataset, bas_dd, api_key=api_key)
    frame = payload_to_frame(dataset, payload)
    if frame.empty:
        raise KRXAPIError(
            f"{bas_dd} 응답에 행이 없습니다. 주말·휴장일인지 확인합니다."
        )

    returned_dates = set(frame["BAS_DD"].dt.strftime("%Y%m%d"))
    if returned_dates != {bas_dd}:
        raise KRXAPIError(
            f"요청일과 응답 기준일이 다릅니다. 응답 기준일: {sorted(returned_dates)}"
        )

    spec = ENDPOINTS[dataset]
    key_columns = (
        ["BAS_DD", "IDX_CLSS", "IDX_NM"]
        if dataset == "krx_index"
        else ["BAS_DD", "ISU_CD"]
    )
    duplicate_count = int(frame.duplicated(key_columns).sum())
    if duplicate_count:
        raise KRXAPIError(
            f"예상 고유키 {key_columns}에 중복 {duplicate_count:,}건이 있습니다."
        )

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = json.dumps(payload, ensure_ascii=False, indent=2)
    _atomic_write_text(paths["raw"], raw_text, encoding="utf-8")
    _atomic_write_csv(paths["csv"], frame)

    metadata = {
        "dataset": dataset,
        "label": spec.label,
        "source_url": spec.url,
        "request_params": {"basDd": bas_dd},
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "key_columns": key_columns,
        "duplicate_key_rows": duplicate_count,
        "raw_sha256": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "raw_file": paths["raw"].relative_to(root).as_posix(),
        "csv_file": paths["csv"].relative_to(root).as_posix(),
        "cached": False,
    }
    _atomic_write_text(
        paths["metadata"],
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def load_dataset(dataset: str, bas_dd: str, *, root: Path | None = None) -> pd.DataFrame:
    """이미 수집한 분석용 CSV를 식별자·날짜 자료형을 보존해 읽습니다."""

    path = snapshot_paths(dataset, bas_dd, root)["csv"]
    if not path.exists():
        raise FileNotFoundError(f"수집 파일이 없습니다: {path}")
    dtype = {"ISU_CD": "string"} if dataset == "kosdaq_stocks" else None
    return pd.read_csv(path, parse_dates=["BAS_DD"], dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description="KRX 1일 스냅샷 수집기")
    parser.add_argument("--date", default="20240823", help="기준일 YYYYMMDD")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(ENDPOINTS),
        default=list(ENDPOINTS),
        help="수집할 데이터셋",
    )
    parser.add_argument("--force", action="store_true", help="기존 캐시를 덮어씁니다.")
    args = parser.parse_args()

    for dataset in args.datasets:
        result = collect_dataset(dataset, args.date, overwrite=args.force)
        state = "캐시 사용" if result["cached"] else "API 수집"
        print(f"[{state}] {result['label']}: {result['rows']:,}행 → {result['csv_file']}")


if __name__ == "__main__":
    main()
