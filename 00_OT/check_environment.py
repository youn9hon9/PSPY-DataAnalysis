"""학습을 시작하기 전에 파이썬 환경과 공통 폴더를 읽기 전용으로 점검합니다.

파일을 만들거나 옮기지 않으며, 인증키와 데이터 내용도 읽지 않습니다.
"""

from importlib import metadata
from pathlib import Path
import sys


# Windows 터미널과 IDE에서도 한글 안내가 같은 인코딩으로 보이게 합니다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PACKAGES = (
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("scikit-learn", "scikit-learn"),
    ("requests", "requests"),
    ("openpyxl", "openpyxl"),
    ("jupyterlab", "jupyterlab"),
)

OPTIONAL_PACKAGES = (
    ("folium", "7주차 선택형 지도"),
    ("pyarrow", "8주차 신용카드 Parquet 트랙"),
)


def installed_version(distribution):
    """설치된 배포판 버전을 반환하고, 없으면 None을 반환합니다."""

    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def main():
    print("[환경 확인] 이 도구는 파일과 인증키를 변경하거나 출력하지 않습니다.")
    print(f"파이썬 실행 파일: {sys.executable}")
    print(f"파이썬 버전: {sys.version.split()[0]}")
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"현재 작업 폴더: {Path.cwd().resolve()}")

    if sys.version_info >= (3, 10):
        print("[정상] Python 3.10 이상입니다.")
    else:
        print("[확인 필요] 이 자료는 Python 3.10 이상을 권장합니다.")

    print("\n[필수 패키지]")
    missing_required = []
    for distribution, label in REQUIRED_PACKAGES:
        version = installed_version(distribution)
        if version is None:
            missing_required.append(distribution)
            print(f"[설치 필요] {label}")
        else:
            print(f"[정상] {label} {version}")

    print("\n[선택 패키지]")
    for distribution, purpose in OPTIONAL_PACKAGES:
        version = installed_version(distribution)
        if version is None:
            print(f"[선택] {distribution}: {purpose}를 진행할 때만 설치합니다.")
        else:
            print(f"[정상] {distribution} {version}: {purpose}")

    print("\n[공통 폴더]")
    expected_directories = (
        PROJECT_ROOT / "00_OT",
        *(PROJECT_ROOT / f"{week:02d}주차" for week in range(1, 9)),
        PROJECT_ROOT / "dataset" / "extracted",
        PROJECT_ROOT / "dataset" / "processed",
    )
    missing_directories = [path for path in expected_directories if not path.is_dir()]
    if missing_directories:
        for path in missing_directories:
            print(f"[확인 필요] 폴더가 없습니다: {path.relative_to(PROJECT_ROOT)}")
        print("다음 명령으로 비어 있는 공통 폴더만 만들 수 있습니다:")
        print("  python 00_OT/setup_folders.py")
    else:
        print("[정상] 00_OT, 01~08주차, dataset/extracted, dataset/processed가 있습니다.")

    print("\n[다음 행동]")
    if missing_required:
        print("필수 패키지가 없다면 프로젝트 루트에서 다음 명령을 실행합니다:")
        print("  python -m pip install -r requirements.txt")
    else:
        print("필수 패키지와 공통 폴더의 기본 점검을 마쳤습니다.")
    print("이 도구는 실제 데이터 파일 유무까지 판단하지 않습니다. 각 주차의 데이터 점검 안내를 이어서 확인합니다.")


if __name__ == "__main__":
    main()
