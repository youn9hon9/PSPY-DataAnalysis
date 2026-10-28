"""%run으로 실행해 주차 경로, 공통 패키지와 한글 폰트를 준비합니다."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_data import DATASETS, prepare_data


def setup(week: int, root: Path = ROOT, *, kind: str = "example", data: str | None = None) -> dict:
    root = root.expanduser().resolve()
    week_dir = root / f"{week:02d}주차"
    if week not in range(1, 9) or not week_dir.is_dir() or not (root / "requirements.txt").is_file():
        raise FileNotFoundError(f"requirements.txt와 {week:02d}주차 폴더를 확인하세요: {root}")

    try:
        import google.colab  # noqa: F401
    except ImportError:
        in_colab = False
    else:
        in_colab = True

    if in_colab:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(root / "requirements.txt")])
        if not Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf").is_file():
            subprocess.check_call(["apt-get", "update", "-qq"])
            subprocess.check_call(["apt-get", "install", "-y", "-qq", "fonts-nanum"])

    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import pandas as pd
        import sklearn
    except ImportError as error:
        raise RuntimeError(
            "공통 패키지가 없습니다. 로컬에서는 python scripts/setup_local.py --reinstall 실행 후 "
            "Python (PSPY .venv) 커널을 선택하세요."
        ) from error

    for font_path in (
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
    ):
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))
    available = {font.name for font in font_manager.fontManager.ttflist}
    font_name = next((name for name in (
        "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP",
    ) if name in available), None)
    if font_name:
        plt.rcParams["font.family"] = font_name
    else:
        print("한글 폰트가 없습니다. 나눔고딕 설치 후 커널을 다시 시작하세요.")
    plt.rcParams["axes.unicode_minus"] = False

    selected = prepare_data(root, week, kind=kind, data=data)
    os.chdir(week_dir)
    if str(week_dir) not in sys.path:
        sys.path.insert(0, str(week_dir))
    print(f"준비 완료: {'Colab' if in_colab else '로컬'} / {week_dir}")
    print(f"Python: {sys.executable}")
    print(f"pandas {pd.__version__} / scikit-learn {sklearn.__version__} / 폰트 {font_name or '없음'}")
    return {
        "PROJECT_ROOT": root, "WEEK_DIR": week_dir, "WEEK_DIR_NAME": week_dir.name,
        "DATA_DIR": root / "dataset" / "open" / "extracted", "IN_COLAB": in_colab,
        "DATASET": selected,
        "pd": pd, "plt": plt,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--kind", choices=("example", "assignment"), default="example")
    parser.add_argument("--data", choices=("none", *DATASETS), help="자동 선택 대신 준비할 데이터 이름")
    args = parser.parse_args()
    globals().update(setup(args.week, kind=args.kind, data=args.data))
