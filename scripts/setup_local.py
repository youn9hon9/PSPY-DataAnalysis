"""프로젝트 전용 가상환경을 준비하고 JupyterLab을 실행합니다."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]


def prepare(root: Path = ROOT, *, reinstall: bool = False) -> Path:
    environment = root / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    needs_environment = not python.is_file()
    if not needs_environment:
        check = subprocess.run(
            [str(python), "-m", "pip", "--version"], capture_output=True, check=False,
        )
        needs_environment = check.returncode != 0
    if needs_environment:
        print("프로젝트 .venv를 만듭니다.", flush=True)
        try:
            venv.EnvBuilder(with_pip=True).create(environment)
        except (subprocess.CalledProcessError, SystemExit) as error:
            raise RuntimeError(
                "가상환경의 pip 준비에 실패했습니다. Ubuntu/Debian에서는 "
                "sudo apt install python3-venv로 venv 지원을 설치한 뒤 같은 명령을 다시 실행하세요. "
                "그 외 환경에서는 Python의 venv/ensurepip 설치 상태를 확인하세요."
            ) from error

    requirements = root / "requirements.txt"
    signature = hashlib.sha256(requirements.read_bytes()).hexdigest()
    marker = environment / ".pspy-requirements.sha256"
    if reinstall or not marker.is_file() or marker.read_text().strip() != signature:
        subprocess.check_call([str(python), "-m", "pip", "install", "-r", str(requirements)])
        subprocess.check_call([
            str(python), "-m", "ipykernel", "install", "--sys-prefix",
            "--name", "pspy-dataanalysis", "--display-name", "Python (PSPY .venv)",
        ])
        marker.write_text(signature + "\n")

    subprocess.check_call([str(python), str(root / "00_OT" / "setup_folders.py")])
    print(f"준비 완료: {python}", flush=True)
    print("노트북 커널은 Python (PSPY .venv)를 선택하세요.", flush=True)
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-launch", action="store_true", help="설치만 하고 JupyterLab은 열지 않습니다.")
    parser.add_argument("--reinstall", action="store_true", help="requirements.txt 설치와 커널 등록을 다시 진행합니다.")
    args = parser.parse_args()
    python = prepare(reinstall=args.reinstall)
    if not args.no_launch:
        subprocess.check_call([str(python), "-m", "jupyterlab"], cwd=ROOT)


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 이상으로 실행하세요.")
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
