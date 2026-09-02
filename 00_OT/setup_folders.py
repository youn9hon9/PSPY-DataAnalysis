"""학습 노트북이 기대하는 공통 폴더 구조를 만듭니다.

이 스크립트는 폴더만 생성하며 기존 파일을 이동하거나 덮어쓰지 않습니다.
프로젝트 루트가 아니라 다른 위치에서 실행해도 스크립트 위치를 기준으로 동작합니다.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

COURSE_DIRECTORIES = (
    "00_OT/assets",
    *(f"{week:02d}주차/assets" for week in range(1, 9)),
)

CARD_DATA_TABLES = (
    "1.회원정보",
    "2.신용정보",
    "3.승인매출정보",
    "4.청구입금정보",
    "5.잔액정보",
    "6.채널정보",
    "7.마케팅정보",
    "8.성과정보",
)

DATASET_DIRECTORIES = (
    "dataset/raw/KRX",
    "dataset/extracted/KRX",
    "dataset/extracted/따릉이 공공데이터/02_이용정보",
    "dataset/extracted/따릉이 공공데이터/03_대여이력",
    "dataset/extracted/감귤 착과량 예측 AI 경진대회",
    "dataset/extracted/2023 전력사용량 예측 AI 경진대회",
    "dataset/extracted/물류 유통량 예측 경진대회",
    "dataset/extracted/Loan Prediction Problem Dataset",
    "dataset/extracted/월간 데이콘 항공편 지연 예측 AI 경진대회",
    "dataset/extracted/Customer Personality Analysis",
    "dataset/extracted/이커머스 고객 세분화 분석 아이디어 경진대회",
    "dataset/extracted/서울시 CCTV 공공데이터/01_자치구_CCTV_설치현황/01_목적별",
    "dataset/extracted/서울시 CCTV 공공데이터/01_자치구_CCTV_설치현황/02_범죄예방_수사용",
    "dataset/extracted/서울시 CCTV 공공데이터/01_자치구_CCTV_설치현황/03_연도별",
    "dataset/extracted/서울시 CCTV 공공데이터/01_자치구_CCTV_설치현황/04_지능형_설치현황",
    "dataset/extracted/서울시 CCTV 공공데이터/01_자치구_CCTV_설치현황/05_지능형_수량",
    "dataset/extracted/서울시 CCTV 공공데이터/02_교통_단속_CCTV/01_도시고속도로",
    "dataset/extracted/서울시 CCTV 공공데이터/02_교통_단속_CCTV/02_불법주정차",
    "dataset/extracted/서울시 CCTV 공공데이터/02_교통_단속_CCTV/03_전용차로",
    "dataset/extracted/서울시 CCTV 공공데이터/03_S-DoT_유동인구/01_설치위치",
    "dataset/extracted/서울시 CCTV 공공데이터/03_S-DoT_유동인구/02_측정데이터/2026",
    "dataset/extracted/서울시 CCTV 공공데이터/04_연계_안전시설/01_스마트폴",
    "dataset/extracted/서울시 CCTV 공공데이터/04_연계_안전시설/02_안심귀갓길_데이터사전",
    "dataset/extracted/서울시 CCTV 공공데이터/99_메타데이터_설명서",
    "dataset/extracted/제주도 도로 교통량 예측 AI 경진대회/open",
    *(
        f"dataset/extracted/신용카드 고객 세그먼트 분류 AI 경진대회/{split}/{table}"
        for split in ("train", "test")
        for table in CARD_DATA_TABLES
    ),
    "dataset/processed",
)


def main() -> None:
    directories = COURSE_DIRECTORIES + DATASET_DIRECTORIES

    for relative_path in directories:
        (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)

    print("[완료] 학습 공통 폴더를 확인했습니다.")
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"데이터 입력 폴더: {PROJECT_ROOT / 'dataset' / 'extracted'}")
    print(f"분석 산출물 폴더: {PROJECT_ROOT / 'dataset' / 'processed'}")


if __name__ == "__main__":
    main()
