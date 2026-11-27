"""노트북이 읽는 두 CSV와 같은 컬럼·인코딩의 합성 데이터를 만든다."""

from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "branch_recommendation.ipynb"

AMOUNT_COLUMNS = [
    "요구불예금잔액", "거치식예금잔액", "적립식예금잔액", "수익증권잔액", "신탁잔액", "퇴직연금잔액",
    "여신한도금액", "여신_운전자금대출잔액", "여신_시설자금대출잔액", "외환_수출실적금액", "외환_수입실적금액",
    "신용카드사용금액", "체크카드사용금액", "창구거래금액", "인터넷뱅킹거래금액", "스마트뱅킹거래금액",
    "폰뱅킹거래금액", "ATM거래금액", "자동이체금액", "요구불입금금액", "요구불출금금액",
]
COUNT_COLUMNS = [
    "요구불예금좌수", "거치식예금좌수", "적립식예금좌수", "수익증권좌수", "신탁좌수", "퇴직연금좌수",
    "여신_운전자금대출좌수", "여신_시설자금대출좌수", "신용카드개수", "외환_수출실적거래건수",
    "외환_수입실적거래건수", "창구거래건수", "인터넷뱅킹거래건수", "스마트뱅킹거래건수", "폰뱅킹거래건수",
    "ATM거래건수", "자동이체거래건수",
]
SYNTHETIC_REGIONS = [
    ("서울특별시", "강남구"), ("서울특별시", "중구"), ("서울특별시", "영등포구"), ("서울특별시", "서초구"),
    ("서울특별시", "구로구"), ("경기도", "화성시"), ("경기도", "시흥시"), ("경기도", "평택시"),
    ("경기도", "안성시"), ("경기도", "김포시"), ("경기도", "성남시 분당구"), ("인천광역시", "서구"),
    ("인천광역시", "남동구"), ("대구광역시", "수성구"), ("대구광역시", "달서구"), ("대구광역시", "북구"),
    ("경상북도", "구미시"), ("경상북도", "경산시"), ("부산광역시", "영도구"), ("부산광역시", "사상구"),
    ("울산광역시", "남구"), ("경상남도", "창원시 진해구"), ("대전광역시", "유성구"), ("광주광역시", "북구"),
]


def write_synthetic_data(data_dir: Path, seed: int = 0) -> None:
    """노트북이 읽는 두 CSV와 같은 컬럼·인코딩의 합성 데이터를 data_dir에 쓴다."""
    rng = np.random.default_rng(seed)
    rows = []
    for month in (202411, 202412):
        for province, district in SYNTHETIC_REGIONS:
            for _ in range(int(rng.integers(5, 20))):
                row = {
                    "기준년월": month,
                    "업종_대분류": "제조업",
                    "업종_중분류": "기타 제조업",
                    "사업장_시도": province,
                    "사업장_시군구": district,
                    "법인_고객등급": rng.choice(["일반", "우수", "최우수"]),
                    "전담고객여부": rng.choice(["Y", "N"]),
                }
                for col in AMOUNT_COLUMNS:
                    value = round(float(rng.lognormal(3, 2)) * float(rng.random() < 0.6), 2)
                    # clean_amount의 콤마 제거 경로를 거치도록 1000 이상은 콤마가 든 문자열로 쓴다.
                    row[col] = f"{value:,.2f}" if value >= 1000 else value
                for col in COUNT_COLUMNS:
                    unit = "건" if col.endswith("건수") else "개"
                    buckets = [f"0{unit}", f"1{unit}", f"2{unit}초과 5{unit}이하", f"10{unit}초과 20{unit}이하"]
                    row[col] = rng.choice([*buckets, f"50{unit} 초과"])
                rows.append(row)
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(data_dir / "imbank_data.csv", index=False, encoding="cp949")

    branches = [
        {
            "검색지역키워드": f"{province} {district}",
            "지점명": f"지점{i}-{j}",
            "은행명": rng.choice(["가은행", "나은행", "다은행"]),
        }
        for i, (province, district) in enumerate(SYNTHETIC_REGIONS[:-2])
        for j in range(int(rng.integers(1, 6)))
    ]
    # 노트북의 "(미상)" 처리 경로를 거치도록 시군구가 없는 키워드도 하나 넣는다.
    branches.append({"검색지역키워드": "세종특별자치시", "지점명": "세종", "은행명": "가은행"})
    pd.DataFrame(branches).to_csv(data_dir / "시중은행_위치(365,ATM)제외.csv", index=False, encoding="utf-8-sig")
