import os
from dotenv import load_dotenv

load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 검색 키워드 (남북 관련)
KEYWORDS = [
    # 기존
    "남북관계", "북한", "통일", "2국가론",
    "북핵", "대북정책", "남북대화", "한반도",
    # 추가
    "종전선언",   # 문재인 정부 핵심 의제
    "비핵화",     # 북핵 협상 핵심 키워드
    "개성공단",   # 남북 경협 상징, 폐쇄·재개 논쟁
    "북미관계",   # 북핵 외교 맥락
    "대북제재",   # 제재 완화·강화 논쟁
    "남북정상",   # 남북 정상회담 관련
]

# 수집 기간
DATE_FROM = "2023-01-01"
DATE_TO   = "2024-12-31"

# 정치인 목록 (이념별)
# 선정 기준: 남북 담론 발언 빈도 상위자 + 탈북 출신 의원
POLITICIANS = {
    "보수": [
        "윤석열", "정진석", "조태용", "한덕수", "김기현", "한동훈", "추경호",
        "홍준표",   # 현 경북지사, 대북 강경 발언 다수
        "나경원",   # 현 외교부장관, 대북정책 핵심 논객
        "태영호",   # 탈북 외교관 출신 의원 — 북한 담론 핵심
        "주호영",   # 전 원내대표
        "이준석",   # 전 국민의힘 대표, 현재도 활발
    ],
    "진보": [
        "문재인", "이재명", "임종석", "박지원", "이인영", "김경협", "김민석",
        "정청래",   # 현 국회 법사위원장, 남북 담론 빈번한 발언
        "윤건영",   # 문재인 정부 국정상황실장, 남북 전문가
        "송영길",   # 전 민주당 대표, 대북 온건파 대표주자
        "우상호",   # 현역 의원, 남북 이슈 발언 多
        "이낙연",   # 전 민주당 대표
    ],
    "중도": [
        "안철수", "유승민",
    ],
}

# 전체 정치인 리스트 (평탄화)
ALL_POLITICIANS = [p for group in POLITICIANS.values() for p in group]

# 이념 역방향 조회 딕셔너리 {이름: 이념}
IDEOLOGY_MAP = {
    name: ideology
    for ideology, names in POLITICIANS.items()
    for name in names
}
