"""
Step 2: 수집된 기사 URL에서 본문 전체 크롤링
- data/raw/articles.json 읽기
- 각 기사 URL 접속 → 본문 텍스트 추출
- 보수+진보 동시 등장 기사만 저장 (크롤링 효율화 사전 필터)
- 결과를 data/raw/articles_with_text.json에 저장

크롤링 원리:
  requests로 URL 접속 → HTML 받기 → BeautifulSoup으로 본문 태그 추출
"""

import requests
import json
import time
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

# assembly_members.json (313명: 299명 현역 + 대통령 등 주요 인물)
def _load_assembly():
    path = os.path.join(os.path.dirname(__file__), "data", "assembly_members.json")
    with open(path, "r", encoding="utf-8") as f:
        import json as _json
        return _json.load(f)

_ASSEMBLY = _load_assembly()
ALL_POLITICIANS = list(_ASSEMBLY.keys())
IDEOLOGY_MAP    = {name: info["ideology"] for name, info in _ASSEMBLY.items()}

# 브라우저처럼 보이기 위한 헤더 (없으면 일부 사이트가 차단)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def crawl_naver_news(url):
    """네이버 뉴스 본문 크롤링"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # 네이버 뉴스 본문 태그 (뉴스 유형마다 다를 수 있음)
        content = None
        for selector in ["#dic_area", "#articleBodyContents", ".go_trans._article_content", "#newsct_article"]:
            tag = soup.select_one(selector)
            if tag:
                content = tag.get_text(separator="\n", strip=True)
                break

        return content
    except Exception as e:
        return None


def crawl_general_news(url):
    """일반 언론사 기사 본문 크롤링 (네이버 외)"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # 공통적으로 본문이 들어있는 태그들 시도
        for tag_name in ["article", "main", ".article-body", ".article_body", "#article-view-content-div"]:
            tag = soup.select_one(tag_name)
            if tag and len(tag.get_text(strip=True)) > 200:
                return tag.get_text(separator="\n", strip=True)

        # 위 태그 못 찾으면 <p> 태그 전체 합치기
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)
        return text if len(text) > 200 else None

    except Exception as e:
        return None


def extract_politicians(text):
    """텍스트에서 정치인 목록 추출"""
    return [p for p in ALL_POLITICIANS if p in text]


def has_ideological_balance(politicians):
    """보수+진보 동시 등장 여부 확인 (공론장 조건 1)"""
    ideologies = set(IDEOLOGY_MAP.get(p) for p in politicians)
    return "보수" in ideologies and "진보" in ideologies


def crawl_articles():
    # 1단계 결과 읽기
    with open("data/raw/articles.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"총 {len(articles)}건 크롤링 시작\n")

    results = []
    failed = 0

    for article in tqdm(articles, desc="본문 크롤링 중"):
        url = article.get("link", "")

        # 네이버 뉴스 vs 외부 링크 구분
        if "news.naver.com" in url:
            full_text = crawl_naver_news(url)
        else:
            full_text = crawl_general_news(url)

        if not full_text:
            failed += 1
            time.sleep(0.3)
            continue

        # 본문에서 정치인 재추출 (요약보다 훨씬 정확)
        politicians_found = extract_politicians(full_text)

        # 크롤링 전 사전 필터: 보수+진보 동시 등장 기사만 저장
        # (수백 건 전부 크롤링하면 시간·서버 부하 과다 → 사전 필터로 효율화)
        if not has_ideological_balance(politicians_found):
            time.sleep(0.3)
            continue

        article["full_text"] = full_text
        article["politicians_found"] = politicians_found
        article["ideologies_found"] = list(set(
            IDEOLOGY_MAP.get(p) for p in politicians_found
        ))
        results.append(article)

        time.sleep(0.5)  # 서버 부하 방지

    # 저장
    with open("data/raw/articles_with_text.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 크롤링 완료")
    print(f"  - 수집 기사: {len(results)}건")
    print(f"  - 본문 추출 실패: {failed}건")
    print(f"  - 저장 위치: data/raw/articles_with_text.json")


if __name__ == "__main__":
    crawl_articles()
