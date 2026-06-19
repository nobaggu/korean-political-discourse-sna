"""
Step 2b: 정치인 1명 이상 등장 기사 전체 크롤링 (balance 필터 없음)

- articles.json (Step 1 결과, 342건) 읽기
- 본문 크롤링
- 정치인 1명 이상 등장이면 전부 저장 (보수+진보 동시등장 조건 없음)
- 결과를 data/raw/articles_with_text_all.json에 저장

용도: 그래프 2 (정치인-키워드 이분 네트워크) 데이터
     → 각 진영이 어떤 주제를 다루는지 분석

함수 흐름:
  crawl_all()
    ├─ articles.json 읽기 (342건)
    ├─ [루프] 각 기사 URL 크롤링
    │    ├─ crawl_naver_news() 또는 crawl_general_news()
    │    └─ 본문에서 정치인 재추출
    ├─ 정치인 1명 이상이면 저장 (이념 균형 조건 없음)
    └─ articles_with_text_all.json 저장
"""

import requests
import json
import time
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import os as _os

def _load_assembly():
    path = _os.path.join(_os.path.dirname(__file__), "data", "assembly_members.json")
    with open(path, "r", encoding="utf-8") as f:
        import json as _json
        return _json.load(f)

_ASSEMBLY = _load_assembly()
ALL_POLITICIANS = list(_ASSEMBLY.keys())
IDEOLOGY_MAP    = {name: info["ideology"] for name, info in _ASSEMBLY.items()}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def crawl_naver_news(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for selector in ["#dic_area", "#articleBodyContents", ".go_trans._article_content", "#newsct_article"]:
            tag = soup.select_one(selector)
            if tag:
                return tag.get_text(separator="\n", strip=True)
        return None
    except:
        return None


def crawl_general_news(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag_name in ["article", "main", ".article-body", ".article_body", "#article-view-content-div"]:
            tag = soup.select_one(tag_name)
            if tag and len(tag.get_text(strip=True)) > 200:
                return tag.get_text(separator="\n", strip=True)
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30)
        return text if len(text) > 200 else None
    except:
        return None


def crawl_all():
    with open("data/raw/articles.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"총 {len(articles)}건 크롤링 시작 (balance 필터 없음)\n")

    results = []
    failed = 0

    for article in tqdm(articles, desc="본문 크롤링 중"):
        url = article.get("link", "")

        if "news.naver.com" in url:
            full_text = crawl_naver_news(url)
        else:
            full_text = crawl_general_news(url)

        if not full_text:
            failed += 1
            time.sleep(0.3)
            continue

        # 본문에서 정치인 재추출
        politicians_found = [p for p in ALL_POLITICIANS if p in full_text]

        # 정치인 1명 이상이면 저장
        if not politicians_found:
            time.sleep(0.3)
            continue

        article["full_text"] = full_text
        article["politicians_found"] = politicians_found
        article["ideologies_found"] = list(set(
            IDEOLOGY_MAP.get(p) for p in politicians_found
        ))
        results.append(article)
        time.sleep(0.5)

    with open("data/raw/articles_with_text_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 크롤링 완료")
    print(f"  - 수집 기사: {len(results)}건")
    print(f"  - 본문 추출 실패: {failed}건")
    print(f"  - 저장 위치: data/raw/articles_with_text_all.json")


if __name__ == "__main__":
    crawl_all()
