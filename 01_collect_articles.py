"""
Step 1: 네이버 뉴스 API로 남북 관련 기사 수집 (날짜 필터 적용)
- API로 최대 1000건 수집 후 pubDate로 2024~2025 기간 필터링
- JS 렌더링 문제로 웹 스크래핑 대신 API 사용
- 결과를 data/raw/articles.json에 저장
"""

import requests
import json
import time
import os
import re
from datetime import datetime
from tqdm import tqdm
from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

os.makedirs("data/raw", exist_ok=True)

# 수집 기간
DATE_FROM = "2024-01-01"
DATE_TO   = "2025-03-31"

KEYWORDS = [
    "남북관계", "북한", "통일", "2국가론",
    "북핵", "대북정책", "남북대화", "한반도",
    "종전선언", "비핵화", "개성공단", "북미관계",
    "대북제재", "남북정상",
]


def load_assembly():
    with open("data/assembly_members.json", "r", encoding="utf-8") as f:
        return json.load(f)


def search_naver_news(query, display=100, start=1):
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "start": start, "sort": "date"}
    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        return resp.json()
    return None


def parse_date(pub_date_str):
    """'Mon, 01 Jan 2024 12:00:00 +0900' → '2024-01-01'"""
    try:
        dt = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")
        return dt.strftime("%Y-%m-%d")
    except:
        return ""


def in_date_range(date_str):
    return DATE_FROM <= date_str <= DATE_TO


def collect_articles():
    assembly = load_assembly()
    all_names = list(assembly.keys())

    all_articles = []
    seen_links   = set()

    for keyword in tqdm(KEYWORDS, desc="키워드 수집 중"):
        keyword_count = 0
        hit_old_date  = False  # 수집 기간 이전 기사 만나면 중단

        for start in range(1, 1001, 100):  # 최대 1000건
            result = search_naver_news(keyword, display=100, start=start)
            if not result or not result.get("items"):
                break

            for item in result["items"]:
                link     = item.get("link", "")
                pub_date = parse_date(item.get("pubDate", ""))

                # 수집 기간 이후 기사 → 스킵
                if pub_date > DATE_TO:
                    continue

                # 수집 기간 이전 기사 → 이후는 더 오래됐으므로 중단
                if pub_date and pub_date < DATE_FROM:
                    hit_old_date = True
                    break

                if link in seen_links:
                    continue
                seen_links.add(link)

                text = re.sub(r"<[^>]+>", "", item.get("title", "") + " " + item.get("description", ""))
                politicians_found = [p for p in all_names if p in text]

                if not politicians_found:
                    continue

                all_articles.append({
                    "keyword":          keyword,
                    "title":            re.sub(r"<[^>]+>", "", item.get("title", "")),
                    "description":      re.sub(r"<[^>]+>", "", item.get("description", "")),
                    "link":             link,
                    "originallink":     item.get("originallink", ""),
                    "pub_date":         pub_date,
                    "politicians_found": politicians_found,
                    "ideologies_found": list(set(assembly[p]["ideology"] for p in politicians_found)),
                })
                keyword_count += 1

            if hit_old_date:
                break
            time.sleep(0.1)

        print(f"  [{keyword}] {keyword_count}건")

    output_path = "data/raw/articles.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 수집 완료: 총 {len(all_articles)}건 → {output_path}")
    print(f"수집 기간: {DATE_FROM} ~ {DATE_TO}")


if __name__ == "__main__":
    collect_articles()
