"""API 응답 확인용 디버그 스크립트"""
import requests
from config import NAVER_CLIENT_ID, NAVER_CLIENT_SECRET

def test_api(query="남북관계"):
    url = "https://openapi.naver.com/v1/search/news.json"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": 3, "sort": "date"}

    response = requests.get(url, headers=headers, params=params)
    print(f"상태코드: {response.status_code}")

    data = response.json()
    for i, item in enumerate(data.get("items", [])):
        print(f"\n--- 기사 {i+1} ---")
        print(f"제목: {item['title']}")
        print(f"요약: {item['description']}")
        print(f"날짜: {item['pubDate']}")
        print(f"링크: {item['link']}")

test_api()
