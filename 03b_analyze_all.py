"""
Step 3b: 전체 기사(405건) GPT 분석

함수 흐름:
  analyze_all()
    ├─ assembly_members.json 읽기 (299명)
    ├─ system prompt 생성 (03과 동일)
    ├─ articles_with_text_all.json 읽기 (405건)
    ├─ [루프] analyze_article(full_text)
    └─ utterances_all.json 저장 + 이념별 키워드 분포 출력

용도: 그래프 2 (정치인-키워드 이분 네트워크)
     그래프 3 (정치인-정치인 공동등장) 은 articles_with_text_all.json 직접 사용
"""

import json
import os
import time
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs("data/processed", exist_ok=True)


def build_system_prompt(assembly: dict) -> str:
    member_list = ", ".join(
        f"{name}({info['ideology']})"
        for name, info in assembly.items()
    )
    return f"""당신은 한국 정치 담론을 분석하는 연구 보조원입니다.
뉴스 기사 본문을 읽고, 남북관계·대북정책·통일·2국가론과 관련된 정치인·공직자·정당인의 발언을 추출하세요.

## 발언자 이념 판단 기준
아래 목록에 있는 인물은 그 이념을 사용하세요.
{member_list}

목록에 없는 인물은 소속 정당·직책을 보고 직접 판단하세요:
- 국민의힘, 개혁신당, 자유통일당 계열 → 보수
- 더불어민주당, 조국혁신당, 진보당, 정의당 계열 → 진보
- 새미래민주당, 무소속 → 기타
- 판단 어려우면 "기타"

## 발언 분류 기준

### 논증형 (엄격하게 적용)
다음 조건 중 하나 이상을 충족해야만 논증형으로 분류:
- 구체적 데이터·통계·수치를 인용한 주장
- 역사적 사례·선례를 근거로 든 주장
- 구체적 정책 대안이나 실행 방안을 제시한 주장
- 논리적 인과관계를 단계적으로 전개한 주장
※ 단순히 "~이기 때문에", "헌법에 위배된다" 한 마디만 붙인 발언은 논증형이 아님

### 진영공격형 (키워드 무관, 수사적 의도로 판단)
특정 단어("종북", "위헌" 등)가 없어도 아래 의도가 드러나면 진영공격형으로 분류:
- 상대방의 국가 충성도·정체성을 의심하게 만드는 발언
- 상대 진영을 적·반국가세력·위험분자로 프레이밍하는 발언
- 정책 내용이 아닌 발언자의 동기·인격·이념을 공격하는 발언
- 상대의 주장을 논리적으로 반박하지 않고 낙인·딱지 붙이기로 대응하는 발언

### 단순언급형
논증도 공격도 없는 사실 전달, 단순 입장 표명, 중립적 언급

## 출력 형식 (JSON만 반환)
{{
  "utterances": [
    {{
      "politician": "발언자 이름",
      "ideology": "보수 또는 진보 또는 중도 또는 기타",
      "quote": "발언 내용 요약 (1~3문장, 직접 인용 우선)",
      "type": "논증형" 또는 "진영공격형" 또는 "단순언급형",
      "target": "발언 대상 정치인 또는 진영 (논증형·진영공격형 모두, 없으면 null)",
      "keywords": ["핵심 주제어 1~3개. 예: 2국가론, 통일백서, 비핵화, 대북제재, 위헌, 평화공존"]
    }}
  ]
}}

남북·2국가론 관련 발언 없으면 {{"utterances": []}} 반환."""


def analyze_article(full_text: str, system_prompt: str) -> list:
    text = full_text[:6000]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음 기사에서 국회의원의 남북 관련 발언을 추출하세요.\n\n기사 본문:\n{text}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("utterances", [])
    except Exception as e:
        print(f"\nGPT 오류: {e}")
        return []


def analyze_all():
    with open("data/assembly_members.json", "r", encoding="utf-8") as f:
        assembly = json.load(f)

    system_prompt = build_system_prompt(assembly)

    with open("data/raw/articles_with_text_all.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"총 {len(articles)}건 GPT 분석 시작 (리스트 무관, 모든 정치인 추출)\n")

    results = []
    total_utterances = 0

    for article in tqdm(articles, desc="GPT 분석 중"):
        full_text = article.get("full_text", "")
        if not full_text:
            continue

        utterances = analyze_article(full_text, system_prompt)
        valid = [u for u in utterances if u.get("politician")]

        if valid:
            results.append({
                "article_title": article.get("title", ""),
                "article_url": article.get("link", ""),
                "pub_date": article.get("pub_date", ""),
                "keyword": article.get("keyword", ""),
                "utterances": valid,
            })
            total_utterances += len(valid)

        time.sleep(0.5)

    with open("data/processed/utterances_all.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 분석 완료")
    print(f"  - 발언 포함 기사: {len(results)}건")
    print(f"  - 총 발언 수: {total_utterances}개")
    print(f"  - 저장 위치: data/processed/utterances_all.json")

    if total_utterances == 0:
        return

    all_u = [u for r in results for u in r["utterances"]]

    print(f"\n담론 유형 분포:")
    type_counts = {}
    for u in all_u:
        t = u.get("type", "알수없음")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, cnt in sorted(type_counts.items()):
        print(f"  - {t}: {cnt}개 ({cnt/total_utterances*100:.1f}%)")

    print(f"\n키워드 빈도 (상위 15개):")
    kw_counts = defaultdict(int)
    for u in all_u:
        for kw in u.get("keywords", []):
            kw_counts[kw] += 1
    for kw, cnt in sorted(kw_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  - {kw}: {cnt}회")

    print(f"\n이념별 상위 키워드:")
    ideology_kw = {"보수": defaultdict(int), "진보": defaultdict(int), "중도": defaultdict(int)}
    for u in all_u:
        ideology = u.get("ideology") or assembly.get(u.get("politician", ""), {}).get("ideology", "기타")
        if ideology in ideology_kw:
            for kw in u.get("keywords", []):
                ideology_kw[ideology][kw] += 1
    for ideology, kw_dict in ideology_kw.items():
        top = sorted(kw_dict.items(), key=lambda x: -x[1])[:5]
        print(f"  [{ideology}] {', '.join(f'{k}({v})' for k, v in top)}")

    print(f"\n정치인별 발언 수 (상위 10명):")
    pol_counts = defaultdict(int)
    for u in all_u:
        pol_counts[u.get("politician", "")] += 1
    for p, cnt in sorted(pol_counts.items(), key=lambda x: -x[1])[:10]:
        ideology = assembly.get(p, {}).get("ideology", "?")
        print(f"  - {p} ({ideology}): {cnt}개")


if __name__ == "__main__":
    analyze_all()
