"""
Step 3: GPT API로 발언 추출 및 담론 유형 분류 (보수+진보 동시등장 64건)

함수 흐름:
  analyze_all()
    ├─ assembly_members.json 읽기 (299명 이름:이념 딕셔너리)
    ├─ system prompt 생성 (분류기준 + 299명 목록 포함)
    ├─ articles_with_text.json 읽기 (64건)
    ├─ [루프] analyze_article(full_text)
    │    ├─ user prompt: 기사 본문만 전달
    │    └─ GPT → 기사에서 정치인 직접 탐지 + 발언 추출
    │         [{politician, quote, type, target, keywords}]
    └─ utterances.json 저장

변경사항 (이전 버전 대비):
  - 27명 사전정의 제거 → 299명 국회의원 전체 목록 기반으로 GPT가 직접 탐지
  - target: 논증형(반박대상) + 진영공격형(공격대상) 모두 추출
  - keywords: 발언에서 다루는 남북 관련 핵심 주제어 추출
"""

import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs("data/processed", exist_ok=True)


def build_system_prompt(assembly: dict) -> str:
    """299명 국회의원 목록을 포함한 시스템 프롬프트 생성"""
    # "이름(이념)" 형태로 압축
    member_list = ", ".join(
        f"{name}({info['ideology']})"
        for name, info in assembly.items()
    )

    return f"""당신은 한국 정치 담론을 분석하는 연구 보조원입니다.
뉴스 기사 본문을 읽고, 남북관계·대북정책·통일과 관련된 정치인 발언을 추출하세요.

## 국회의원 목록 (이름(이념) 형식, 이 목록에 있는 사람만 추출)
{member_list}

## 발언 분류 기준
- 논증형: 근거·데이터·정책 대안을 제시하며 주장하는 발언
  (예: "북한 비핵화 없이 제재 완화는 불가능하다", "개성공단 재개로 경협을 복원해야 한다")

- 진영공격형: 상대 진영을 비난하거나 이념적으로 의심하게 만드는 발언.
  직접 비난어 + 상대를 안보위협·반국가세력으로 프레임화하는 수사적 표현 포함.
  (예: "종북", "전쟁광", "빨갱이", "반국가세력",
       "북한이 주적이냐고 묻는 것", "저쪽은 북한 편", "안보를 팔아넘기는 세력")

- 단순언급형: 사실 전달, 단순 입장 표명, 위 두 유형에 해당하지 않는 발언

※ 판단 애매할 때: 수사적 목적이 상대를 이념적으로 의심하게 만드는 것이면 진영공격형

## 출력 형식 (JSON만 반환)
{{
  "utterances": [
    {{
      "politician": "이름 (위 목록에 있는 사람만)",
      "ideology": "보수 또는 진보 또는 중도 (위 목록 기준)",
      "quote": "발언 내용 요약 (1~3문장, 직접 인용 우선)",
      "type": "논증형" 또는 "진영공격형" 또는 "단순언급형",
      "target": "발언 대상 정치인 또는 진영 (논증형·진영공격형 모두 작성, 없으면 null)",
      "keywords": ["남북 관련 핵심 주제어 1~3개. 예: 비핵화, 대북제재, 종전선언, 개성공단, 주적, 2국가론, 북핵"]
    }}
  ]
}}

남북 관련 발언이 없으면 {{"utterances": []}} 반환.
같은 정치인이 여러 발언을 했으면 각각 별도 항목으로 작성."""


def analyze_article(full_text: str, system_prompt: str) -> list:
    """기사 본문 하나를 GPT로 분석 → 발언 목록 반환"""
    text = full_text[:3000]

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
    # 299명 국회의원 목록 로드
    with open("data/assembly_members.json", "r", encoding="utf-8") as f:
        assembly = json.load(f)

    system_prompt = build_system_prompt(assembly)

    with open("data/raw/articles_with_text.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"총 {len(articles)}건 GPT 분석 시작 (299명 기준)\n")

    results = []
    total_utterances = 0

    for article in tqdm(articles, desc="GPT 분석 중"):
        full_text = article.get("full_text", "")
        if not full_text:
            continue

        utterances = analyze_article(full_text, system_prompt)

        # 목록에 있는 정치인만 유효
        valid = [u for u in utterances if u.get("politician") in assembly]

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

    with open("data/processed/utterances.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 분석 완료")
    print(f"  - 발언 포함 기사: {len(results)}건")
    print(f"  - 총 발언 수: {total_utterances}개")
    print(f"  - 저장 위치: data/processed/utterances.json")

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

    print(f"\n정치인별 발언 수 (상위 10명):")
    pol_counts = {}
    for u in all_u:
        p = u.get("politician", "")
        pol_counts[p] = pol_counts.get(p, 0) + 1
    for p, cnt in sorted(pol_counts.items(), key=lambda x: -x[1])[:10]:
        ideology = assembly.get(p, {}).get("ideology", "?")
        print(f"  - {p} ({ideology}): {cnt}개")


if __name__ == "__main__":
    analyze_all()
