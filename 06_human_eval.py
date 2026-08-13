"""
Step 6: 사람 라벨링으로 GPT vs 로지스틱회귀 성능 독립 검증

함수 흐름:
  generate_sheet() : test set에서 40건 층화추출 → 라벨링 시트(정답 안 보임) + 정답지 생성
  score_sheet()     : 사용자가 채운 시트 파싱 → 정답지와 비교 → 3자 비교 결과 산출

용도: 지금까지의 성능 지표(05_ml_classify.py)는 전부 GPT 라벨을 정답으로 가정한
     순환 검증이었음. 사람이 직접 라벨링한 결과를 외부 기준으로 삼아
     GPT와 로지스틱회귀 각각의 실제 정확도를 독립적으로 비교한다.

실행:
  python 06_human_eval.py generate   # 라벨링 시트 생성
  (사람이 data/processed/human_label_sheet.md 를 직접 채움)
  python 06_human_eval.py score      # 채점
"""

import json
import os
import re
import sys

from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

CLASS_ORDER = ["논증형", "진영공격형", "단순언급형"]
SHEET_PATH = "data/processed/human_label_sheet.md"
ANSWERKEY_PATH = "data/processed/human_label_answerkey.json"
RESULTS_PATH = "data/processed/human_eval_results.json"

kiwi = Kiwi()
CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "XR"}  # 일반명사, 고유명사, 동사, 형용사, 어근


def load_dataset(path: str = "data/processed/utterances_all.json") -> tuple[list, list]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_u = [u for r in data for u in r["utterances"]]
    valid = [u for u in all_u if u.get("type") in CLASS_ORDER]
    quotes = [u["quote"] for u in valid]
    labels = [u["type"] for u in valid]
    return quotes, labels


def kiwi_tokenize(text: str) -> list:
    tokens = []
    for token in kiwi.tokenize(text):
        if token.tag in CONTENT_TAGS and len(token.form) > 1:
            form = token.form
            if token.tag in ("VV", "VA"):  # 용언은 원형 어미 '다' 부착해 정규화
                form += "다"
            tokens.append(form)
    return tokens


def train_model():
    """05_ml_classify.py와 동일한 파이프라인·random_state로 재학습 (완전 재현 가능)"""
    quotes, labels = load_dataset()
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        quotes, labels, test_size=0.2, stratify=labels, random_state=42
    )

    vectorizer = TfidfVectorizer(
        tokenizer=kiwi_tokenize, token_pattern=None,
        min_df=2, max_df=0.9, ngram_range=(1, 1), sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test_text, X_test, y_test


CRITERIA_TEXT = """# 사람 라벨링 — 정치 발언 유형 판단

GPT-4o-mini와 로지스틱 회귀 모델이 각 발언을 얼마나 정확히 분류하는지 검증하기 위해,
아래 발언 40개를 직접 읽고 유형을 판단해주세요. **정답 라벨은 보이지 않습니다.**

## 분류 기준

- **논증형**: 구체적 데이터·통계, 역사적 사례, 정책 대안, 논리적 인과관계 중 하나 이상을 갖춘 주장.
  (단순히 "~때문에", "헌법에 위배된다"처럼 근거 없이 한 마디만 붙인 발언은 논증형 아님)
- **진영공격형**: 특정 단어("종북", "위헌" 등) 유무와 무관하게, 상대의 정체성·충성도를 의심하게 만들거나
  적·반국가세력으로 낙인찍거나, 주장 내용이 아닌 발언자의 동기·인격을 공격하는 발언
- **단순언급형**: 논증도 공격도 아닌 사실 전달, 단순 입장 표명, 중립적 언급

## 작성 방법

아래 각 줄의 대괄호 `[ ]` 안에 `논증형` / `진영공격형` / `단순언급형` 중 하나를 그대로 입력해주세요.
예: `1. [논증형] "..."`

## 발언 목록

"""


def generate_sheet(n: int = 40):
    model, X_test_text, X_test, y_test = train_model()
    y_pred = model.predict(X_test)

    idx = list(range(len(X_test_text)))
    sample_idx, _ = train_test_split(idx, train_size=n, stratify=y_test, random_state=42)
    sample_idx.sort()

    os.makedirs("data/processed", exist_ok=True)

    lines = [CRITERIA_TEXT]
    answerkey = []
    for i, sidx in enumerate(sample_idx, start=1):
        quote = X_test_text[sidx]
        lines.append(f'{i}. [ ] "{quote}"\n')
        answerkey.append({
            "num": i,
            "quote": quote,
            "gpt_label": y_test[sidx],
            "logreg_pred": y_pred[sidx],
        })

    with open(SHEET_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(ANSWERKEY_PATH, "w", encoding="utf-8") as f:
        json.dump(answerkey, f, ensure_ascii=False, indent=2)

    print(f"✅ 라벨링 시트 생성: {SHEET_PATH} ({len(sample_idx)}건)")
    print(f"   (정답지는 {ANSWERKEY_PATH}에 따로 저장, 채점 전까지 열어보지 마세요)")
    print(f"\n다 채우신 뒤 실행: python 06_human_eval.py score")


LINE_RE = re.compile(r"^\s*(\d+)\.\s*\[\s*([^\]]*?)\s*\]")


def parse_sheet(path: str = SHEET_PATH) -> dict:
    human_labels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line)
            if not m:
                continue
            num, label = int(m.group(1)), m.group(2).strip()
            if label in CLASS_ORDER:
                human_labels[num] = label
            elif label:
                print(f"⚠️  {num}번 라벨 '{label}'을(를) 인식할 수 없음 (논증형/진영공격형/단순언급형 중 하나로 입력) — 스킵")
    return human_labels


def compare(name_a: str, a: list, name_b: str, b: list) -> dict:
    acc = accuracy_score(a, b)
    cm = confusion_matrix(a, b, labels=CLASS_ORDER)
    print(f"\n[{name_a} vs {name_b}] 일치율: {acc:.3f} ({sum(x == y for x, y in zip(a, b))}/{len(a)})")
    print(f"혼동행렬 (행={name_a}, 열={name_b}, 순서={CLASS_ORDER}):")
    for label, row in zip(CLASS_ORDER, cm):
        print(f"  {label}: {row.tolist()}")
    return {"accuracy": round(float(acc), 4), "confusion_matrix": {"labels": CLASS_ORDER, "matrix": cm.tolist()}}


def score_sheet():
    if not os.path.exists(SHEET_PATH) or not os.path.exists(ANSWERKEY_PATH):
        print(f"❌ 먼저 'python 06_human_eval.py generate'를 실행하세요.")
        return

    human_labels = parse_sheet()
    with open(ANSWERKEY_PATH, "r", encoding="utf-8") as f:
        answerkey = json.load(f)

    missing = [a["num"] for a in answerkey if a["num"] not in human_labels]
    if missing:
        print(f"⚠️  아직 라벨을 채우지 않은 항목: {missing}")

    matched = [a for a in answerkey if a["num"] in human_labels]
    if not matched:
        print("❌ 채점할 항목이 없습니다.")
        return

    human = [human_labels[a["num"]] for a in matched]
    gpt = [a["gpt_label"] for a in matched]
    logreg = [a["logreg_pred"] for a in matched]

    print(f"\n총 {len(matched)}/{len(answerkey)}건 채점")

    results = {
        "n_scored": len(matched),
        "human_vs_gpt": compare("사람", human, "GPT", gpt),
        "human_vs_logreg": compare("사람", human, "로지스틱회귀", logreg),
        "gpt_vs_logreg_subset": compare("GPT", gpt, "로지스틱회귀", logreg),
    }

    os.makedirs("data/processed", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 결과 저장: {RESULTS_PATH}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("generate", "score"):
        print("사용법: python 06_human_eval.py [generate|score]")
        return
    if sys.argv[1] == "generate":
        generate_sheet()
    else:
        score_sheet()


if __name__ == "__main__":
    main()
