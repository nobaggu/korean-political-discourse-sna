"""
Step 5b: quote에 target(발언 대상)을 합쳐 TF-IDF + 로지스틱 회귀 재학습

함수 흐름:
  main()
    ├─ load_dataset() : quote 앞에 "[{target}에 대해]"를 붙여 피처 재구성
    ├─ kiwi_tokenize() : Kiwi 형태소 분석기로 명사/동사/형용사 추출 (TfidfVectorizer tokenizer)
    ├─ train_test_split (stratify=y, 05_ml_classify.py와 동일 random_state로 표본 구성 일치)
    ├─ TfidfVectorizer.fit_transform / transform
    ├─ LogisticRegression 학습
    ├─ evaluate() : accuracy, classification_report, confusion matrix
    ├─ top_words_per_class() : 클래스별 상위 가중치 단어 추출
    └─ error_analysis() : 오분류 사례 추출·저장

용도: 05_ml_classify.py는 quote 텍스트만 피처로 썼는데, quote 자체에는
     "누구를 향한 발언인지"가 전혀 안 담겨 있고 target 필드에 별도 저장돼 있음
     (예: quote="통일을 부정하는 통일백서는 명백한 헌법 위반", target="이재명").
     target을 quote에 합쳐 넣으면 성능이 개선되는지 05번 결과(정확도 62.1%)와 비교 검증.
"""

import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CLASS_ORDER = ["논증형", "진영공격형", "단순언급형"]

kiwi = Kiwi()
CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "XR"}  # 일반명사, 고유명사, 동사, 형용사, 어근


def load_dataset(path: str = "data/processed/utterances_all.json") -> tuple[list, list]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_u = [u for r in data for u in r["utterances"]]
    valid = [u for u in all_u if u.get("type") in CLASS_ORDER]

    skipped = len(all_u) - len(valid)
    print(f"전체 발언 {len(all_u)}건 중 결측/파싱오류 {skipped}건 제외 → 학습 사용 {len(valid)}건")

    with_target = sum(1 for u in valid if u.get("target"))
    print(f"target 있는 발언: {with_target}건 / target 없는 발언: {len(valid) - with_target}건")

    texts = []
    for u in valid:
        target = u.get("target")
        if target:
            texts.append(f"[{target}에 대해] {u['quote']}")
        else:
            texts.append(u["quote"])

    labels = [u["type"] for u in valid]
    return texts, labels


def kiwi_tokenize(text: str) -> list:
    tokens = []
    for token in kiwi.tokenize(text):
        if token.tag in CONTENT_TAGS and len(token.form) > 1:
            form = token.form
            if token.tag in ("VV", "VA"):  # 용언은 원형 어미 '다' 부착해 정규화
                form += "다"
            tokens.append(form)
    return tokens


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    print(f"\n✅ 정확도: {acc:.3f}")
    print(f"\n분류 리포트:")
    print(classification_report(y_test, y_pred, labels=CLASS_ORDER, zero_division=0))

    print(f"혼동 행렬 (행=실제, 열=예측, 순서: {CLASS_ORDER}):")
    for label, row in zip(CLASS_ORDER, cm):
        print(f"  {label}: {row.tolist()}")

    return {
        "accuracy": round(float(acc), 4),
        "classification_report": report,
        "confusion_matrix": {"labels": CLASS_ORDER, "matrix": cm.tolist()},
    }


def top_words_per_class(model, vectorizer, top_n: int = 20) -> dict:
    feature_names = vectorizer.get_feature_names_out()
    by_class = {}
    for i, cls in enumerate(model.classes_):
        coefs = model.coef_[i]
        top_idx = np.argsort(coefs)[::-1][:top_n]
        by_class[cls] = [(feature_names[j], round(float(coefs[j]), 3)) for j in top_idx]

    result = {cls: by_class[cls] for cls in CLASS_ORDER}

    print(f"\n클래스별 상위 가중치 단어 (top {top_n}):")
    for cls in CLASS_ORDER:
        print(f"\n  [{cls}]")
        for word, coef in result[cls][:10]:
            print(f"    {word}: {coef:+.3f}")

    return result


def error_analysis(model, vectorizer, X_test_text: list, X_test, y_test: list) -> list:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    class_idx = {cls: i for i, cls in enumerate(model.classes_)}

    errors = []
    for text, true, pred, p in zip(X_test_text, y_test, y_pred, proba):
        if true == pred:
            continue
        errors.append({
            "text": text,
            "true": true,
            "pred": pred,
            "proba_true": round(float(p[class_idx[true]]), 3),
            "proba_pred": round(float(p[class_idx[pred]]), 3),
        })

    print(f"\n오분류 {len(errors)}건 / test {len(X_test_text)}건 ({len(errors)/len(X_test_text)*100:.1f}%)")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/ml_error_analysis_with_target.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"✅ 오분류 사례 저장: data/processed/ml_error_analysis_with_target.json")

    return errors


def save_results(eval_result: dict, top_words: dict, path: str = "data/processed/ml_classification_results_with_target.json"):
    os.makedirs("data/processed", exist_ok=True)
    output = {
        "config": {
            "feature": "target + quote",
            "test_size": 0.2,
            "min_df": 2,
            "max_df": 0.9,
            "ngram_range": [1, 1],
            "class_weight": "balanced",
            "random_state": 42,
        },
        **eval_result,
        "top_words": top_words,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 결과 저장: {path}")


def main():
    texts, labels = load_dataset()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"train {len(X_train_text)}건 / test {len(X_test_text)}건")

    vectorizer = TfidfVectorizer(
        tokenizer=kiwi_tokenize,
        token_pattern=None,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 1),
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)  # fit은 train에만! test는 transform만 (데이터 누수 방지)
    print(f"TF-IDF 어휘 크기: {len(vectorizer.get_feature_names_out())}개")

    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",  # 클래스 불균형(진영공격형 19.3%) 보정
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train, y_train)

    eval_result = evaluate(model, X_test, y_test)
    top_words = top_words_per_class(model, vectorizer, top_n=20)
    error_analysis(model, vectorizer, X_test_text, X_test, y_test)
    save_results(eval_result, top_words)


if __name__ == "__main__":
    main()
