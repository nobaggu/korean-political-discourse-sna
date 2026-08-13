"""
Step 5: GPT 라벨(type)을 정답으로 삼아 TF-IDF + 로지스틱 회귀로 담론 유형 재현

함수 흐름:
  main()
    ├─ load_dataset() : utterances_all.json 평탄화 + type 결측 필터링
    ├─ kiwi_tokenize() : Kiwi 형태소 분석기로 명사/동사/형용사 추출 (TfidfVectorizer tokenizer)
    ├─ train_test_split (stratify=y) : test는 처음 분리 후 끝까지 손대지 않음
    ├─ TfidfVectorizer.fit_transform / transform
    ├─ tune_hyperparameter() : train 내부 5-fold CV로 정규화 강도 C 탐색(validation)
    ├─ evaluate() : 최적 C로 재학습한 모델을 test에서 딱 한 번 평가
    ├─ top_words_per_class() : 클래스별 상위 가중치 단어 추출
    └─ error_analysis() : 오분류 사례(quote, 실제/예측 라벨, 확률) 추출·저장

용도: GPT-4o-mini가 매긴 논증형/진영공격형/단순언급형 라벨을 전통적 ML로
     재현 가능한지 검증하고, 어떤 단어가 각 유형을 가리키는지 계수로 해석
     (README의 "보수=헌법/북한, 진보=평화/대북정책" 프레임 발견과 연결)
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

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


def tune_hyperparameter(X_train, y_train) -> tuple:
    """train 안에서 5-fold 교차검증(validation)으로 정규화 강도 C를 탐색.
    test는 이 과정에 전혀 관여하지 않는다."""
    base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(base, {"C": C_GRID}, cv=cv, scoring="f1_macro")
    search.fit(X_train, y_train)

    cv_results = [
        {"C": c, "mean_f1_macro": round(float(m), 4), "std": round(float(s), 4)}
        for c, m, s in zip(C_GRID, search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"])
    ]

    print(f"\n[Validation] 5-fold 교차검증으로 C 탐색 (train {len(y_train)}건 내부, test는 아직 안 봄):")
    for r in cv_results:
        marker = "  ← 선택" if r["C"] == search.best_params_["C"] else ""
        print(f"  C={r['C']:<6} macro-F1={r['mean_f1_macro']:.4f} (±{r['std']:.4f}){marker}")
    print(f"✅ 최적 C = {search.best_params_['C']}")

    return search.best_estimator_, cv_results, search.best_params_["C"]


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
            "quote": text,
            "true": true,
            "pred": pred,
            "proba_true": round(float(p[class_idx[true]]), 3),
            "proba_pred": round(float(p[class_idx[pred]]), 3),
            "tokens": kiwi_tokenize(text),
        })

    print(f"\n오분류 {len(errors)}건 / test {len(X_test_text)}건 ({len(errors)/len(X_test_text)*100:.1f}%)")
    for e in errors:
        print(f"\n  실제={e['true']}({e['proba_true']}) → 예측={e['pred']}({e['proba_pred']})")
        print(f"  \"{e['quote']}\"")
        print(f"  토큰: {e['tokens']}")

    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/ml_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 오분류 사례 저장: data/processed/ml_error_analysis.json")

    return errors


def save_results(eval_result: dict, top_words: dict, cv_results: list, best_C: float,
                  path: str = "data/processed/ml_classification_results.json"):
    os.makedirs("data/processed", exist_ok=True)
    output = {
        "config": {
            "test_size": 0.2,
            "min_df": 2,
            "max_df": 0.9,
            "ngram_range": [1, 1],
            "class_weight": "balanced",
            "random_state": 42,
            "C_grid": C_GRID,
            "best_C": best_C,
        },
        "cv_results": cv_results,
        **eval_result,
        "top_words": top_words,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 결과 저장: {path}")


def main():
    quotes, labels = load_dataset()

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        quotes, labels, test_size=0.2, stratify=labels, random_state=42
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

    model, cv_results, best_C = tune_hyperparameter(X_train, y_train)  # validation: train 내부 5-fold CV로만 C를 고름

    eval_result = evaluate(model, X_test, y_test)  # test는 여기서 딱 한 번만 사용
    top_words = top_words_per_class(model, vectorizer, top_n=20)
    error_analysis(model, vectorizer, X_test_text, X_test, y_test)
    save_results(eval_result, top_words, cv_results, best_C)


if __name__ == "__main__":
    main()
