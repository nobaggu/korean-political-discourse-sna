"""
Step 4: 세 가지 네트워크 시각화

그래프 1 — 정치인→정치인 방향 발언 네트워크 (utterances.json, 64건)
  - 노드: 정치인 (크기=발언수, 색=이념)
  - 엣지: target 있는 논증형·진영공격형 발언만, 방향 화살표
          파랑=논증형, 빨강=진영공격형
  - output/graph1_directed.html

그래프 2 — 정치인-키워드 이분 네트워크 (utterances_all.json, 405건)
  - 노드: 정치인(색=이념) + 키워드(색=회색, 크기=언급빈도)
  - 엣지: 정치인이 해당 키워드를 다룬 경우 (색=이념색)
  - output/graph2_bipartite.html

그래프 3 — 정치인-정치인 공동등장 네트워크 (articles_with_text_all.json, 405건)
  - 같은 기사에 2명 이상 등장한 정치인끼리 비방향 연결
  - 엣지 굵기 = 공동등장 기사 수 (co-occurrence 빈도)
  - 비교 목적: "같이 언급만 된 경우" vs "실제 발언 교환(그래프1)"
  - output/graph3_cooccurrence.html
"""

import json
from collections import defaultdict
import networkx as nx
from pyvis.network import Network

# ── 이념별 색상 ──────────────────────────────────────────────
IDEOLOGY_COLOR = {
    "보수": "#E74C3C",   # 빨강
    "진보": "#3498DB",   # 파랑
    "중도": "#2ECC71",   # 초록
}


def load_assembly() -> dict:
    with open("data/assembly_members.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_ideology(name: str, assembly: dict) -> str:
    return assembly.get(name, {}).get("ideology", "기타")


# ══════════════════════════════════════════════════════════════
# 그래프 1: 정치인→정치인 방향 발언 네트워크
# ══════════════════════════════════════════════════════════════

def build_graph1(assembly: dict):
    with open("data/processed/utterances_all.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    all_utterances = [u for a in articles for u in a["utterances"]]

    # 노드 집계
    node_count = defaultdict(int)
    for u in all_utterances:
        node_count[u["politician"]] += 1

    # 엣지 집계 (self-loop 제거, target이 발언자 자신이면 건너뜀)
    edge_data = defaultdict(lambda: defaultdict(int))
    for u in all_utterances:
        speaker = u["politician"]
        target  = u.get("target")
        utype   = u.get("type", "")

        if not target or utype == "단순언급형":
            continue
        if target == speaker:          # self-loop 제거
            continue
        if target not in assembly:     # 299명 외 이름 제거
            continue

        edge_data[(speaker, target)][utype] += 1

    # NetworkX 방향 그래프
    G = nx.DiGraph()
    for politician, count in node_count.items():
        G.add_node(politician, ideology=get_ideology(politician, assembly), count=count)

    for (speaker, target), types in edge_data.items():
        total = sum(types.values())
        if types.get("진영공격형", 0) >= types.get("논증형", 0):
            color, label = "#E74C3C", "공격형"
        else:
            color, label = "#3498DB", "논증형"
        G.add_edge(speaker, target,
                   weight=total, color=color, label=label,
                   논증형=types.get("논증형", 0),
                   진영공격형=types.get("진영공격형", 0))

    # 통계
    print("=== 그래프 1: 정치인→정치인 방향 발언 네트워크 ===")
    print(f"노드: {G.number_of_nodes()}명  엣지: {G.number_of_edges()}개")
    print("\n[방향 엣지 목록]")
    for s, t, d in G.edges(data=True):
        print(f"  {s}({get_ideology(s, assembly)}) → {t}({get_ideology(t, assembly)})"
              f"  [{d['label']}]  논증:{d['논증형']} 공격:{d['진영공격형']}")

    # Pyvis
    net = Network(height="750px", width="100%",
                  bgcolor="#1a1a2e", font_color="white", directed=True)
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200)

    max_count = max(node_count.values()) if node_count else 1
    for p, data in G.nodes(data=True):
        ideology = data.get("ideology", "기타")
        count    = data.get("count", 1)
        net.add_node(p,
                     label=p,
                     title=f"{p} ({ideology})\n발언 {count}개",
                     color=IDEOLOGY_COLOR.get(ideology, "#95A5A6"),
                     size=15 + (count / max_count) * 50)

    for s, t, d in G.edges(data=True):
        net.add_edge(s, t,
                     width=1 + d["weight"],
                     color=d["color"],
                     title=f"{s} → {t}\n{d['label']}\n논증형:{d['논증형']} 공격형:{d['진영공격형']}",
                     arrows="to")

    net.save_graph("output/graph1_directed.html")
    print("\n✅ 저장: output/graph1_directed.html\n")


# ══════════════════════════════════════════════════════════════
# 그래프 2: 정치인-키워드 이분 네트워크
# ══════════════════════════════════════════════════════════════

def build_graph2(assembly: dict, top_n_keywords: int = 20):
    with open("data/processed/utterances_all.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    all_utterances = [u for a in articles for u in a["utterances"]]

    # 키워드 빈도
    keyword_total = defaultdict(int)
    for u in all_utterances:
        for kw in u.get("keywords", []):
            keyword_total[kw] += 1

    top_keywords = set(
        kw for kw, _ in
        sorted(keyword_total.items(), key=lambda x: -x[1])[:top_n_keywords]
    )

    # 엣지 집계
    edge_count    = defaultdict(int)
    node_pol_count = defaultdict(int)
    for u in all_utterances:
        p = u["politician"]
        node_pol_count[p] += 1
        for kw in u.get("keywords", []):
            if kw in top_keywords:
                edge_count[(p, kw)] += 1

    # Pyvis
    net = Network(height="800px", width="100%",
                  bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-3000, central_gravity=0.1, spring_length=250)

    added_politicians, added_keywords = set(), set()

    for (p, kw), cnt in edge_count.items():
        if p not in added_politicians:
            ideology = get_ideology(p, assembly)
            count    = node_pol_count[p]
            net.add_node(p,
                         label=p,
                         title=f"{p} ({ideology})\n발언 {count}개",
                         color=IDEOLOGY_COLOR.get(ideology, "#95A5A6"),
                         size=15 + min(count, 60),
                         shape="dot")
            added_politicians.add(p)

        if kw not in added_keywords:
            freq = keyword_total[kw]
            net.add_node(kw,
                         label=kw,
                         title=f"키워드: {kw}\n총 {freq}회 언급",
                         color="#BDC3C7",
                         size=10 + min(freq // 3, 40),
                         shape="diamond")
            added_keywords.add(kw)

        ideology = get_ideology(p, assembly)
        net.add_edge(p, kw,
                     width=0.5 + cnt * 0.5,
                     color=IDEOLOGY_COLOR.get(ideology, "#95A5A6"),
                     title=f"{p} → {kw}: {cnt}회")

    net.save_graph("output/graph2_bipartite.html")

    print("=== 그래프 2: 정치인-키워드 이분 네트워크 ===")
    print(f"정치인 노드: {len(added_politicians)}명")
    print(f"키워드 노드: {len(added_keywords)}개 (상위 {top_n_keywords}개)")
    print(f"엣지: {len(edge_count)}개")

    ideology_kw = {"보수": defaultdict(int), "진보": defaultdict(int), "중도": defaultdict(int)}
    for (p, kw), cnt in edge_count.items():
        ideology = get_ideology(p, assembly)
        if ideology in ideology_kw:
            ideology_kw[ideology][kw] += cnt
    print("\n이념별 상위 키워드:")
    for ideology, kw_counts in ideology_kw.items():
        top = sorted(kw_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  [{ideology}] {', '.join(f'{k}({v})' for k, v in top)}")

    print("\n✅ 저장: output/graph2_bipartite.html\n")


# ══════════════════════════════════════════════════════════════
# 그래프 3: 정치인-정치인 공동등장 (co-occurrence)
# ══════════════════════════════════════════════════════════════

def build_graph3(assembly: dict):
    """
    405건 전체 기사에서 같은 기사에 함께 등장한 정치인 쌍을 엣지로 연결.
    - full_text를 299명 목록으로 직접 스캔 (politicians_found는 구 27명 기준이라 재스캔)
    - 엣지 굵기 = 공동등장 기사 수
    - 이 그래프와 그래프 1을 비교하면:
        "같이 언급만 됐는가" vs "실제 발언 교환이 있었는가" 차이가 드러남
    """
    with open("data/raw/articles_with_text_all.json", "r", encoding="utf-8") as f:
        articles = json.load(f)

    assembly_names = list(assembly.keys())  # 299명 이름 목록

    # 공동등장 엣지 집계 (무방향: 이름 정렬하여 중복 방지)
    cooccurrence = defaultdict(int)
    node_appear  = defaultdict(int)  # 각 정치인 등장 기사 수

    for article in articles:
        full_text = article.get("full_text", "")
        if not full_text:
            continue

        found = [name for name in assembly_names if name in full_text]
        if len(found) < 2:
            continue

        for name in found:
            node_appear[name] += 1

        for i in range(len(found)):
            for j in range(i + 1, len(found)):
                pair = tuple(sorted([found[i], found[j]]))
                cooccurrence[pair] += 1

    # 엣지 너무 많으면 min_cooc 이상만 사용 (기본 2회 이상 공동등장)
    min_cooc = 2
    filtered = {pair: cnt for pair, cnt in cooccurrence.items() if cnt >= min_cooc}

    # 엣지에 참여한 정치인만 노드로
    involved = set(n for pair in filtered for n in pair)

    print("=== 그래프 3: 정치인-정치인 공동등장 네트워크 ===")
    print(f"노드: {len(involved)}명  엣지: {len(filtered)}개 (공동등장 {min_cooc}회 이상)")
    print("\n[공동등장 상위 15 쌍]")
    for (a, b), cnt in sorted(filtered.items(), key=lambda x: -x[1])[:15]:
        ai, bi = get_ideology(a, assembly), get_ideology(b, assembly)
        print(f"  {a}({ai}) ↔ {b}({bi}): {cnt}건")

    # Pyvis
    net = Network(height="800px", width="100%",
                  bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut(gravity=-4000, central_gravity=0.2, spring_length=200)

    max_appear = max(node_appear.values()) if node_appear else 1

    for name in involved:
        ideology = get_ideology(name, assembly)
        appear   = node_appear.get(name, 1)
        net.add_node(name,
                     label=name,
                     title=f"{name} ({ideology})\n{appear}건 기사 등장",
                     color=IDEOLOGY_COLOR.get(ideology, "#95A5A6"),
                     size=10 + (appear / max_appear) * 50)

    max_cooc = max(filtered.values()) if filtered else 1
    for (a, b), cnt in filtered.items():
        net.add_edge(a, b,
                     width=1 + (cnt / max_cooc) * 8,
                     title=f"{a} ↔ {b}: {cnt}건 기사에 공동등장",
                     color="#7F8C8D")

    net.save_graph("output/graph3_cooccurrence.html")
    print("\n✅ 저장: output/graph3_cooccurrence.html\n")


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("output", exist_ok=True)

    assembly = load_assembly()
    print(f"국회의원 {len(assembly)}명 로드 완료\n")

    build_graph1(assembly)
    build_graph2(assembly)
    build_graph3(assembly)
