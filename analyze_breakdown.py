import json
from collections import defaultdict

with open("data/processed/utterances_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)
with open("data/assembly_members.json", "r", encoding="utf-8") as f:
    assembly = json.load(f)

all_u = [u for r in data for u in r["utterances"]]

# 이념별 유형
ideo_type = defaultdict(lambda: defaultdict(int))
for u in all_u:
    ideo_type[u.get("ideology","기타")][u.get("type","?")] += 1

print("=== 이념별 담론 유형 ===")
for ideo in ["보수","진보","기타"]:
    d = ideo_type[ideo]
    sub = sum(d.values())
    if sub:
        print(f"[{ideo}] 총 {sub}개")
        for t in ["논증형","진영공격형","단순언급형"]:
            c = d.get(t,0)
            print(f"  {t}: {c} ({c/sub*100:.1f}%)")

# 진영간 교차
print("\n=== 진영간 교차 발언 ===")
cross = defaultdict(lambda: defaultdict(int))
for r in data:
    for u in r["utterances"]:
        src = u.get("ideology","기타")
        tgt_name = u.get("target")
        if not tgt_name: continue
        tgt_ideo = assembly.get(tgt_name, {}).get("ideology","기타")
        if src != tgt_ideo:
            cross[f"{src}→{tgt_ideo}"][u.get("type","?")] += 1

for direction, types in sorted(cross.items()):
    print(f"{direction}: 총 {sum(types.values())}개")
    for t,c in types.items():
        print(f"  {t}: {c}")

# 임종석 타겟
print("\n=== 임종석 타겟 발언 ===")
jong = [u for r in data for u in r["utterances"] if u.get("target")=="임종석"]
jt = defaultdict(int)
ji = defaultdict(int)
for u in jong:
    jt[u.get("type","?")] += 1
    ji[u.get("ideology","?")] += 1
print(f"총 {len(jong)}개, 유형: {dict(jt)}, 이념: {dict(ji)}")