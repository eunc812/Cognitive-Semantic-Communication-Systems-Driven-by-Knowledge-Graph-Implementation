"""Preprocessing utilities (WebNLG parsing, normalization, triple ↔ id helpers)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

def pick_text(row):
    # row["lex"]["text"]는 list 형태
    lex = row.get("lex", {})
    texts = lex.get("text", []) if isinstance(lex, dict) else []
    if texts and isinstance(texts, list):
        return texts[0]
    return ""

def pick_triples(row):
    # row["modified_triple_sets"]가 list[dict{subject,property,object}]
    triples = row.get("modified_triple_sets", [])
    out=[]
    for tri in triples:
        if isinstance(tri, dict):
            h = tri.get("subject"); r = tri.get("property"); t = tri.get("object")
            if h is not None and r is not None and t is not None:
                out.append([h, r, t])
        elif isinstance(tri, (list, tuple)) and len(tri) >= 3:
            out.append([tri[0], tri[1], tri[2]])
    return out

def _strip_uri_datatype(x: str) -> str:
    x = str(x).strip()
    if "^^<" in x:
        x = x.split("^^<")[0].strip()
    return x

def norm_entity(x: str) -> str:
    x = _strip_uri_datatype(x)
    x = x.replace("_"," ").replace('"',"").strip()
    x = re.sub(r"^<http[^>]+/([^/>]+)>$", r"\1", x)
    x = re.sub(r"\s+"," ", x)
    return x.lower()

def align_triples_by_contains(text: str, triples):
    s = text.lower()
    out=[]
    for h,r,t in triples:
        hh = norm_entity(h); tt = norm_entity(t)
        # 논문 Table I 조건: sentence contains(h) and contains(t)
        if hh and tt and (hh in s) and (tt in s):
            out.append([h,r,t])
    return out

def norm(x: str) -> str:
    x = strip_datatype(x)
    x = x.replace("_", " ").replace('"', " ").strip()
    x = re.sub(r"^<http[^>]+/([^/>]+)>$", r"\1", x)
    x = re.sub(r"\s+", " ", x)
    return x.lower()

def find_candidate_heads_in_text(s_norm: str, heads: list):
    # 간단 버전: head 문자열이 text에 포함되는지 체크
    # (더 빠르게 하려면 Aho-corasick 같은 걸 쓰지만, 우선은 이걸로 충분)
    cand = []
    for h in heads:
        if h and h in s_norm:
            cand.append(h)
    return cand

def strip_datatype(x: str) -> str:
    x = str(x).strip()
    if "^^<" in x:
        x = x.split("^^<")[0].strip()
    return x

def norm_key(x: str) -> str:
    x = strip_datatype(x)
    x = x.replace("_"," ").replace('"'," ").strip()
    x = re.sub(r"^<http[^>]+/([^/>]+)>$", r"\1", x)
    x = re.sub(r"\s+"," ", x)
    return x.lower()

def norm_triple_key(tri):
    h,r,t = tri
    return (norm_key(h), norm_key(r), norm_key(t))

def triple_to_ids(tri):
    h,r,t = tri
    h = strip_datatype(h); t = strip_datatype(t)
    r = str(r)
    return int(entity2id[h]), int(rel2id[r]), int(entity2id[t])

def clean_obj(x: str) -> str:
    x = str(x).replace("_", " ").replace('"', "").strip()
    if "^^<" in x:
        x = x.split("^^<")[0].strip()
    x = re.sub(r"\s+"," ", x)
    return x

def linearize_paper_style(triples):
    trs=[]
    for tri in triples:
        if isinstance(tri, dict) and "triple" in tri:
            tri = tri["triple"]
        if not isinstance(tri, (list, tuple)) or len(tri) < 3:
            continue
        h,r,t = tri[0], tri[1], tri[2]
        trs.append((clean_obj(h), clean_obj(r), clean_obj(t)))

    if not trs:
        return ""
    head_counts = Counter([h for h,_,_ in trs])
    main_head = head_counts.most_common(1)[0][0]
    main_pairs = [(r,t) for (h,r,t) in trs if h == main_head]
    if not main_pairs:
        main_head = trs[0][0]
        main_pairs = [(trs[0][1], trs[0][2])]
    rt = ", ".join([f"{r} {t}" for (r,t) in main_pairs])
    return f"kg2text: {main_head} {rt}"
