"""Lightweight metrics/utilities used for analysis/plots."""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np

def avg_in_bin(key):
    out = []
    for (lo,hi) in bins:
        vals = [r[key] for r in rows if (r["chars"] >= lo and r["chars"] < hi)]
        out.append(float(np.mean(vals)) if len(vals) else np.nan)
    return out

def cumulative_kb(key, n):
    total_bits = sum(r[key] for r in rows[:n])  # 앞에서 n개
    return float(total_bits / 1000.0)  # kb (논문 축 관례)

def embed_sentences(texts):
    return emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def cosine_sim(a, b):
    return np.sum(a*b, axis=1)

def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def corpus_bleu_n(preds, refs, n):
    total_clipped = 0
    total_count = 0
    pred_len = 0
    ref_len = 0

    for ptxt, rtxt in zip(preds, refs):
        ptok = ptxt.strip().split()
        rtok = rtxt.strip().split()
        pred_len += len(ptok)
        ref_len  += len(rtok)

        if len(ptok) < n:
            continue

        p_ng = Counter(ngrams(ptok, n))
        r_ng = Counter(ngrams(rtok, n))
        clipped = sum((p_ng & r_ng).values())
        count = sum(p_ng.values())

        total_clipped += clipped
        total_count += count

    prec = (total_clipped / total_count) if total_count > 0 else 0.0

    if pred_len == 0:
        bp = 0.0
    elif pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (ref_len / max(pred_len, 1)))

    return 100.0 * bp * prec
