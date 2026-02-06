"""High-level experiment runners and reconstruction helpers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

def bit7_baseline_reconstruct(text: str, p: float, rep=3):
    bits = bit7_encode(text)
    tx = rep_encode(bits, rep=rep)
    rx = bsc_flip_bits(tx, p)
    dec = rep_decode(rx, rep=rep)
    return bit7_decode(dec)

def huff_baseline_reconstruct(text: str, p: float, rep=3):
    bits = huff_encode(text)
    tx = rep_encode(bits, rep=rep)
    rx = bsc_flip_bits(tx, p)
    dec = rep_decode(rx, rep=rep)
    return huff_decode(dec)

def ours_reconstruct(idx: int, p: float, rep=3):
    item = abstr_by_idx.get(idx)
    if item is None:
        return ""

    syms = item.get("semantic_symbols_ssc", [])
    if not syms:
        return ""

    final_triples = []
    for s in syms:
        b = s.get("bits")
        if b is None:
            continue
        code = np.array(b, dtype=np.int8)

        tx = rep_encode(code, rep=rep)
        rx = bsc_flip_bits(tx, p)
        dec = rep_decode(rx, rep=rep)
        corr = correction_algo(dec)

        tri_rec = bits2triple.get(tuple(corr.tolist()))
        if tri_rec is not None and len(tri_rec) >= 3:
            final_triples.append([tri_rec[0], tri_rec[1], tri_rec[2]])

    if not final_triples:
        return ""

    return kg2text_generate(final_triples)

def run_one_p(p):
    gold_texts, rec_huff, rec_bit7, rec_ours = [], [], [], []

    for ex in eval_samples:
        idx = to_int_idx(ex.get("idx"))
        gold = ex["text"]
        gold_texts.append(gold)

        rec_huff.append(huff_baseline_reconstruct(gold, p, rep=REP))
        rec_bit7.append(bit7_baseline_reconstruct(gold, p, rep=REP))
        rec_ours.append(ours_reconstruct(idx, p, rep=REP))

    # similarity
    eg = embed_sentences(gold_texts)
    eh = embed_sentences(rec_huff)
    eb = embed_sentences(rec_bit7)
    eo = embed_sentences(rec_ours)

    sim_h = float(np.mean(cosine_sim(eg, eh)))
    sim_b = float(np.mean(cosine_sim(eg, eb)))
    sim_o = float(np.mean(cosine_sim(eg, eo)))

    # BLEU-1..4
    b_h = [corpus_bleu_n(rec_huff, gold_texts, n) for n in [1,2,3,4]]
    b_b = [corpus_bleu_n(rec_bit7, gold_texts, n) for n in [1,2,3,4]]
    b_o = [corpus_bleu_n(rec_ours, gold_texts, n) for n in [1,2,3,4]]

    empty_rate = sum([1 for s in rec_ours if s.strip()==""]) / len(rec_ours)
    return (sim_h, sim_b, sim_o), b_h, b_b, b_o, empty_rate
