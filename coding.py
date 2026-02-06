"""Bit accounting + (baseline / Huffman / proposed) encoders used in the notebook."""

from __future__ import annotations

import math
import heapq
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

def int_to_bits(x, width):
    return [(x >> k) & 1 for k in range(width-1, -1, -1)]

def triple_to_bits(tri):
    h,r,t = tri
    hid = ent2id[_strip_uri_datatype(h)]
    rid = rel2id[str(r)]
    tid = ent2id[_strip_uri_datatype(t)]
    bits = int_to_bits(hid, B_ent) + int_to_bits(rid, B_rel) + int_to_bits(tid, B_ent)
    return bits

class HuffNode:
    __slots__ = ("ch","freq","left","right")
    def __init__(self, ch=None, freq=0, left=None, right=None):
        self.ch = ch
        self.freq = freq
        self.left = left
        self.right = right

def build_huffman_code_lengths(char_freq: dict):
    import heapq
    heap = []
    uid = 0
    for ch, fr in char_freq.items():
        heapq.heappush(heap, (fr, uid, HuffNode(ch=ch, freq=fr)))
        uid += 1

    if len(heap) == 1:
        fr, _, node = heap[0]
        return {node.ch: 1}

    while len(heap) > 1:
        fr1, _, n1 = heapq.heappop(heap)
        fr2, _, n2 = heapq.heappop(heap)
        merged = HuffNode(ch=None, freq=fr1+fr2, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, uid, merged))
        uid += 1

    root = heap[0][2]
    lengths = {}

    def dfs(node, depth):
        if node.ch is not None:
            lengths[node.ch] = max(depth, 1)
            return
        dfs(node.left, depth+1)
        dfs(node.right, depth+1)

    dfs(root, 0)
    return lengths

def huffman_bits(text: str) -> int:
    return int(sum(huff_len.get(ch, 8) for ch in text))  # unseen 대비 8bit fallback

def bit7_bits(text: str) -> int:
    if USE_UNICODE_BYTES_FOR_BIT7:
        return int(8 * len(text.encode("utf-8")))
    else:
        return int(7 * len(text))

def ours_bits_from_abstr(idx: int) -> int:
    item = abstr_by_idx.get(idx)
    if item is None:
        return 0
    # ✅ 우리가 보내는 건 semantic_symbols_ssc의 개수 * L
    n_tr = len(item.get("semantic_symbols_ssc", []))
    return int(n_tr * L)

def to_int_idx(x):
    try: return int(x)
    except: return x

def rep_encode(bits: np.ndarray, rep=3):
    return np.repeat(bits, rep).astype(np.int8)

def rep_decode(bits: np.ndarray, rep=3):
    n = (bits.size // rep) * rep
    bits = bits[:n].reshape(-1, rep)
    return (np.sum(bits, axis=1) >= (rep/2)).astype(np.int8)

def bit7_encode(text: str) -> np.ndarray:
    arr=[]
    for ch in text:
        v = ord(ch) & 0x7F
        arr.extend([(v >> k) & 1 for k in range(6,-1,-1)])
    return np.array(arr, dtype=np.int8)

def bit7_decode(bits: np.ndarray) -> str:
    n = (bits.size // 7) * 7
    bits = bits[:n].reshape(-1, 7)
    chars=[]
    for b in bits:
        v=0
        for i in range(7):
            v = (v << 1) | int(b[i])
        chars.append(chr(v))
    return "".join(chars)

def build_huffman_codebook(text_corpus: str):
    import heapq
    freq = Counter(text_corpus)
    heap=[]
    uid=0
    for ch,fr in freq.items():
        heapq.heappush(heap,(fr,uid,(ch,None,None)))
        uid+=1
    if len(heap)==1:
        ch = heap[0][2][0]
        return {ch:"0"}
    while len(heap)>1:
        fr1,_,n1=heapq.heappop(heap)
        fr2,_,n2=heapq.heappop(heap)
        heapq.heappush(heap,(fr1+fr2,uid,(None,n1,n2))); uid+=1
    root=heap[0][2]
    codes={}
    def dfs(node,prefix):
        ch,left,right = node
        if ch is not None:
            codes[ch]=prefix or "0"
            return
        dfs(left,prefix+"0")
        dfs(right,prefix+"1")
    dfs(root,"")
    return codes

class HuffTrie:
    def __init__(self): self.root={}
    def add(self,ch,code):
        cur=self.root
        for c in code:
            cur=cur.setdefault(c,{})
        cur["$"]=ch
    def decode(self,bits):
        cur=self.root
        out=[]
        for b in bits:
            cur=cur.get("1" if b else "0", {})
            if "$" in cur:
                out.append(cur["$"])
                cur=self.root
        return "".join(out)

def huff_encode(text: str) -> np.ndarray:
    bits=[]
    for ch in text:
        code = huff_code.get(ch)
        if code is None:
            code = format(ord(ch)&0xFF, "08b")
        bits.extend([1 if c=="1" else 0 for c in code])
    return np.array(bits, dtype=np.int8)

def huff_decode(bits: np.ndarray) -> str:
    return trie.decode(bits.tolist())

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
