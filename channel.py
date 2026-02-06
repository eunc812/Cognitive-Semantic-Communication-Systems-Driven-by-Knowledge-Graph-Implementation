"""Channel models and simple error-correction helpers."""

from __future__ import annotations

import random
from typing import List

def bsc(bits: np.ndarray, p: float) -> np.ndarray:
    flip = (np.random.rand(bits.size) < p).astype(np.int8)
    return (bits ^ flip).astype(np.int8)

def correction(o_bits: np.ndarray) -> dict:
    """
    Table II의 'most similar semantic symbol'을 구현.
    - 논문 문맥상: Hamming distance 최소가 가장 유사.
    """
    # brute-force Hamming to all KG codes
    dists = np.sum(KG_BITS != o_bits[None, :], axis=1)
    best = int(np.argmin(dists))
    best_bits = tuple(KG_BITS[best].tolist())
    return bits2meta[best_bits]

def bsc_flip_bits(bits: np.ndarray, p: float) -> np.ndarray:
    flip = (np.random.rand(bits.size) < p).astype(np.int8)
    return (bits ^ flip).astype(np.int8)

def correction_algo(o: np.ndarray) -> np.ndarray:
    dists = np.sum(KG_CODES != o[None, :], axis=1)
    return KG_CODES[int(np.argmin(dists))]
