# Cognitive / Semantic Communication (KG-driven) — Repro Code

Minimal, paper-reproduction-oriented codebase split from a single Colab notebook.

## Quickstart
```bash
pip install -r requirements.txt
python scripts/reproduce_colab.py --base_dir ./runs/webnlg_run
```

## What this repo contains
- PyTorch implementation of Cognitive Semantic Communication Systems Driven by Knowledge Graph
- This implementation is based on the paper [Cognitive Semantic Communication Systems Driven by Knowledge Graph: Principle, Implementation, and Performance Evaluation](https://ieeexplore.ieee.org/document/10262128)

## Architecture
![Architecture](architecture.jpg)

- 
- WebNLG download + preprocessing / alignment
- KG→Text helpers (T5-based)
- Bit accounting + baselines (bit7 / Huffman) + proposed representation coding
- BSC channel + simple correction utilities
- Small analysis helpers (bins / cumulative plots)
