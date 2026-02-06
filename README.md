# Cognitive / Semantic Communication (KG-driven) — Repro Code

Minimal, paper-reproduction-oriented codebase split from a single Colab notebook.

## Quickstart
```bash
pip install -r requirements.txt
python scripts/reproduce_colab.py --base_dir ./runs/webnlg_run
```

## What this repo contains
- WebNLG download + preprocessing / alignment
- KG→Text helpers (T5-based)
- Bit accounting + baselines (bit7 / Huffman) + proposed representation coding
- BSC channel + simple correction utilities
- Small analysis helpers (bins / cumulative plots)

## Code map
- `src/kg_semcom/preprocess.py` : WebNLG parsing, normalization, triple/id helpers
- `src/kg_semcom/kg2text.py`    : KG→Text dataset + generation helpers
- `src/kg_semcom/channel.py`    : BSC + correction helpers
- `src/kg_semcom/coding.py`     : baseline + Huffman + proposed bit accounting
- `src/kg_semcom/metrics.py`    : small metrics utilities
- `scripts/reproduce_colab.py`  : faithful linear reproduction of the original notebook

## Notes
- This repo keeps the original notebook in `notebooks/original.ipynb` for reference.
- Large artifacts (datasets/results) are intentionally not committed.

## License
Add a LICENSE file before public release.
