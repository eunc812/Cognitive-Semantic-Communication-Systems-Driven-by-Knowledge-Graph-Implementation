# Cognitive Semantic Communication Systems Driven by Knowledge Graph Implementation
- This implementation is based on the paper [Cognitive Semantic Communication Systems Driven by Knowledge Graph: Principle, Implementation, and Performance Evaluation](https://ieeexplore.ieee.org/document/10262128)
- WebNLG download + preprocessing / alignment
- KG→Text helpers (T5-based)
- Bit accounting + baselines (bit7 / Huffman) + proposed representation coding
- BSC channel + simple correction utilities
- Small analysis helpers (bins / cumulative plots)

## Quickstart
```bash
pip install -r requirements.txt
python scripts/reproduce_colab.py --base_dir ./runs/webnlg_run
```
## Architecture
![Architecture](Images/architecture.jpg)

## Result
- bits vs sentence length

![AWGN PSNR](Images/result_1.png)


- bits vs number of texts

![AWGN PSNR](Images/result_2.png)


- Sentence Similarity vs p

![AWGN PSNR](Images/result_3.png)


- BLEU vs p

![AWGN PSNR](Images/result_4.png)
