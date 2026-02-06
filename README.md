# Cognitive Semantic Communication Systems Driven by Knowledge Graph Implementation
- This repository provides a reproducible implementation of the paper [Cognitive Semantic Communication Systems Driven by Knowledge Graph: Principle, Implementation, and Performance Evaluation](https://ieeexplore.ieee.org/document/10262128)
- The pipeline covers WebNLG preprocessing, T5-based KG-to-Text, bit-level baselines vs. the proposed coding, and BSC channel evaluation, and reproduces the main experimental figures in the paper.

## Architecture
![Architecture](Images/architecture.jpg)

## Preprocess
python preprocess.py

## Channel
python channel.py

## Train
python train.py 

## Evaluation
python eval.py

## Result
- bits vs sentence length

![AWGN PSNR](Images/result_1.png)


- bits vs number of texts

![AWGN PSNR](Images/result_2.png)


- Sentence Similarity vs p

![AWGN PSNR](Images/result_3.png)


- BLEU vs p

![AWGN PSNR](Images/result_4.png)
