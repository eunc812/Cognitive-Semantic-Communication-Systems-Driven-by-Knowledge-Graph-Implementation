## Cognitive Semantic Communication Systems Driven by Knowledge Graph

+ This repository provides a implementation of the paper: [Cognitive Semantic Communication Systems Driven by Knowledge Graph: Principle, Implementation, and Performance Evaluation](https://ieeexplore.ieee.org/document/10262128)
+ The repository covers WebNLG aligning, Knowledge Base error correction, T5 finetuning, T5 based KG-to-Text, and the main experimental figures in the paper.
+ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/drive/1a5VWUFEnRR0qWTyPuYXS-ZhchxqzV1Cu?usp=sharing
)
## Architecture
  ![Architecture](Images/architecture.png)

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






- Sentence Similarity vs p (BSC wrong)

	![AWGN PSNR](Images/result_3.png)






- BLEU vs p (BSC wrong)

	![AWGN PSNR](Images/result_4.png)
