"""KG→Text model helpers (T5 fine-tuning + generation) used in the notebook."""

from __future__ import annotations

import math
from typing import Dict, List, Any

import torch
from torch.utils.data import Dataset

class KG2TextDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        ex = self.items[i]
        src = linearize_paper_style(ex["triples"])
        tgt = ex["text"]
        return {"src": src, "tgt": tgt}

def collate_fn(batch: List[Dict]):
    src_texts = [b["src"] for b in batch]
    tgt_texts = [b["tgt"] for b in batch]

    enc = tokenizer(
        src_texts,
        max_length=MAX_SRC_LEN,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    lab = tokenizer(
        text_target=tgt_texts,
        max_length=MAX_TGT_LEN,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    labels = lab["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    enc["labels"] = labels
    return enc

def eval_loss():
    model.eval()
    tot, cnt = 0.0, 0
    for batch in val_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        out = model(**batch)
        bs = batch["input_ids"].size(0)
        tot += out.loss.item() * bs
        cnt += bs
    model.train()
    return tot / max(cnt, 1)

def gen_one(triples, max_new_tokens=80):
    src = linearize_paper_style(triples)
    inp = tokenizer(src, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN).to(device)
    y = model.generate(**inp, num_beams=4, do_sample=False, max_new_tokens=max_new_tokens)
    return src, tokenizer.decode(y[0], skip_special_tokens=True)

def generate_text(triples, num_beams=4, max_new_tokens=80):
    src = linearize_paper_style(triples)
    inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN).to(device)
    with torch.no_grad():
        y = model.generate(
            **inputs,
            num_beams=num_beams,
            do_sample=False,
            max_new_tokens=max_new_tokens
        )
    gen = tokenizer.decode(y[0], skip_special_tokens=True)
    return src, gen

def kg2text_generate(triples, max_new_tokens=80):
    src = linearize_paper_style(triples)
    if not src.strip():
        return ""
    inputs = t5_tok(src, return_tensors="pt", truncation=True, max_length=MAX_SRC_LEN).to(device)
    y = t5.generate(**inputs, num_beams=1, do_sample=False, max_new_tokens=max_new_tokens)
    return t5_tok.decode(y[0], skip_special_tokens=True)
