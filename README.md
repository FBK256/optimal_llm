# Optimal-LLM 🚀

A research-oriented, lightweight Large Language Model framework built for rapid experimentation  
with **ALiBi**, **Multi-Query Attention (MQA)**, an **enhanced SentencePiece tokenizer**, and an end-to-end **DeepSpeed** pre-training pipeline.

<p align="center">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/yourname/optimal-llm">
  <img alt="GitHub license" src="https://img.shields.io/github/license/yourname/optimal-llm">
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
</p>

---

## ✨ Key Features

* **State-of-the-art architecture**  
  * ALiBi positional bias for context windows up to 65 k tokens  
  * Multi-Query Attention (shared K/V) for faster decoding  
  * SwiGLU feed-forward and RMS-ready LayerNorms  
* **Extensible tokenizer**  
  * SentencePiece Unigram with byte-fallback, domain-aware preprocessing, dynamic special tokens, and usage analytics.  
* **Multi-domain pre-training**  
  * Weighted sampling for *math* / *code* / *dialogue* corpora  
  * DeepSpeed ZeRO-3 + activation checkpointing out of the box.  
* **Easy save / load** via `safetensors`.  
* **Search-augmented generation** placeholder for future RAG extensions.  

---

## 🏗 Repository Layout

