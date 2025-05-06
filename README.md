# optimal-llm

![python](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-%23EE4C2C?logo=pytorch)
![license](https://img.shields.io/badge/license-Apache%202.0-green)

大規模 Transformer 言語モデルを **研究用途で素早く実験** できる軽量フレームワークです。  
Multi-Query Attention + ALiBi 実装を特徴とし、SentencePiece ベースの **拡張トークナイザー** と DeepSpeed に最適化した **学習パイプライン** を同梱しています。これ一本で *Tokenizer → Pre-training → Inference* まで完結します。

## ✨ 主な特徴

| 機能 | 説明 |
|------|------|
| **Multi-Query Attention with ALiBi** | トークン長 65 k まで高速に扱える注意機構を自前実装。ヘッドごとに線形バイアスを付与する ALiBi も標準対応。:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1} |
| **SwiGLU FFN + LayerNorm** | 標準 Transformer を軽量化しながら表現力を維持。:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3} |
| **EnhancedTokenizer** | URL/数値マスキング・特殊トークン確保・実行時拡張・トークン統計など、実務で欲しかった SentecePiece ユーティリティを一式。:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5} |
| **OptimizedPretrainer** | DeepSpeed Stage-3+ZeRO, 勾配累積、ドメイン別重み付け損失、キャッシュ高速化を実装。:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7} |
| **Safetensors save / load** | モデル・データセットを安全＆高速に保存。 |

## 📦 依存ライブラリ

```bash
pip install torch>=2.0 deepspeed sentencepiece safetensors tqdm
# 検索トークンを有効にしたい場合（オプション）
pip install tavily-python

from optimal_llm import EnhancedTokenizer, TokenizerConfig

tokenizer = EnhancedTokenizer.train(
    corpus_files=["./corpus/wiki.txt", "./corpus/books.txt"],
    output_dir="./tokenizer_model",
    config=TokenizerConfig(vocab_size=65536),
    additional_special_tokens=["[IMG]"]
)

from optimal_llm import PretrainingConfig, OptimizedPretrainer

cfg = PretrainingConfig()
pretrainer = OptimizedPretrainer(cfg)
pretrainer.train()

from optimal_llm import OptimalLLMModel, EnhancedTokenizer
tok = EnhancedTokenizer.load("./tokenizer_model")
model = OptimalLLMModel.from_pretrained("./checkpoints/step_100000").eval()

prompt = "Q: 大阪城はどこにありますか？\nA:"
print(model.generate(tok, prompt, max_new_tokens=40))

optimal_llm/
├─ optimal_llm.py        # メイン実装
├─ examples/             # 使い方スクリプト
├─ data/                 # (任意) コーパス配置
└─ checkpoints/          # 学習済みモデル

