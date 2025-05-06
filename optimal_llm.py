import os
import re
import json
import math
import logging
import torch
import sentencepiece as spm
from typing import List, Dict, Optional, Union
from collections import defaultdict
from dataclasses import dataclass
import deepspeed
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm.auto import tqdm
from pathlib import Path
from collections import defaultdict
from safetensors.torch import save_file, load_file

#LLM_program
class OptimalLLMConfig:
    def __init__(self, 
                 vocab_size=65536, 
                 max_position_embeddings=65536, 
                 n_layers=10, 
                 d_model=512, 
                 n_heads=8, 
                 d_ff=2048,
                 use_alibi=True):
        """モデル設定を保持するクラス。"""
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.use_alibi = use_alibi  # 長文用のALiBiバイアス使用フラグ

class MultiheadAttentionWithALiBi(nn.Module):
    def __init__(self, d_model, n_heads, max_pos, use_alibi=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 注意機構の線形変換パラメータ
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)  # MQA: キーはhead_dim (共有)
        self.v_proj = nn.Linear(d_model, self.head_dim)  # MQA: バリューもhead_dim
        self.out_proj = nn.Linear(d_model, d_model)
        # ALiBiバイアスの準備（各ヘッドに一様な負の直線傾斜を持たせる）
        self.use_alibi = use_alibi
        if use_alibi:
            # ヘッドごとのバイアス斜率: ヘッド番号が大きいほどバイアス弱 (論文に基づき等比列)
            slopes = [1.0 / (2**(i/(n_heads-1))) for i in range(n_heads)]
            # バイアステンソル: shape (n_heads, max_pos)
            alibi = torch.tensor([
                [ -slope * j for j in range(max_pos) ]
                for slope in slopes
            ])
            self.register_buffer('alibi', alibi, persistent=False)  # 勾配不要
        else:
            self.register_buffer('alibi', None, persistent=False)
    
    def forward(self, x, mask=None):
        B, T, C = x.shape  # (バッチ, シーケンス長, d_model)
        # Q, K, V計算
        q = self.q_proj(x)      # (B, T, d_model)
        k = self.k_proj(x)      # (B, T, head_dim)
        v = self.v_proj(x)      # (B, T, head_dim)
        # ヘッドに分割
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        # MQA: k, vは各ヘッド共通なのでhead次元に同じ値を繰り返し
        k = k.unsqueeze(1).expand(B, self.n_heads, T, self.head_dim)   # (B, n_heads, T, head_dim)
        v = v.unsqueeze(1).expand(B, self.n_heads, T, self.head_dim)   # (B, n_heads, T, head_dim)
        # スケーリング
        q = q / math.sqrt(self.head_dim)
        # アテンションスコア計算
        attn_weights = torch.einsum('bhqd, bhkd -> bhqk', q, k)  # (B, n_heads, T, T)
        # ALiBiバイアス追加
        if self.use_alibi and self.alibi is not None:
            # 先頭T列のみ取り出し適用
            attn_weights += self.alibi[:, :T].unsqueeze(0)  # (1, n_heads, T)
            # 上式でbroadcastし (B, n_heads, T, T) に自動拡張される
        # マスク処理（未来トークン遮断 or パディング遮断）
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        # ソフトマックス
        attn_probs = F.softmax(attn_weights, dim=-1)
        # 加重和
        attn_output = torch.einsum('bhqk, bhkd -> bhqd', attn_probs, v)  # (B, n_heads, T, head_dim)
        # 結合して出力サイズに戻す
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(attn_output)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, config: OptimalLLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model, eps=1e-6)  # RMSNormも可
        self.attn = MultiheadAttentionWithALiBi(config.d_model, config.n_heads, 
                                                config.max_position_embeddings, 
                                                use_alibi=config.use_alibi)
        self.ln2 = nn.LayerNorm(config.d_model, eps=1e-6)
        # SwiGLU: 2 * d_ff weights for gated linear unit
        self.ffn_gate = nn.Linear(config.d_model, config.d_ff)
        self.ffn_lin = nn.Linear(config.d_model, config.d_ff)
        self.ffn_out = nn.Linear(config.d_ff, config.d_model)
    
    def forward(self, x, mask=None):
        # Attentionサブレイヤ
        a = self.ln1(x)
        attn_out = self.attn(a, mask=mask)
        x = x + attn_out  # 残差接続
        # FFNサブレイヤ (SwiGLU)
        m = self.ln2(x)
        gate = F.silu(self.ffn_gate(m))        # ゲート: SiLU活性
        ff = self.ffn_lin(m) * gate            # 要素積でゲート適用
        ffn_out = self.ffn_out(ff)
        x = x + ffn_out  # 残差接続
        return x

class OptimalLLMModel(nn.Module):
    def __init__(self, config: OptimalLLMConfig):
        super().__init__()
        self.config = config
        # トークン埋め込み
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        # Rotary埋め込み用: 実装簡略のためここでは省略（MultiheadAttention内で適用想定）
        # Transformer層スタック
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        # 最終層正規化
        self.ln_f = nn.LayerNorm(config.d_model, eps=1e-6)
        # ラベル予測ヘッド（出力層）
        self.output_head = nn.Linear(config.d_model, config.vocab_size)
        # 入出力埋め込みを**共有しない**（embeddingと別の重み）
        # （embedding重みとoutput_head重みが独立している）
    
    def forward(self, input_ids, attention_mask=None):
        # 入力を埋め込みベクトルに変換
        x = self.embed_tokens(input_ids)  # (batch, seq_len, d_model)
        # マスク（未来のトークンを見ないようにするマスク）を作成
        if attention_mask is None:
            # 下三角マスクを生成
            seq_len = input_ids.size(1)
            mask = torch.ones((seq_len, seq_len), device=input_ids.device).tril_()  # (T, T) 下三角(含対角)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T) に拡張
        else:
            # attention_maskが与えられた場合、それを利用（0がマスク対象とする）
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            # 未来のトークン遮断も同時に行う
            seq_len = input_ids.size(1)
            causal_mask = torch.ones((seq_len, seq_len), device=input_ids.device).tril_()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,T,T)
            mask = (mask * causal_mask)  # パディングと因果関係の複合マスク
        # Transformer層を適用
        for layer in self.layers:
            x = layer(x, mask=mask)
        # 最終正規化
        x = self.ln_f(x)
        # 出力トークン確率
        logits = self.output_head(x)  # (batch, seq_len, vocab_size)
        return logits
    
    def generate(self, tokenizer, prompt, max_new_tokens=256, temperature=1.0, top_p=0.95, 
                 num_beams=1, debug=False):
        """
        プロンプトからテキストを生成する。チェイン・オブ・ソートや検索も内部で処理。
        """
        self.eval()
        # トークナイズ
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(next(self.parameters()).device)  # モデルと同じデバイスへ
        past_search_results = ""  # 検索結果の蓄積（ユーザには直接出力しない）
        generated_text = ""
        # 推論ループ：逐次トークン生成
        for step in range(max_new_tokens):
            # 必要に応じ過去の検索結果を追加
            if past_search_results:
                # 検索結果をモデル入力に含める（単純に直前に連結）
                # 実際には特殊トークンで区切る等処理が必要だが簡略化
                result_ids = tokenizer.encode(past_search_results, return_tensors='pt').to(input_ids.device)
                input_with_result = torch.cat([input_ids, result_ids], dim=1)
            else:
                input_with_result = input_ids
            # 出力を計算
            logits = self(input_with_result)
            # 最終時刻ステップのlogitsを取得
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            # 温度スケーリングと確率分布計算
            next_token_logits = next_token_logits / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            # top-p フィルタリング
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.sum(cumulative_probs <= top_p)
            probs[sorted_indices[cutoff_index:]] = 0
            probs = probs / probs.sum()  # 再正規化
            # トークンをサンプリング
            next_token_id = torch.multinomial(probs, num_samples=1)
            # ビームサーチ: num_beams>1の場合、ここで上位ビーム分岐処理を入れる（省略）
            # 生成トークンをシーケンスに追加
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            token_str = tokenizer.decode(next_token_id)
            if debug:
                print(f"[DEBUG] Generated token: {token_str}")
            # 特殊トークンに応じた処理
            if token_str == "[SEARCH]":
                # 直後の検索クエリをモデルが出力するはずなので、それを組み立てる
                query_text = ""
                if debug:
                    print("[DEBUG] Search token encountered. Collecting query...")
                continue  # 次ループでクエリテキスト収集を継続
            if 'query_text' in locals() and token_str != "[RESULT]":
                # [SEARCH]が出現し、[RESULT]前の間は検索クエリを構築
                if token_str.strip() == "":
                    # 空白トークンは無視
                    continue
                if token_str == "</s>":  # 文末ならクエリ終了
                    token_str = ""
                query_text += token_str
                if debug:
                    print(f"[DEBUG] Building search query: {query_text}")
                continue
            if 'query_text' in locals() and token_str == "[RESULT]":
                # 検索クエリ終了、API検索実行
                if debug:
                    print(f"[DEBUG] Running search for query: {query_text}")
                try:
                    from tavily import TavilyClient
                    tavily_client = TavilyClient(api_key="YOUR_TAVILY_API_KEY")
                    search_response = tavily_client.get_search_context(query=query_text)
                except Exception as e:
                    search_response = "検索結果を取得できませんでした。"
                # 検索結果を保存し、モデル入力に追加
                past_search_results = f"【検索結果】{search_response}\n"
                if debug:
                    print(f"[DEBUG] Search result: {search_response[:100]}...")  # 先頭100文字を表示
                # [SEARCH]トークンとクエリをクリア
                del query_text
                continue  # 改めてループ先頭へ（検索結果を追加して次トークン生成）
            # 通常トークンの場合、結果文字列に追加
            generated_text += token_str
            # EOSトークンなら生成終了
            if token_str == "</s>" or token_str == "<|endoftext|>" or token_str.strip() == "":
                break
        return generated_text
    
    def save_pretrained(self, save_dir: str):
        """モデルの重みと設定を保存する。"""
        # 重みをsafetensors形式で保存
        state_dict = self.state_dict()
        save_file(state_dict, f"{save_dir}/model.safetensors")
        # コンフィグをjsonに保存
        import json
        config_dict = {
            "vocab_size": self.config.vocab_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "n_layers": self.config.n_layers,
            "d_model": self.config.d_model,
            "n_heads": self.config.n_heads,
            "d_ff": self.config.d_ff,
            "use_alibi": self.config.use_alibi
        }
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(config_dict, f)
    
    @classmethod
    def from_pretrained(cls, load_dir: str):
        """保存された重みと設定からモデルを復元する。"""
        import json
        # 設定読み込み
        with open(f"{load_dir}/config.json", "r") as f:
            config_dict = json.load(f)
        config = OptimalLLMConfig(**config_dict)
        # モデル初期化
        model = cls(config)
        # safetensorsから重みロード
        state_dict = load_file(f"{load_dir}/model.safetensors")
        model.load_state_dict(state_dict, strict=True)
        return model

#train_tokenizer
@dataclass
class TokenizerConfig:
    """トークナイザー設定のデータクラス"""
    vocab_size: int = 65536
    max_sentence_length: int = 16384
    normalization: str = "nmt_nfkc_cf"  # 大文字小文字を区別しないNFKC
    split_digits: bool = True
    allow_whitespace: bool = True
    control_symbols: List[str] = None
    user_defined_symbols: List[str] = None
    byte_fallback: bool = True  # 未知語をUTF-8バイトで分解

class EnhancedTokenizer:
    def __init__(self, model_file: str = None, config: TokenizerConfig = None):
        self.sp = spm.SentencePieceProcessor()
        if model_file:
            self.sp.load(model_file)
        self.config = config or TokenizerConfig()
        self.special_tokens = self._initialize_special_tokens()
        self.token_metrics = defaultdict(int)  # トークン使用統計

    def _initialize_special_tokens(self):
        """特殊トークンを確実に確保するための初期化"""
        base_specials = [
            "<|endoftext|>", "[SEARCH]", "[RESULT]", 
            "<s>", "</s>", "<pad>", "<unk>", "[IMG]"
        ]
        return {t: self.sp.piece_to_id(t) for t in base_specials}

    def _preprocess_text(self, text: str) -> str:
        """テキストの前処理（正規化と特殊パターン処理）"""
        # URLと数値のマスキング
        text = re.sub(r"http\S+", "[URL]", text)
        text = re.sub(r"\d+", "[NUM]", text)
        
        # 正規化処理
        if self.config.normalization == "nmt_nfkc_cf":
            text = text.lower().strip()
        return text

    @staticmethod
    def train(
        corpus_files: List[str],
        output_dir: str,
        config: TokenizerConfig,
        additional_special_tokens: List[str] = None
    ):
        """改良された訓練メソッド"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 特殊トークン統合
        all_special = list({
            *config.control_symbols or [],
            *config.user_defined_symbols or [],
            *additional_special_tokens or []
        })
        
        # 語彙サイズ調整（特殊トークン分を正確に確保）
        effective_vocab_size = config.vocab_size - len(all_special)
        
        # SentencePieceパラメータ
        train_params = {
            'input': corpus_files,
            'model_prefix': f"{output_dir}/sp_model",
            'vocab_size': effective_vocab_size,
            'model_type': 'unigram',
            'character_coverage': 0.99999,
            'num_threads': os.cpu_count(),
            'max_sentence_length': config.max_sentence_length,
            'split_digits': config.split_digits,
            'allow_whitespace_only_pieces': config.allow_whitespace,
            'normalization_rule_name': config.normalization,
            'control_symbols': all_special,
            'byte_fallback': config.byte_fallback,
            'pad_id': 3,
            'bos_id': 1,
            'eos_id': 2,
            'unk_id': 0
        }
        
        # 訓練実行
        spm.SentencePieceTrainer.train(**train_params)
        
        # モデル再ロードと検証
        tokenizer = EnhancedTokenizer(
            f"{output_dir}/sp_model.model", 
            config
        )
        tokenizer._validate_vocab()
        return tokenizer

    def _validate_vocab(self):
        """語彙の整合性チェック"""
        required_tokens = ["<|endoftext|>", "[SEARCH]", "[RESULT]", "[IMG]"]
        for tok in required_tokens:
            if tok not in self.special_tokens:
                raise ValueError(f"Missing required token: {tok}")

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        return_token_metrics: bool = False
    ) -> Union[List[int], dict]:
        """拡張エンコードメソッド"""
        processed = self._preprocess_text(text)
        ids = self.sp.encode(processed, out_type=int)
        
        # 統計情報更新
        for token_id in ids:
            self.token_metrics[token_id] += 1
        
        result = {"input_ids": ids}
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([ids], dtype=torch.long)
        
        if return_token_metrics:
            result["token_metrics"] = self._get_token_metrics(ids)
        
        return result

    def _get_token_metrics(self, token_ids: List[int]) -> Dict:
        """トークン統計情報を取得"""
        return {
            "avg_length": len(token_ids),
            "unk_count": sum(1 for i in token_ids if i == self.special_tokens["<unk>"]),
            "special_token_ratio": sum(1 for i in token_ids if i in self.special_tokens.values()) / len(token_ids)
        }

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = False,
        clean_up: bool = True
    ) -> str:
        """改良デコードメソッド"""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # 特殊トークンフィルタリング
        filtered_ids = [
            i for i in token_ids
            if not (skip_special_tokens and i in self.special_tokens.values())
        ]
        
        text = self.sp.decode(filtered_ids)
        
        # 後処理
        if clean_up:
            text = re.sub(r"\s+", " ", text).strip()
            text = text.replace("[NUM]", "123").replace("[URL]", "https://example.com")
        
        return text

    def analyze_efficiency(self, sample_texts: List[str]) -> Dict:
        """トークン効率分析メソッド"""
        results = []
        for text in sample_texts:
            encoded = self.encode(text)
            results.append({
                "original_length": len(text),
                "token_count": len(encoded["input_ids"]),
                "compression_ratio": len(text) / len(encoded["input_ids"]),
                "unk_tokens": encoded.get("token_metrics", {}).get("unk_count", 0)
            })
        return {
            "avg_compression": sum(r["compression_ratio"] for r in results) / len(results),
            "max_token_length": max(r["token_count"] for r in results),
            "unk_rate": sum(r["unk_tokens"] for r in results) / sum(r["token_count"] for r in results)
        }

    def save(self, output_dir: str):
        """モデルと設定を保存"""
        self.sp.save(f"{output_dir}/sp.model")
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(asdict(self.config), f)
        
        # トークン統計を保存
        with open(f"{output_dir}/token_metrics.json", "w") as f:
            json.dump(self.token_metrics, f)

    @classmethod
    def load(cls, output_dir: str):
        """モデルと設定をロード"""
        config_path = f"{output_dir}/config.json"
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        tokenizer = cls(
            model_file=f"{output_dir}/sp.model",
            config=TokenizerConfig(**config_data))
        
        # トークン統計をロード
        metrics_path = f"{output_dir}/token_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                tokenizer.token_metrics = json.load(f)
        
        return tokenizer

    def add_special_tokens(self, new_tokens: List[str]):
        """実行時特殊トークン追加（動的拡張）"""
        for tok in new_tokens:
            if tok not in self.special_tokens:
                new_id = self.sp.get_piece_size()
                self.sp.add_control_symbols([tok])
                self.special_tokens[tok] = new_id

    def image_token_handler(self, image_tensor: torch.Tensor):
        """画像トークン処理（マルチモーダル対応）"""
        # 画像処理ロジック（例: Vision Transformerからの埋め込み）
        return [self.special_tokens["[IMG]"]]

#pretrain
class OptimizedPretrainer:
    def __init__(self, config):
        self.config = config
        self._init_distributed()
        self._load_components()
        self._prepare_datasets()
        self._setup_engine()
        self.metrics = {
            'loss': [],
            'math_loss': [],
            'code_loss': [],
            'dialogue_loss': [],
            'ppl': []
        }

    def _init_distributed(self):
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.world_size = int(os.getenv("WORLD_SIZE", 1))
        torch.cuda.set_device(self.local_rank)
        deepspeed.init_distributed()

    def _load_components(self):
        # トークナイザーとモデルの初期化
        if self.config.resume_dir:
            self.model = OptimalLLMModel.from_pretrained(self.config.resume_dir)
            self.tokenizer = EnhancedTokenizer.load(self.config.resume_dir)
            with open(f"{self.config.resume_dir}/training_state.json") as f:
                self.training_state = json.load(f)
        else:
            model_config = OptimalLLMConfig(
                vocab_size=self.config.vocab_size,
                d_model=self.config.d_model,
                max_position_embeddings=self.config.max_seq_length,
                use_alibi=True
            )
            self.model = OptimalLLMModel(model_config)
            self.tokenizer = EnhancedTokenizer.load(self.config.tokenizer_path)
            self.training_state = {'step': 0, 'epoch': 0}

        # ドメイン重み設定
        self.domain_weights = {
            'math': 2.0,
            'code': 1.5,
            'dialogue': 1.0
        }

    def _prepare_datasets(self):
        # マルチドメインデータセットの準備
        self.datasets = {}
        for domain in ['math', 'code', 'dialogue']:
            cache_path = Path(self.config.cache_dir) / f"{domain}_tokenized.safetensors"
            if cache_path.exists():
                self.datasets[domain] = self._load_cached_dataset(cache_path)
            else:
                self.datasets[domain] = self._process_domain(domain, cache_path)

        # 分散サンプラー
        self.samplers = {
            domain: DistributedSampler(dataset, num_replicas=self.world_size, 
                                     rank=self.local_rank)
            for domain, dataset in self.datasets.items()
        }

    def _process_domain(self, domain: str, cache_path: Path):
        # ドメイン別データ処理（トークン化とキャッシュ）
        raw_files = list((Path(self.config.data_dir)/domain).glob("*.json"))
        sequences = []
        
        for file in tqdm(raw_files, desc=f"Processing {domain} data"):
            with open(file) as f:
                data = json.load(f)
            
            # ドメイン固有前処理
            processed = self._domain_specific_processing(domain, data['text'])
            
            # トークン化とチャンキング
            encoded = self.tokenizer.encode(processed, return_tensors="pt")['input_ids'][0]
            for i in range(0, len(encoded), self.config.seq_length):
                chunk = encoded[i:i+self.config.seq_length]
                if len(chunk) < self.config.min_seq_length:
                    continue
                sequences.append({
                    'input_ids': chunk,
                    'domain': domain
                })
        
        # Safetensors形式で保存
        save_data = {
            'input_ids': torch.stack([s['input_ids'] for s in sequences]),
            'domains': [s['domain'] for s in sequences]
        }
        save_file(save_data, str(cache_path))
        return sequences

    def _domain_specific_processing(self, domain: str, text: str):
        # ドメイン別前処理ロジック
        if domain == 'math':
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\$(.*?)\$', r'\1', text)  # LaTeX簡素化
        elif domain == 'code':
            text = text.replace('\t', '    ').replace('\\n', '\n')
        return text.strip()

    def _setup_engine(self):
        # DeepSpeed設定
        ds_config = {
            "train_batch_size": self.config.batch_size * self.world_size,
            "gradient_accumulation_steps": self.config.grad_accum,
            "fp16": {"enabled": self.config.fp16},
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu"},
                "offload_optimizer": {"device": "cpu"},
                "contiguous_gradients": True
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "number_checkpoints": 4
            },
            "steps_per_print": 100,
            "wall_clock_breakdown": False
        }

        self.model_engine, *_ = deepspeed.initialize(
            model=self.model,
            config=ds_config,
            model_parameters=self.model.parameters()
        )

    def train(self):
        # メイン訓練ループ
        start_step = self.training_state['step']
        progress_bar = tqdm(
            initial=start_step,
            total=self.config.total_steps,
            disable=self.local_rank != 0
        )

        while self.training_state['step'] < self.config.total_steps:
            batches = self._prepare_batches()
            
            for batch in batches:
                if self.training_state['step'] >= self.config.total_steps:
                    break

                loss, domain_losses = self._training_step(batch)
                
                # メトリクス記録
                if self.local_rank == 0:
                    self._update_metrics(loss, domain_losses)
                    self._log_progress(progress_bar)
                
                # チェックポイント保存
                if self.training_state['step'] % self.config.save_interval == 0:
                    self._save_checkpoint()
                
                self.training_state['step'] += 1

    def _prepare_batches(self):
        # ドメイン重みに基づくバッチサンプリング
        domains = list(self.domain_weights.keys())
        weights = torch.tensor([self.domain_weights[d] for d in domains])
        probs = weights / weights.sum()

        batches = []
        for _ in range(self.config.grad_accum):
            # ドメイン選択
            selected_domain = domains[torch.multinomial(probs, 1).item()]
            
            # データセットからサンプリング
            dataset = self.datasets[selected_domain]
            sampler = self.samplers[selected_domain]
            indices = torch.randint(0, len(dataset), (self.config.batch_size,))
            
            batch = [dataset[i] for i in indices]
            batches.append(self._collate_fn(batch))
        
        return batches

    def _training_step(self, batch):
        # フォワード/バックワード処理
        self.model_engine.train()
        
        inputs = batch['input_ids'].to(self.model_engine.device)
        targets = inputs.clone()
        attention_mask = (inputs != self.tokenizer.special_tokens['<pad>']).long()

        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            outputs = self.model_engine(inputs, attention_mask=attention_mask)
            loss = self._calculate_domain_loss(outputs, targets, batch['domains'])

        self.model_engine.backward(loss)
        self.model_engine.step()

        return loss.item(), self._compute_domain_losses(outputs, targets, batch['domains'])

    def _calculate_domain_loss(self, logits, targets, domains):
        # ドメイン別損失重み付け
        loss = 0
        for i, domain in enumerate(domains):
            shift_logits = logits[i, :-1].contiguous()
            shift_labels = targets[i, 1:].contiguous()
            
            domain_loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.tokenizer.special_tokens['<pad>']
            )
            
            loss += domain_loss * self.domain_weights[domain]
        
        return loss / len(domains)

    def _save_checkpoint(self):
        # チェックポイント保存
        save_dir = Path(self.config.save_dir) / f"step_{self.training_state['step']}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # モデル保存
        self.model_engine.module.save_pretrained(save_dir)
        
        # トークナイザー保存
        self.tokenizer.save(save_dir)
        
        # トレーニング状態保存
        with open(save_dir / "training_state.json", "w") as f:
            json.dump(self.training_state, f)
        
        # メトリクス保存
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(self.metrics, f)

    def _update_metrics(self, loss, domain_losses):
        # メトリクス更新
        self.metrics['loss'].append(loss)
        self.metrics['ppl'].append(math.exp(loss))
        for domain, loss_val in domain_losses.items():
            self.metrics[f'{domain}_loss'].append(loss_val)

    def _log_progress(self, progress_bar):
        # 進捗表示
        avg_loss = sum(self.metrics['loss'][-100:]) / len(self.metrics['loss'][-100:])
        progress_bar.set_postfix({
            'loss': f"{avg_loss:.4f}",
            'math': f"{self.metrics['math_loss'][-1]:.4f}",
            'code': f"{self.metrics['code_loss'][-1]:.4f}",
            'ppl': f"{math.exp(avg_loss):.2f}"
        })
        progress_bar.update(1)

class PretrainingConfig:
    def __init__(self):
        # データ設定
        self.data_dir = "./data"
        self.cache_dir = "./tokenized_cache"
        self.tokenizer_path = "./tokenizer_model"
        self.seq_length = 4096
        self.min_seq_length = 512
        
        # 学習パラメータ
        self.total_steps = 100000
        self.batch_size = 8
        self.grad_accum = 4
        self.fp16 = True
        
        # チェックポイント
        self.save_dir = "./checkpoints"
        self.save_interval = 1000
        self.resume_dir = None
        
        # モデル設定
        self.vocab_size = 65536
        self.d_model = 512
        self.n_heads = 8
        self.d_ff = 2048