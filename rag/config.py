"""
全局配置
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
EVAL_DIR = PROJECT_ROOT / "eval_results"

# ===== LLM =====
LLM_MODEL_PATH = str(MODEL_DIR / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
LLM_N_CTX = 4096
LLM_N_THREADS = os.cpu_count() or 4
LLM_TEMPERATURE = 0.1   # RAG 需低温度减少幻觉

# ===== Embedding =====
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# ===== 文档分块 =====
CHUNK_SIZE = 512        # 为什么 512？测试文档一个 API/概念约 300-800 token
CHUNK_OVERLAP = 64      # 为什么 64？512 的 12.5%，覆盖被切断的句子；128 浪费

# ===== 检索 =====
RETRIEVAL_TOP_K = 5     # 5 个 chunk × 512 token ≈ 2560 token，Qwen 4K 上下文刚好
BM25_WEIGHT = 0.3       # 30% 关键词检索（保底精确匹配），70% 语义检索

# ===== 生成 =====
MAX_GENERATION_TOKENS = 512