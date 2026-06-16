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

# ===== LLM本地模型配置 =====
# LLM_MODEL_PATH = str(MODEL_DIR / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
# LLM_N_CTX = 4096
# LLM_N_THREADS = os.cpu_count() or 4
# LLM_TEMPERATURE = 0.1  # RAG 需低温度减少幻觉

# ── 方案 A: DeepSeek ──
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY")
LLM_API_BASE = "https://api.deepseek.com/v1"
LLM_MODEL_NAME = "deepseek-chat"

# [CHANGED] LLM_TEMPERATURE: 0.7 → 0.2 — RAG 需要低温度减少幻觉
# [CHANGED] 删除重复的 MAX_GENERATION_TOKENS = 512（被第40行覆盖，无意义）
LLM_TEMPERATURE = 0.2

# ===== Embedding =====
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DIM = 512

# ===== 文档分块 =====
CHUNK_SIZE = 512  # 为什么 512？测试文档一个 API/概念约 300-800 token
CHUNK_OVERLAP = 64  # 为什么 64？512 的 12.5%，覆盖被切断的句子；128 浪费

# [CHANGED] 检索 — 加入 Reranker 配置
RETRIEVAL_TOP_K = 5  # 5 个 chunk × 512 token ≈ 2560 token，Qwen 4K 上下文刚好
BM25_WEIGHT = 0.3  # 30% 关键词检索（保底精确匹配），70% 语义检索
CANDIDATE_MULTIPLIER = 3  # 初检取 top_k × 3 候选，Reranker 精排后截断到 top_k
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 用 BGE 系列 Reranker，与 BGE Embedding 配套

# ===== 生成 =====
MAX_GENERATION_TOKENS = 1024
