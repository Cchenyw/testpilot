"""
2.混合检索 + Reranker 精排（Dense、BM25 → RRF 融合 → Reranker → Top-K）
"""
import sys
from pathlib import Path
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

# [CHANGED] 新增 FlagReranker 导入
from FlagEmbedding import FlagReranker

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *
from rag.ingest import BGEEmbedding


class HybridRetriever:
    def __init__(self, chroma_dir=CHROMA_DIR, embedding_model=EMBEDDING_MODEL,
                 top_k=RETRIEVAL_TOP_K, bm25_weight=BM25_WEIGHT):
        self.top_k = top_k
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight
        self.embedder = BGEEmbedding(embedding_model)
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.client.get_collection("testpilot_docs")
        self._build_bm25_index()

        # [CHANGED] 初始化 Reranker（延迟加载，节省内存）
        self._reranker = None
        self._reranker_model = RERANKER_MODEL

    # [CHANGED] 延迟加载 Reranker，避免每次 import 都初始化
    @property
    def reranker(self):
        if self._reranker is None:
            print(f"🔄 加载 Reranker: {self._reranker_model}")
            self._reranker = FlagReranker(
                self._reranker_model,
                use_fp16=False,  # CPU 推理用 FP32 更稳定
            )
        return self._reranker

    def _build_bm25_index(self):
        all_docs = self.collection.get()
        self.doc_texts = all_docs["documents"] or []
        tokenized = [list(jieba.cut(t)) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"🔍 BM25 索引: {len(self.doc_texts)} 文档")

    def _dense_search(self, query, top_k):
        q_emb = self.embedder.embed([query])[0]
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k,
                                         include=["documents", "distances"])
        scored = []
        for i in range(len(results["ids"][0])):
            idx = int(results["ids"][0][i].replace("chunk_", ""))
            score = 1.0 / (1.0 + results["distances"][0][i])
            scored.append((score, idx))
        return scored

    def _bm25_search(self, query, top_k):
        tok = list(jieba.cut(query))
        scores = self.bm25.get_scores(tok)
        mx = max(scores) if max(scores) > 0 else 1
        normalized = [(s / mx, i) for i, s in enumerate(scores)]
        normalized.sort(key=lambda x: x[0], reverse=True)
        return normalized[:top_k]

    # [CHANGED] 新增：Reranker 精排
    def _rerank(self, query, candidates: list[dict]) -> list[dict]:
        """用 Cross-Encoder Reranker 对候选列表重新打分排序"""
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self.reranker.compute_score(pairs)
        # scores 是 list[float] 或单个 float
        if not isinstance(scores, list):
            scores = [scores]
        # 绑定新分数
        for c, s in zip(candidates, scores):
            c["score"] = round(float(s), 4)
        # 按新分数降序排列
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:self.top_k]

    def retrieve(self, query):
        # ───── ① 初检：多拿一些候选 ─────
        fetch_k = self.top_k * CANDIDATE_MULTIPLIER
        dense = self._dense_search(query, fetch_k)
        bm25 = self._bm25_search(query, fetch_k)

        # ───── ② RRF 融合 ─────
        fused = {}
        for score, idx in dense:
            fused[idx] = fused.get(idx, 0) + score * self.dense_weight
        for score, idx in bm25:
            fused[idx] = fused.get(idx, 0) + score * self.bm25_weight
        sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # ───── ③ 构建候选列表 ─────
        candidates = []
        for idx, score in sorted_ids:
            if idx < len(self.doc_texts):
                candidates.append({
                    "content": self.doc_texts[idx],
                    "score": round(score, 4),
                    "idx": idx,
                })

        # [CHANGED] ───── ④ Reranker 精排 ─────
        if candidates:
            candidates = self._rerank(query, candidates)

        return [{k: c[k] for k in ("content", "score")} for c in candidates]
