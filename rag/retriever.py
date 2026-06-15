"""
2.混合检索（Dense、BM25）
"""
import sys
from pathlib import Path
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
import jieba

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

    def _build_bm25_index(self):
        all_docs = self.collection.get()
        self.doc_texts = all_docs["documents"] or []
        tokenized = [list(jieba.cut(t)) for t in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"🔍 BM25 索引: {len(self.doc_texts)} 文档")

    def _dense_search(self, query, top_k):
        q_emb = self.embedder.embed([query])[0]
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k * 2,
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
        return normalized[:top_k * 2]

    def retrieve(self, query):
        dense = self._dense_search(query, self.top_k)
        bm25 = self._bm25_search(query, self.top_k)
        fused = {}
        for score, idx in dense:
            fused[idx] = fused.get(idx, 0) + score * self.dense_weight
        for score, idx in bm25:
            fused[idx] = fused.get(idx, 0) + score * self.bm25_weight
        sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return [{"content": self.doc_texts[idx], "score": round(fused[idx], 4)}
                for idx, _ in sorted_ids if idx < len(self.doc_texts)]