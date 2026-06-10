"""
文档加载、分块、入库
"""
import sys
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from pathlib import Path

import chromadb
from FlagEmbedding import FlagModel
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *


def load_documents(data_dir: Path) -> list[Document]:
    """遍历 data/docs/，加载 .md / .txt / .pdf"""
    docs = []

    """
    实际逻辑中使用rglob的方法已经是正确的，
    它会自动处理指定路径下包括子文件夹在内的所有匹配文件。
    """
    # 递归查找并加载所有.md文件
    for md_file in data_dir.rglob("*.md"):
        docs.extend(TextLoader(str(md_file), encoding="utf-8").load())
    # 递归查找并加载所有.txt文件
    for txt_file in data_dir.rglob("*.txt"):
        docs.extend(TextLoader(str(txt_file), encoding="utf-8").load())
    # 递归查找并尝试加载所有.pdf文件
    for pdf_file in data_dir.rglob("*.pdf"):
        try:
            docs.extend(PyPDFLoader(str(pdf_file)).load())
        except Exception as e:
            print(f"⚠️  跳过 PDF {pdf_file.name}: {e}")
    print(f"📄 加载了 {len(docs)} 个文档")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """分块。separators 顺序保证 Markdown 标题处优先切割，语义完整"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  分块: {len(docs)} 文档 → {len(chunks)} 块")
    return chunks


class BGEEmbedding:
    """直接用 FlagEmbedding，避免 LangChain 封装层性能损失"""

    def __init__(self, model_name: str):
        self.model = FlagModel(
            model_name,
            query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
            use_fp16=False,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()


def ingest_to_chromadb(chunks, embedding_fn, chroma_dir, collection_name="testpilot_docs"):
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = [{"source": Path(doc.metadata.get("source", "")).name,
                      "page": str(doc.metadata.get("page", 0))} for doc in batch]
        ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
        embeddings = embedding_fn.embed(texts)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        print(f"  ✅ 批次 {i // batch_size + 1}: {len(batch)} 块入库")
    print(f"\n🎉 共入库 {len(chunks)} 块 → collection: {collection_name}")


if __name__ == "__main__":
    print("=" * 60)
    print("📥 TestPilot 文档摄取")
    print("=" * 60)
    docs = load_documents(DATA_DIR)
    if not docs:
        print(f"❌ {DATA_DIR} 为空！请先放入测试框架文档。")
        sys.exit(1)
    chunks = split_documents(docs)
    print(f"🧠 加载 Embedding: {EMBEDDING_MODEL}")
    embedder = BGEEmbedding(EMBEDDING_MODEL)
    ingest_to_chromadb(chunks, embedder, CHROMA_DIR)
    print("\n✅ 摄取完成！")
