"""
主流程
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *
from rag.retriever import HybridRetriever
from rag.generator import create_rag_module


class TestPilotRAGPipeline:
    def __init__(self):
        print("🚀 初始化 TestPilot RAG...")
        print("  📚 加载检索器...")
        self.retriever = HybridRetriever()
        print("  🧠 加载模型...")
        self.rag_module = create_rag_module()
        print("✅ 就绪！\n")

    def ask(self, question):
        results = self.retriever.retrieve(question)
        if not results:
            return {"question": question,
                    "answer": "抱歉，知识库中没有找到相关内容。",
                    "sources": [], "context": []}
        context = "\n\n---\n\n".join(
            [f"[文档 {i+1}] (相关度: {r['score']})\n{r['content']}"
             for i, r in enumerate(results)])
        prediction = self.rag_module(context=context, question=question)
        return {"question": question, "answer": prediction.answer,
                "sources": prediction.sources, "context": results}

    def interactive(self):
        print("=" * 60)
        print("🧪 TestPilot RAG 问答")
        print("   quit 退出 | sources 查看引用")
        print("=" * 60)
        while True:
            try:
                q = input("\n🤔 问题: ").strip()
                if not q:
                    continue
                if q.lower() == "quit":
                    print("👋 再见！")
                    break
                r = self.ask(q)
                print(f"\n📝 回答:\n{r['answer']}")
                if r.get("context"):
                    print(f"\n📚 参考了 {len(r['context'])} 个文档片段")
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


if __name__ == "__main__":
    TestPilotRAGPipeline().interactive()