"""
6.DSpy 自动化
"""
"""DSPy BootstrapFewShot 自动优化"""
import sys, json
from pathlib import Path
import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *
from rag.retriever import HybridRetriever
from rag.generator import LlamaCppLM, RAGModule
from rag.evaluate import load_test_questions


def run_optimization():
    print("=" * 60)
    print("🔧 DSPy 自动优化 (BootstrapFewShot)")
    print("=" * 60)

    lm = LlamaCppLM(LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_TEMPERATURE)
    dspy.configure(lm=lm)
    retriever = HybridRetriever()
    questions = load_test_questions()

    trainset = []
    for q in questions[:8]:
        results = retriever.retrieve(q["question"])
        context = "\n\n---\n\n".join(
            [f"[文档 {i+1}]\n{r['content']}" for i, r in enumerate(results)])
        example = dspy.Example(
            context=context, question=q["question"],
            answer=q["ground_truth"],
            sources=", ".join([r.get("content", "")[:50] + "..."
                               for r in results])
        ).with_inputs("context", "question")
        trainset.append(example)

    print(f"\n📊 训练集: {len(trainset)} 个示例")

    def simple_metric(example, pred, trace=None):
        return sum(1 for kw in example.answer.split() if kw in pred.answer) \
               / max(len(example.answer.split()), 1)

    optimizer = dspy.BootstrapFewShot(
        metric=simple_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
    )

    print("\n🔄 开始优化（可能需要几分钟）...")
    optimized = optimizer.compile(RAGModule(), trainset=trainset)

    save_path = EVAL_DIR / "optimized_rag.json"
    optimized.save(str(save_path))
    print(f"\n✅ 优化完成！保存至: {save_path}")
    print("优化后可使用 dspy.load() 加载此模型进行推理")


if __name__ == "__main__":
    run_optimization()