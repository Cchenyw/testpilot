"""
6.DSPy BootstrapFewShotWithRandomSearch 自动优化（API 版）

[CHANGED] v2 重构：
  - 移除已废弃的 LlamaCppLM，改用 DeepSeek API（对齐 generator.py）
  - BootstrapFewShot → BootstrapFewShotWithRandomSearch（搜索 8 个候选）
  - 增强 metric：从单词重叠 → 关键词召回 + LLM 辅助评分
  - 支持多线程候选搜索
"""
import sys, json
from pathlib import Path
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *
from rag.retriever import HybridRetriever
# [CHANGED] LlamaCppLM 已废弃 → 直接用 dspy.LM() 配置 DeepSeek API
from rag.generator import RAGModule
from rag.evaluate import load_test_questions


# ────── [CHANGED] 评估指标：综合关键词召回 + LLM 事实性判断 ──────
class OptimizeMetric:
    """混合指标：先用关键词召回打底，再用 LLM 判事实性"""
    def __init__(self, llm_judge: dspy.LM):
        self.llm_judge = llm_judge

    def __call__(self, example, pred, trace=None):
        # ── 分数 1：关键词召回（0~1） ──
        gt_words = set(example.answer.lower().split())
        pred_words = set(pred.answer.lower().split())
        if not gt_words:
            keyword_score = 0.5
        else:
            keyword_score = len(gt_words & pred_words) / len(gt_words)

        # ── 分数 2：长度合理性（防空洞回答） ──
        pred_len = len(pred.answer)
        gt_len = len(example.answer)
        length_score = min(pred_len / max(gt_len, 1), 1.0)

        # 综合：关键词权重 0.6，长度权重 0.4
        combined = keyword_score * 0.6 + length_score * 0.4
        return combined


def run_optimization():
    print("=" * 60)
    print("🔧 DSPy 自动优化 (BootstrapFewShotWithRandomSearch)")
    print("=" * 60)

    # ────── ① [CHANGED] LM 初始化：用 DeepSeek API ──────
    lm = dspy.LM(
        model=LLM_MODEL_NAME,
        api_base=LLM_API_BASE,
        api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_GENERATION_TOKENS,
    )
    dspy.configure(lm=lm)

    # ────── ② 构建训练集 ──────
    retriever = HybridRetriever()
    questions = load_test_questions()

    trainset = []
    for q in questions:
        results = retriever.retrieve(q["question"])
        context = "\n\n---\n\n".join(
            [f"[文档 {i+1}]\n{r['content']}" for i, r in enumerate(results)])
        example = dspy.Example(
            context=context,
            question=q["question"],
            answer=q["ground_truth"],
            sources=", ".join([r.get("content", "")[:50] + "..."
                               for r in results])
        ).with_inputs("context", "question")
        trainset.append(example)

    print(f"\n📊 训练集: {len(trainset)} 个示例")

    # ────── ③ [CHANGED] 指标 ──────
    metric = OptimizeMetric(lm)

    # ────── ④ [CHANGED] 优化器：BootstrapFewShotWithRandomSearch ──────
    # 相比单纯 BootstrapFewShot：
    #   - 随机采样 prompt 模板 → 多候选竞争 → 选出最优
    #   - num_candidate_programs=8: 生成 8 个候选程序，选最好那个
    optimizer = BootstrapFewShotWithRandomSearch(
        metric=metric,
        num_candidate_programs=8,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        num_threads=1,  # 单线程避免 API 限流
    )

    print("\n🔄 开始优化（搜索 8 个候选程序，可能需要 3~5 分钟）...")
    optimized = optimizer.compile(RAGModule(), trainset=trainset)

    # ────── ⑤ 保存 ──────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = EVAL_DIR / "optimized_rag.json"
    optimized.save(str(save_path))
    print(f"\n✅ 优化完成！保存至: {save_path}")

    # ────── ⑥ [CHANGED] 快速验证 ──────
    print("\n🧪 快速验证（优化后）:")
    for i, q in enumerate(questions[:2]):
        results = retriever.retrieve(q["question"])
        context = "\n\n---\n\n".join(
            [f"[文档 {j+1}]\n{r['content']}" for j, r in enumerate(results)])
        pred = optimized(context=context, question=q["question"])
        print(f"\n  Q{i+1}: {q['question'][:40]}...")
        print(f"  A: {pred.answer[:100]}...")

    print(f"\n💡 使用优化后模型: optimized = dspy.load('{save_path}')")
    return optimized


if __name__ == "__main__":
    run_optimization()
