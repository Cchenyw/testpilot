"""
5.RAGAS评估 (v0.4.3 适配)
"""
# [CHANGED] ① 不再需要 dspy — 用 OpenAI 兼容 client 替代
import asyncio                                                           # ← 新增：异步支持
from openai import AsyncOpenAI                                           # ← 新增：LLM client
from ragas.llms import llm_factory                                       # ← 替代 LangchainLLMWrapper
# [CHANGED] ③ LangchainEmbeddingsWrapper 已废弃 → 用内置 HuggingFaceEmbeddings
from ragas.embeddings import HuggingFaceEmbeddings                       # ← 替代 LangchainEmbeddingsWrapper

"""RAGAS 评估"""
import sys, json
from pathlib import Path
# [CHANGED] ② evaluate 已废弃 → 改用 experiment
from ragas import Dataset, experiment                                    # ← evaluate → experiment
# [CHANGED] ③ 指标从 collections 导入（类，非实例）
from ragas.metrics.collections import Faithfulness     # 忠诚度
from ragas.metrics.collections import AnswerRelevancy  # 相关性
from ragas.metrics.collections import ContextPrecision # 检索精度
from ragas.metrics.collections import ContextRecall    # 召回率

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *
from rag.pipeline import TestPilotRAGPipeline

console = Console()


# [CHANGED] ④ LLM 初始化：llm_factory() 替代 LangchainLLMWrapper(dspy.LM(...))
evaluator_llm = llm_factory(
    LLM_MODEL_NAME,
    client=AsyncOpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_API_BASE,
    ),
)

# [CHANGED] ④+③ Embeddings：HuggingFaceEmbeddings 替代 LangchainEmbeddingsWrapper(FlagModel)
evaluator_embeddings = HuggingFaceEmbeddings(
    model=EMBEDDING_MODEL,
    normalize_embeddings=True,
)


# [CHANGED] ⑤ 指标实例化 — llm/embeddings 在构造时绑定
metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    ContextPrecision(llm=evaluator_llm),
    ContextRecall(llm=evaluator_llm),
]

# [CHANGED] 每个指标的 ascore() 参数签名不同，硬编码映射
_METRIC_PARAMS = {
    Faithfulness:      ["user_input", "response", "retrieved_contexts"],
    AnswerRelevancy:   ["user_input", "response"],
    ContextPrecision:  ["user_input", "reference", "retrieved_contexts"],
    ContextRecall:     ["user_input", "retrieved_contexts", "reference"],
}


def load_test_questions():
    test_file = PROJECT_ROOT / "tests" / "test_questions.json"
    if test_file.exists():
        return json.loads(test_file.read_text(encoding="utf-8"))
    return [
        {"question": "pytest 中 fixture 的作用域有哪些？",
         "ground_truth": "pytest fixture 有 function、class、module、package、session 五个作用域。"},
        {"question": "如何用 pytest.mark.parametrize 进行参数化测试？",
         "ground_truth": "使用 @pytest.mark.parametrize 装饰器，传入参数名和值列表。"},
        {"question": "Selenium 中 find_element 和 find_elements 的区别？",
         "ground_truth": "find_element 返回第一个匹配元素，找不到抛异常；find_elements 返回列表，找不到返回空列表。"},
        {"question": "pytest 如何跳过某个测试？",
         "ground_truth": "@pytest.mark.skip 无条件跳过，@pytest.mark.skipif 条件跳过。"},
        {"question": "pytest monkeypatch 的 setattr 方法有什么用？",
         "ground_truth": "monkeypatch.setattr 临时替换对象属性，测试后自动恢复，常用于 mock。"},
    ]


# [CHANGED] ⑥ Pipeline 提到函数外，避免每次调用重新初始化
pipeline = TestPilotRAGPipeline()


# [CHANGED] ⑦ 评估函数：run_evaluation() → @experiment + async
@experiment()
async def run_experiment(row):
    r = pipeline.ask(row["question"])
    contexts = [c["content"] for c in r.get("context", [])]
    answer = r["answer"]

    # [CHANGED] 逐指标评分 — 按 _METRIC_PARAMS 映射构建 kwargs
    scores = {}
    for m in metrics:
        param_names = _METRIC_PARAMS.get(type(m), [])
        kwargs = {}
        if "user_input" in param_names:
            kwargs["user_input"] = row["question"]
        if "response" in param_names:
            kwargs["response"] = answer
        if "retrieved_contexts" in param_names:
            kwargs["retrieved_contexts"] = contexts
        if "reference" in param_names:
            kwargs["reference"] = row["ground_truth"]
        result = await m.ascore(**kwargs)
        scores[m.name] = result.value

    return {
        **row,
        "answer": answer,
        "contexts": str(contexts),
        **scores,
    }


# [CHANGED] ⑧ run_evaluation() → async main()
async def main():
    console.print("\n[bold cyan]📊 RAGAS v0.4.3 评估[/bold cyan]\n")

    # [CHANGED] ⑨ Dataset 构建：datasets.Dataset.from_dict → ragas.Dataset
    questions = load_test_questions()
    dataset = Dataset(
        name="testpilot_eval",
        backend="local/csv",
        root_dir=str(EVAL_DIR),
    )
    for q in questions:
        dataset.append({
            "question": q["question"],
            "ground_truth": q["ground_truth"],
        })
    dataset.save()

    # [CHANGED] ⑩ 运行：evaluate() → arun()
    results = await run_experiment.arun(dataset)

    # 保存 CSV
    results.save()
    csv_path = EVAL_DIR / f"{results.name}.csv"
    console.print(f"[green]✓ 结果 CSV: {csv_path}[/green]")

    # [CHANGED] ⑪ 结果表格 — 从 DataTable 提取平均值
    df = results.to_pandas()
    table = Table(title="RAGAS v0.4.3 结果")
    table.add_column("指标", style="cyan")
    table.add_column("平均分", style="green")
    table.add_column("说明", style="dim")
    for col_key, desc in [
        ("faithfulness",            "忠实度：回答是否基于文档"),
        ("answer_relevancy",        "相关性：回答是否切题"),
        ("context_precision",       "检索精度：命中几个相关的"),
        ("context_recall",          "检索召回：相关文档找回几个"),
    ]:
        if col_key in df.columns:
            table.add_row(col_key, f"{df[col_key].mean():.4f}", desc)
    console.print(table)

    # 同时保存 JSON 报告
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report = {col: float(df[col].mean()) for col in df.columns
              if col in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    (EVAL_DIR / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    console.print(f"\n📁 JSON 报告: {EVAL_DIR / 'evaluation_report.json'}")

    return results


if __name__ == "__main__":
    # [CHANGED] ⑫ 入口：asyncio.run() 启动异步 main
    asyncio.run(main())