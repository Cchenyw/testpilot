"""
5.RAGAS评估
"""
"""RAGAS 评估"""
import sys, json
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *
from rag.pipeline import TestPilotRAGPipeline

console = Console()


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


def run_evaluation():
    console.print("\n[bold cyan]📊 RAGAS 评估[/bold cyan]\n")
    pipeline = TestPilotRAGPipeline()
    questions = load_test_questions()
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for q in questions:
        r = pipeline.ask(q["question"])
        data["question"].append(q["question"])
        data["answer"].append(r["answer"])
        data["contexts"].append([c["content"] for c in r.get("context", [])])
        data["ground_truth"].append(q["ground_truth"])

    dataset = Dataset.from_dict(data)
    scores = evaluate(dataset, metrics=[context_precision, context_recall,
                                         faithfulness, answer_relevancy])

    table = Table(title="RAGAS 结果")
    table.add_column("指标", style="cyan")
    table.add_column("分数", style="green")
    table.add_column("说明", style="dim")
    for metric, desc in [
        ("context_precision", "检索精度：命中几个相关的"),
        ("context_recall", "检索召回：相关文档找回几个"),
        ("faithfulness", "忠实度：回答是否基于文档"),
        ("answer_relevancy", "相关性：回答是否切题"),
    ]:
        if metric in scores:
            table.add_row(metric, f"{scores[metric]:.4f}", desc)
    console.print(table)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "evaluation_report.json").write_text(
        json.dumps({k: float(v) if hasattr(v, 'item') else v
                     for k, v in scores.items()}, indent=2, ensure_ascii=False))
    console.print(f"\n📁 报告: {EVAL_DIR / 'evaluation_report.json'}")
    return scores


if __name__ == "__main__":
    run_evaluation()