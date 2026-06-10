"""
DSpy RAG生成
"""
"""DSPy RAG 生成（API 版）"""
import sys
from pathlib import Path
import dspy

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.config import *

# ===== 就这三行，搞定 LLM =====
# DSPy 内置对 OpenAI 兼容 API 的支持，不需要手写任何适配器
lm = dspy.LM(
    model=LLM_MODEL_NAME,
    api_base=LLM_API_BASE,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_tokens=MAX_GENERATION_TOKENS,
)


# ===== DSPy 签名（和之前完全一样）=====
class TestPilotRAG(dspy.Signature):
    """测试开发知识问答
    要求：基于文档回答、给代码示例、不知道就诚实说"""
    context = dspy.InputField(desc="检索到的相关文档片段")
    question = dspy.InputField(desc="用户关于测试框架的问题")
    answer = dspy.OutputField(desc="基于文档的回答，含代码示例")
    sources = dspy.OutputField(desc="引用的文档来源")


# ===== RAG 模块（不变）=====
class RAGModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(TestPilotRAG)

    def forward(self, context, question):
        return self.generate(context=context, question=question)


# ===== 工厂函数 =====
def create_rag_module():
    dspy.configure(lm=lm)
    return RAGModule()
