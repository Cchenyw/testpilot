"""第6步修复版 v2：直接下载 .rst / .txt / .pdf 文档源文件

思路：
  pytest  → GitHub raw .rst 源文件（纯文本，TextLoader 原生支持）
  Selenium → ReadTheDocs 官方 PDF
  unittest → Python 官方文档提取纯文本
"""
import requests
from pathlib import Path

HEADERS = {"User-Agent": "TestPilot/1.0"}
TIMEOUT = 30

# ═══════════════════════════════════════════════════
# pytest — GitHub 仓库真实路径（已通过 API 验证）
#   doc/en/           → 根级 .rst
#   doc/en/how-to/    → how-to 指南
#   doc/en/reference/ → API 参考
# ═══════════════════════════════════════════════════
PYTEST_RAW = "https://raw.githubusercontent.com/pytest-dev/pytest/main/doc/en"

PYTEST_FILES = [
    # 根级
    "getting-started.rst",
    "fixture.rst",
    # how-to/
    "how-to/fixtures.rst",
    "how-to/parametrize.rst",
    "how-to/monkeypatch.rst",
    "how-to/mark.rst",
    "how-to/skipping.rst",
    "how-to/assert.rst",
    "how-to/capture-warnings.rst",
    "how-to/capture-stdout-stderr.rst",
    "how-to/usage.rst",
    "how-to/logging.rst",
    "how-to/tmp_path.rst",
    "how-to/plugins.rst",
    "how-to/unittest.rst",
    "how-to/cache.rst",
    "how-to/writing_hook_functions.rst",
    "how-to/writing_plugins.rst",
    # reference/
    "reference/reference.rst",
    "reference/fixtures.rst",
    "reference/customize.rst",
    "reference/exit-codes.rst",
]

# ═══════════════════════════════════════════════════
# Selenium — ReadTheDocs PDF
# ═══════════════════════════════════════════════════
SELENIUM_PDF = (
    "https://selenium-python.readthedocs.io/_/downloads/en/latest/pdf/"
)


def download(url: str, dest: Path) -> bool:
    """下载单个文件，返回成功/失败"""
    try:
        resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"  ❌ {dest.name}: {e}")
        return False


# ═══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("📥 TestPilot 文档下载 v2（直接获取 .rst / .pdf / .txt）")
    print("=" * 60)

    # ——— pytest .rst → 存为 .txt（TextLoader 原生支持，ingest.py 无需改）———
    pytest_dir = Path("data/docs/pytest")
    pytest_dir.mkdir(parents=True, exist_ok=True)
    print("\n📦 [pytest] GitHub .rst 源文件 → 存为 .txt...")
    ok = 0
    for path in PYTEST_FILES:
        url = f"{PYTEST_RAW}/{path}"
        # 保留子目录结构，扩展名改为 .txt
        dest = pytest_dir / path.replace(".rst", ".txt")
        if download(url, dest):
            print(f"  ✅ {path} → {dest.relative_to(pytest_dir)} ({dest.stat().st_size:,} bytes)")
            ok += 1
    print(f"  📊 {ok}/{len(PYTEST_FILES)}")

    # ——— Selenium PDF ———
    sel_dir = Path("data/docs/selenium")
    sel_dir.mkdir(parents=True, exist_ok=True)
    print("\n📦 [Selenium] ReadTheDocs PDF...")
    dest = sel_dir / "selenium-python.pdf"
    if download(SELENIUM_PDF, dest):
        print(f"  ✅ selenium-python.pdf ({dest.stat().st_size:,} bytes)")

    # ——— unittest 纯文本 ———
    uni_dir = Path("data/docs/unittest")
    uni_dir.mkdir(parents=True, exist_ok=True)
    print("\n📦 [unittest] Python 官方文档...")
    try:
        import re
        resp = requests.get(
            "https://docs.python.org/3/library/unittest.html",
            timeout=TIMEOUT, headers=HEADERS,
        )
        resp.raise_for_status()
        text = resp.text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        (uni_dir / "unittest.txt").write_text(text, encoding="utf-8")
        print(f"  ✅ unittest.txt ({(uni_dir/'unittest.txt').stat().st_size:,} bytes)")
    except Exception as e:
        print(f"  ❌ unittest.txt: {e}")

    # ——— 汇总 ———
    print("\n" + "=" * 60)
    print("📁 data/docs/ 内容:")
    total = size_sum = 0
    for ext in ["*.txt", "*.pdf"]:
        for f in Path("data/docs").rglob(ext):
            s = f.stat().st_size
            total += 1
            size_sum += s
            print(f"  {f.relative_to('data/docs')} ({s:,} bytes)")

    print(f"\n✅ 总计 {total} 个文件, {size_sum:,} bytes (~{size_sum/1024:.0f} KB)")
    print("   → .rst  → TextLoader 原生支持")
    print("   → .pdf  → PyPDFLoader 原生支持")
    print("   → .txt  → TextLoader 原生支持")
    print("   → 无需 HTML 转码，直接 python rag/ingest.py ✅")