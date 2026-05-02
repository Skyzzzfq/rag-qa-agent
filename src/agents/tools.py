import ast
import os

from langchain_core.documents import Document
from langchain_core.tools import tool

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 全局引用，由应用启动时设置
_hybrid_retriever = None
_documents: list[Document] = []


def set_retriever(retriever) -> None:
    """设置全局混合检索器实例"""
    global _hybrid_retriever
    _hybrid_retriever = retriever


def set_documents(documents: list[Document]) -> None:
    """设置全局文档列表"""
    global _documents
    _documents = documents


@tool
def search_knowledge_base(query: str) -> str:
    """在知识库中搜索与问题相关的文档内容。参数 query: 搜索关键词"""
    if _hybrid_retriever is None:
        return "知识库尚未初始化，请先上传文档。"
    try:
        results = _hybrid_retriever.retrieve(query)
        if not results:
            return "未找到相关文档内容。"
        content_parts = []
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            page_info = f" (第{page}页)" if page else ""
            content_parts.append(f"[{i}] 来源: {source}{page_info}\n{doc.page_content}")
        return "\n\n---\n\n".join(content_parts)
    except Exception as e:
        logger.error(f"知识库搜索失败: {e}")
        return f"搜索失败: {e}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式的值。参数 expression: 数学表达式，如 '2 + 3 * 4'"""
    try:
        # 只允许安全表达式，禁止函数调用和属性访问
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Call, ast.Attribute, ast.Name)):
                # 允许常量名称如 True/False/None
                if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
                    continue
                return f"不安全的表达式: {expression}"
        result = eval(compile(tree, "<calculate>", "eval"))
        return f"计算结果: {result}"
    except Exception as e:
        logger.error(f"计算失败: expression='{expression}', 错误: {e}")
        return f"计算失败: {e}"


@tool
def list_documents() -> str:
    """列出知识库中已加载的文档列表"""
    data_dir = settings.data_dir
    if not os.path.exists(data_dir):
        return "文档目录不存在。"
    files = os.listdir(data_dir)
    supported = [f for f in files if os.path.splitext(f)[1].lower() in {".pdf", ".md", ".markdown"}]
    if not supported:
        return "知识库中暂无文档。"
    return "已加载的文档:\n" + "\n".join(f"  - {f}" for f in supported)


@tool
def summarize_document(doc_name: str) -> str:
    """总结指定文档的主要内容。参数 doc_name: 文档名"""
    file_path = os.path.join(settings.data_dir, doc_name)
    if not os.path.exists(file_path):
        return f"文档不存在: {doc_name}"

    try:
        from src.rag.loader import load_document
        from langchain_openai import ChatOpenAI
        from src.config import settings as cfg

        documents = load_document(file_path)
        full_text = "\n\n".join(doc.page_content for doc in documents)
        if len(full_text) > 4000:
            full_text = full_text[:4000] + "...(已截断)"

        llm = ChatOpenAI(
            api_key=cfg.llm_api_key,
            base_url=cfg.llm_api_base,
            model=cfg.llm_model_name,
            temperature=0,
        )
        prompt = f"请用中文总结以下文档的主要内容:\n\n{full_text}"
        summary = llm.invoke(prompt).content
        return f"文档《{doc_name}》摘要:\n{summary}"
    except Exception as e:
        logger.error(f"文档摘要失败: {doc_name}, 错误: {e}")
        return f"生成摘要失败: {e}"
