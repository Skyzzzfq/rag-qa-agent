"""Agent 模块单元测试：工具调用、Agent 执行"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agents.tools import calculate, list_documents, search_knowledge_base, summarize_document


# ===== 计算工具测试 =====

class TestCalculate:
    def test_simple_addition(self):
        assert "12" in calculate("5 + 7")

    def test_multiplication(self):
        assert "14" in calculate("2 * 7")

    def test_complex_expression(self):
        result = calculate("2 + 3 * 4")
        assert "14" in result

    def test_float_calculation(self):
        result = calculate("1.5 + 2.5")
        assert "4" in result

    def test_unsafe_expression_blocked(self):
        """不允许函数调用"""
        result = calculate("print('hello')")
        assert "不安全" in result

    def test_attribute_access_blocked(self):
        """不允许属性访问"""
        result = calculate("''.__class__")
        assert "不安全" in result

    def test_name_access_blocked(self):
        """不允许变量名（除 True/False/None）"""
        result = calculate("os.system('rm -rf /')")
        assert "不安全" in result

    def test_boolean_constants_allowed(self):
        """True/False/None 应被允许"""
        result = calculate("True + True")
        assert "2" in result

    def test_invalid_expression(self):
        """无效表达式应返回错误"""
        result = calculate("abc + def")
        assert "不安全" in result or "失败" in result


# ===== 知识库搜索工具测试 =====

class TestSearchKnowledgeBase:
    def test_search_without_retriever(self):
        """未初始化检索器时应返回提示"""
        from src.agents import tools
        original = tools._hybrid_retriever
        tools._hybrid_retriever = None
        result = search_knowledge_base.invoke({"query": "测试"})
        assert "尚未初始化" in result
        tools._hybrid_retriever = original

    def test_search_with_results(self):
        """检索器返回结果时应格式化输出"""
        from src.agents import tools

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            Document(page_content="测试内容", metadata={"source": "test.md", "page": 1}),
        ]
        original = tools._hybrid_retriever
        tools._hybrid_retriever = mock_retriever

        result = search_knowledge_base.invoke({"query": "测试"})
        assert "测试内容" in result
        assert "test.md" in result

        tools._hybrid_retriever = original

    def test_search_no_results(self):
        """检索器无结果时应返回提示"""
        from src.agents import tools

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        original = tools._hybrid_retriever
        tools._hybrid_retriever = mock_retriever

        result = search_knowledge_base.invoke({"query": "不存在的内容"})
        assert "未找到" in result

        tools._hybrid_retriever = original


# ===== 文档列表工具测试 =====

class TestListDocuments:
    def test_list_with_documents(self):
        """有文档时应列出文件名"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_data_dir = settings.data_dir
            settings.data_dir = tmpdir

            # 创建测试文件
            open(os.path.join(tmpdir, "test.md"), "w").close()
            open(os.path.join(tmpdir, "doc.pdf"), "w").close()
            open(os.path.join(tmpdir, "ignore.txt"), "w").close()

            result = list_documents.invoke({})
            assert "test.md" in result
            assert "doc.pdf" in result
            assert "ignore.txt" not in result

            settings.data_dir = original_data_dir

    def test_list_empty_directory(self):
        """空目录应返回提示"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_data_dir = settings.data_dir
            settings.data_dir = tmpdir

            result = list_documents.invoke({})
            assert "暂无文档" in result

            settings.data_dir = original_data_dir


# ===== 文档摘要工具测试 =====

class TestSummarizeDocument:
    def test_nonexistent_document(self):
        """不存在的文档应返回错误"""
        result = summarize_document.invoke({"doc_name": "nonexistent.pdf"})
        assert "不存在" in result

    @patch("src.agents.tools.ChatOpenAI")
    def test_summarize_success(self, mock_llm_class):
        """正常文档应返回摘要"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "这是摘要内容"
        mock_llm_class.return_value = mock_llm

        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_data_dir = settings.data_dir
            settings.data_dir = tmpdir

            # 创建测试 Markdown 文件
            with open(os.path.join(tmpdir, "test.md"), "w", encoding="utf-8") as f:
                f.write("# 测试文档\n\n这是测试内容。")

            with patch("src.agents.tools.load_document") as mock_load:
                mock_load.return_value = [
                    Document(page_content="这是测试内容。", metadata={"source": "test.md"})
                ]
                result = summarize_document.invoke({"doc_name": "test.md"})
                assert "摘要" in result

            settings.data_dir = original_data_dir


# ===== QAAgent 测试 =====

class TestQAAgent:
    def test_agent_not_initialized(self):
        """未初始化的 Agent 应抛出异常"""
        from src.agents.qa_agent import QAAgent
        agent = QAAgent()
        agent.agent_executor = None
        with pytest.raises(ValueError, match="Agent 未初始化"):
            agent.run("测试问题")

    def test_update_retriever(self):
        """update_retriever 应调用全局设置"""
        from src.agents.qa_agent import QAAgent

        with patch("src.agents.qa_agent.set_retriever") as mock_set_ret, \
             patch("src.agents.qa_agent.set_documents") as mock_set_docs:
            agent = QAAgent()
            mock_retriever = MagicMock()
            docs = [Document(page_content="test", metadata={})]
            agent.update_retriever(mock_retriever, docs)
            mock_set_ret.assert_called_once_with(mock_retriever)
            mock_set_docs.assert_called_once_with(docs)
