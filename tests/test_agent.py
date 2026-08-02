"""Agent 模块单元测试：工具调用、Agent 执行"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.agents.tools import calculate, list_documents, search_knowledge_base, summarize_document


# ===== 计算工具测试 =====

class TestCalculate:
    def test_simple_addition(self):
        assert "12" in calculate.invoke({"expression": "5 + 7"})

    def test_multiplication(self):
        assert "14" in calculate.invoke({"expression": "2 * 7"})

    def test_complex_expression(self):
        result = calculate.invoke({"expression": "2 + 3 * 4"})
        assert "14" in result

    def test_float_calculation(self):
        result = calculate.invoke({"expression": "1.5 + 2.5"})
        assert "4" in result

    def test_unsafe_expression_blocked(self):
        """不允许函数调用"""
        result = calculate.invoke({"expression": "print('hello')"})
        assert "不安全" in result

    def test_attribute_access_blocked(self):
        """不允许属性访问"""
        result = calculate.invoke({"expression": "''.__class__"})
        assert "不安全" in result

    def test_name_access_blocked(self):
        """不允许变量名（除 True/False/None）"""
        result = calculate.invoke({"expression": "os.system('rm -rf /')"})
        assert "不安全" in result

    def test_boolean_constants_allowed(self):
        """True/False/None 应被允许"""
        result = calculate.invoke({"expression": "True + True"})
        assert "2" in result

    def test_invalid_expression(self):
        """无效表达式应返回错误"""
        result = calculate.invoke({"expression": "abc + def"})
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
            open(os.path.join(tmpdir, "note.txt"), "w").close()
            open(os.path.join(tmpdir, "ignore.xlsx"), "w").close()

            result = list_documents.invoke({})
            assert "test.md" in result
            assert "doc.pdf" in result
            assert "note.txt" in result
            assert "ignore.xlsx" not in result

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
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_data_dir = settings.data_dir
            settings.data_dir = tmpdir

            result = summarize_document.invoke({"doc_name": "nonexistent.pdf"})
            assert "不存在" in result

            settings.data_dir = original_data_dir

    def test_rejects_document_outside_data_directory(self):
        """摘要工具不能读取数据目录之外的文件。"""
        result = summarize_document.invoke({"doc_name": "../secret.txt"})
        assert "无效的文档名" in result

    def test_summarize_success(self):
        """正常文档应返回摘要"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.config import settings
            original_data_dir = settings.data_dir
            settings.data_dir = tmpdir

            with open(os.path.join(tmpdir, "test.md"), "w", encoding="utf-8") as f:
                f.write("# 测试文档\n\n这是测试内容。")

            mock_llm = MagicMock()
            mock_llm.invoke.return_value.content = "这是摘要内容"

            with patch("langchain_openai.ChatOpenAI", return_value=mock_llm), \
                 patch("src.rag.loader.load_document") as mock_load:
                mock_load.return_value = [
                    Document(page_content="这是测试内容。", metadata={"source": "test.md"})
                ]
                result = summarize_document.invoke({"doc_name": "test.md"})
                assert "摘要" in result

            settings.data_dir = original_data_dir


# ===== QAAgent 测试 =====

class TestQAAgent:
    @patch("src.agents.qa_agent.ChatOpenAI")
    def test_agent_not_initialized(self, mock_llm_class):
        """未初始化的 Agent 应抛出异常"""
        from src.agents.qa_agent import QAAgent
        agent = QAAgent()
        agent._executor = None
        with pytest.raises(ValueError, match="Agent 未初始化"):
            agent.run("测试问题")

    @patch("src.agents.qa_agent.ChatOpenAI")
    def test_update_retriever(self, mock_llm_class):
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

    @patch("src.agents.qa_agent.ChatOpenAI")
    def test_setup_agent_creates_executor(self, mock_llm_class):
        """setup_agent 应创建 AgentExecutor 并包含工具"""
        from src.agents.qa_agent import QAAgent
        agent = QAAgent()
        assert agent._executor is None
        agent.setup_agent()
        assert agent._executor is not None
        assert len(agent._executor.tools) == 4
        assert agent._executor.return_intermediate_steps is True

    def test_extract_sources(self):
        """_extract_sources 应从中间步骤中提取来源"""
        from src.agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        steps = [
            (MagicMock(tool="search_knowledge_base"),
             "[1] 来源: doc1.md (第1页)\n内容1"),
            (MagicMock(tool="calculate"),
             "计算结果: 42"),
            (MagicMock(tool="search_knowledge_base"),
             "[1] 来源: doc2.md\n内容2"),
        ]
        sources = agent._extract_sources(steps)
        assert "doc1.md" in sources
        assert "doc2.md" in sources
        assert len(sources) == 2

    def test_build_context_from_steps(self):
        """_build_context_from_steps 应拼接知识库检索结果"""
        from src.agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        steps = [
            (MagicMock(tool="search_knowledge_base"), "检索结果1"),
            (MagicMock(tool="calculate"), "计算结果: 42"),
            (MagicMock(tool="search_knowledge_base"), "检索结果2"),
        ]
        context = agent._build_context_from_steps(steps)
        assert "检索结果1" in context
        assert "检索结果2" in context
        assert "42" not in context

    @patch("src.agents.qa_agent.ChatOpenAI")
    def test_run_with_executor(self, mock_llm_class):
        """run() 应使用 executor 并提取来源"""
        from src.agents.qa_agent import QAAgent

        agent = QAAgent()
        mock_result = {
            "output": "Python是一种编程语言",
            "intermediate_steps": [
                (MagicMock(tool="search_knowledge_base"),
                 "[1] 来源: intro.md\nPython 是一种编程语言"),
            ],
        }
        agent._executor = MagicMock()
        agent._executor.invoke.return_value = mock_result
        agent._reflect = MagicMock(return_value=(True, ""))

        result = agent.run("什么是Python")
        assert result["answer"] == "Python是一种编程语言"
        assert "intro.md" in result["sources"]

    @patch("src.agents.qa_agent.ChatOpenAI")
    def test_async_run_uses_async_reflection(self, mock_llm_class):
        """异步问答必须走异步反思并保留来源。"""
        from src.agents.qa_agent import QAAgent

        agent = QAAgent()
        agent._executor = MagicMock()
        agent._executor.ainvoke = AsyncMock(return_value={
            "output": "异步回答",
            "intermediate_steps": [
                (MagicMock(tool="search_knowledge_base"), "[1] 来源: async.md\n内容"),
            ],
        })
        agent._areflect = AsyncMock(return_value=(True, ""))

        result = asyncio.run(agent.arun("异步问题"))

        assert result == {"answer": "异步回答", "sources": ["async.md"]}
        agent._areflect.assert_awaited_once()

    def test_stream_events_include_sources_and_complete_answer(self):
        """流式事件必须来自完成反思后的最终结果。"""
        from src.agents.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        agent.arun = AsyncMock(return_value={
            "answer": "这是最终回答",
            "sources": ["source.md"],
        })

        async def collect_events():
            return [event async for event in agent.astream_events("问题")]

        events = asyncio.run(collect_events())
        assert events[0] == {"type": "sources", "data": ["source.md"]}
        assert "".join(event["data"] for event in events[1:]) == "这是最终回答"
