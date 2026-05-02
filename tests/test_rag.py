"""RAG 模块单元测试：文档加载、切分、检索"""

import os
import tempfile

import pytest
from langchain_core.documents import Document

from src.rag.splitter import split_documents, CHINESE_SEPARATORS
from src.rag.retriever import HybridRetriever, _tokenize_chinese


# ===== 文档切分测试 =====

class TestSplitter:
    def test_split_short_text(self):
        """短文本不应被切分"""
        doc = Document(page_content="这是一段短文本", metadata={"source": "test.md"})
        chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 1
        assert chunks[0].page_content == "这是一段短文本"

    def test_split_long_chinese_text(self):
        """长中文文本应被切分为多个块"""
        text = "这是第一段内容。" * 100  # 约 800 字
        doc = Document(page_content=text, metadata={"source": "test.md"})
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1

    def test_split_preserves_metadata(self):
        """切分后应保留原始元数据"""
        doc = Document(page_content="内容。" * 200, metadata={"source": "doc.pdf", "page": 1})
        chunks = split_documents([doc], chunk_size=200, chunk_overlap=20)
        for chunk in chunks:
            assert chunk.metadata["source"] == "doc.pdf"

    def test_chinese_separators_defined(self):
        """验证中文分隔符列表已定义"""
        assert "\n\n" in CHINESE_SEPARATORS
        assert "。" in CHINESE_SEPARATORS
        assert "！" in CHINESE_SEPARATORS

    def test_split_multiple_documents(self):
        """多个文档应分别切分"""
        docs = [
            Document(page_content="文档一内容。" * 100, metadata={"source": "a.md"}),
            Document(page_content="文档二内容。" * 100, metadata={"source": "b.md"}),
        ]
        chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 2
        sources = {c.metadata["source"] for c in chunks}
        assert "a.md" in sources
        assert "b.md" in sources


# ===== 文档加载测试 =====

class TestLoader:
    def test_load_markdown_file(self):
        """测试 Markdown 文件加载"""
        from src.rag.loader import load_document

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
            f.write("# 标题\n\n这是正文内容。")
            f.flush()
            docs = load_document(f.name)
        os.unlink(f.name)

        assert len(docs) >= 1
        assert "正文内容" in docs[0].page_content
        assert docs[0].metadata["type"] == "markdown"

    def test_load_unsupported_format(self):
        """不支持的格式应抛出异常"""
        from src.rag.loader import load_document

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            f.flush()
            with pytest.raises(ValueError, match="不支持的文件格式"):
                load_document(f.name)
        os.unlink(f.name)

    def test_load_nonexistent_file(self):
        """不存在的文件应抛出 FileNotFoundError"""
        from src.rag.loader import load_document

        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/file.pdf")


# ===== 中文分词测试 =====

class TestTokenizeChinese:
    def test_basic_chinese(self):
        """中文字符应逐字切分"""
        tokens = _tokenize_chinese("你好世界")
        assert tokens == ["你", "好", "世", "界"]

    def test_mixed_text(self):
        """中英混合文本应正确处理"""
        tokens = _tokenize_chinese("hello世界")
        assert "世" in tokens
        assert "界" in tokens

    def test_whitespace_removed(self):
        """空白字符应被去除"""
        tokens = _tokenize_chinese("你 好")
        assert " " not in tokens
        assert tokens == ["你", "好"]


# ===== 混合检索测试 =====

class TestHybridRetriever:
    @pytest.fixture
    def retriever_setup(self):
        """构建模拟的 HybridRetriever"""
        from unittest.mock import MagicMock

        docs = [
            Document(page_content="Python 是一种编程语言，广泛用于数据科学和人工智能。", metadata={"source": "intro.md"}),
            Document(page_content="Java 是一种面向对象的编程语言，常用于企业级应用。", metadata={"source": "intro.md"}),
            Document(page_content="机器学习是人工智能的一个分支，通过数据训练模型。", metadata={"source": "ai.md"}),
            Document(page_content="深度学习使用神经网络来处理复杂的模式识别问题。", metadata={"source": "ai.md"}),
            Document(page_content="自然语言处理让计算机理解和生成人类语言。", metadata={"source": "nlp.md"}),
        ]

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = docs[:3]

        mock_embeddings = MagicMock()

        retriever = HybridRetriever(
            vectorstore=mock_vectorstore,
            documents=docs,
            embeddings=mock_embeddings,
            top_k=3,
        )
        return retriever, docs

    def test_retrieve_returns_documents(self, retriever_setup):
        """检索应返回文档列表"""
        retriever, _ = retriever_setup
        results = retriever.retrieve("编程语言")
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_retrieve_uses_vector_and_bm25(self, retriever_setup):
        """检索应同时使用向量检索和 BM25"""
        retriever, _ = retriever_setup
        retriever.retrieve("人工智能")
        retriever.vectorstore.similarity_search.assert_called_once()

    def test_bm25_search_returns_ranked(self, retriever_setup):
        """BM25 检索应返回带排名的结果"""
        retriever, _ = retriever_setup
        results = retriever._bm25_search("编程语言")
        assert len(results) > 0
        for doc, rank in results:
            assert isinstance(doc, Document)
            assert rank >= 1
