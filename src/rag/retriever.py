from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize_chinese(text: str) -> list[str]:
    """简单的中文分词：按字符级切分，适合 BM25 关键词匹配"""
    # 去除空白字符，逐字符切分
    return [char for char in text if char.strip()]


class HybridRetriever:
    """混合检索器：向量检索 + BM25 关键词检索，RRF 合并排序"""

    def __init__(
        self,
        vectorstore: FAISS,
        documents: list[Document],
        embeddings: Embeddings,
        top_k: int | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.top_k = top_k or settings.retrieval_top_k
        self.rrf_k = rrf_k

        # 构建 BM25 索引
        self.documents = documents
        tokenized_corpus = [_tokenize_chinese(doc.page_content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"HybridRetriever 初始化完成, 文档数: {len(documents)}, top_k: {self.top_k}")

    def _vector_search(self, query: str) -> list[tuple[Document, int]]:
        """向量检索，返回 (Document, rank) 列表，rank 从 1 开始"""
        fetch_k = self.top_k * 2
        results = self.vectorstore.similarity_search(query, k=fetch_k)
        return [(doc, rank) for rank, doc in enumerate(results, start=1)]

    def _bm25_search(self, query: str) -> list[tuple[Document, int]]:
        """BM25 关键词检索，返回 (Document, rank) 列表，rank 从 1 开始"""
        tokenized_query = _tokenize_chinese(query)
        scores = self.bm25.get_scores(tokenized_query)
        # 按分数降序排列，取 top_k * 2
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[: self.top_k * 2]
        return [(self.documents[i], rank) for rank, i in enumerate(ranked_indices, start=1)]

    def retrieve(self, query: str) -> list[Document]:
        """混合检索：向量 + BM25，RRF 合并排序，返回 top_k 结果"""
        # 1. 向量检索
        vector_results = self._vector_search(query)
        # 2. BM25 检索
        bm25_results = self._bm25_search(query)

        # 3. RRF 合并：对每个文档计算 RRF 分
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for doc, rank in vector_results:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.page_content[:64]}"
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[doc_id] = doc

        for doc, rank in bm25_results:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.page_content[:64]}"
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            doc_map[doc_id] = doc

        # 4. 按 RRF 分排序，取 top_k
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: self.top_k]
        results = [doc_map[doc_id] for doc_id in sorted_ids]

        logger.info(f"混合检索完成: query='{query}', 向量结果={len(vector_results)}, "
                    f"BM25结果={len(bm25_results)}, 合并后={len(results)}")
        return results
