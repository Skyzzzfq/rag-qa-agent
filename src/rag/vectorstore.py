import os

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreManager:
    """FAISS 向量存储管理，支持创建、保存、加载和增量添加文档"""

    def __init__(self) -> None:
        self.vectorstore: FAISS | None = None

    def create_from_documents(
        self, documents: list[Document], embeddings: Embeddings
    ) -> FAISS:
        """从文档创建 FAISS 索引"""
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        logger.info(f"FAISS 索引创建完成，文档数: {len(documents)}")
        return self.vectorstore

    def save_index(self, path: str) -> None:
        """保存索引到本地"""
        if self.vectorstore is None:
            logger.warning("无索引可保存")
            return
        os.makedirs(path, exist_ok=True)
        self.vectorstore.save_local(path)
        logger.info(f"FAISS 索引已保存到: {path}")

    def load_index(self, path: str, embeddings: Embeddings) -> FAISS | None:
        """从本地加载索引，索引不存在时返回 None"""
        if not os.path.exists(path) or not os.listdir(path):
            logger.info(f"索引目录为空或不存在: {path}")
            return None
        try:
            self.vectorstore = FAISS.load_local(
                path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS 索引已加载: {path}")
            return self.vectorstore
        except Exception as e:
            logger.error(f"FAISS 索引加载失败: {e}")
            return None

    def add_documents(self, documents: list[Document]) -> None:
        """向已有索引增量添加文档"""
        if self.vectorstore is None:
            raise ValueError("索引未初始化，请先创建或加载索引")
        self.vectorstore.add_documents(documents)
        logger.info(f"增量添加文档完成，新增: {len(documents)} 个块")
