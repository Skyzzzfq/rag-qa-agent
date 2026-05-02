from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_embeddings_instance: OpenAIEmbeddings | None = None


def get_embeddings() -> OpenAIEmbeddings:
    """获取 Embedding 实例（单例模式），全局共享"""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = OpenAIEmbeddings(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_api_base,
            model=settings.embedding_model_name,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
        )
        logger.info(f"Embedding 实例已创建: {settings.embedding_model_name}")
    return _embeddings_instance
