from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """从 .env 文件读取全局配置"""

    # LLM 配置
    llm_api_key: str = ""
    llm_api_base: str = ""
    llm_model_name: str = ""

    # Embedding 配置
    embedding_api_key: str = ""
    embedding_api_base: str = ""
    embedding_model_name: str = ""

    # 存储路径
    faiss_index_path: str = "./indexes"
    data_dir: str = "./data"

    # RAG 参数
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5

    # 对话记忆
    conversation_window: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
