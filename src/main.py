from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.api import routes as api_routes
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化组件，关闭时保存索引"""
    from src.rag.vectorstore import VectorStoreManager
    from src.rag.embeddings import get_embeddings
    from src.rag.retriever import HybridRetriever
    from src.agents.qa_agent import QAAgent

    # 初始化向量存储管理器
    embeddings = get_embeddings()
    manager = VectorStoreManager()
    manager.load_index(settings.faiss_index_path, embeddings)
    api_routes.vectorstore_manager = manager

    # 初始化问答 Agent
    agent = QAAgent()
    agent.setup_agent()
    api_routes.qa_agent = agent

    # 如果已有索引，重建检索器
    if manager.vectorstore is not None:
        try:
            # 从 data/ 目录重新加载所有文档用于 BM25
            import os
            from src.rag.loader import load_document
            from src.rag.splitter import split_documents

            all_docs = []
            data_dir = settings.data_dir
            if os.path.exists(data_dir):
                for fname in os.listdir(data_dir):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in {".pdf", ".md", ".markdown", ".txt", ".docx"}:
                        docs = load_document(os.path.join(data_dir, fname))
                        chunks = split_documents(docs)
                        all_docs.extend(chunks)

            api_routes.all_documents = all_docs
            if all_docs:
                retriever = HybridRetriever(
                    vectorstore=manager.vectorstore,
                    documents=all_docs,
                    embeddings=embeddings,
                )
                api_routes.hybrid_retriever = retriever
                agent.update_retriever(retriever, all_docs)
                logger.info(f"已有索引加载完成，文档块数: {len(all_docs)}")
        except Exception as e:
            logger.error(f"已有索引加载失败: {e}")

    logger.info("应用启动完成")

    yield

    # 关闭：保存索引
    if manager.vectorstore is not None:
        manager.save_index(settings.faiss_index_path)
    logger.info("应用已关闭")


app = FastAPI(
    title="智能文档问答系统",
    description="基于 RAG + Agent 技术的智能文档问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载路由
app.include_router(api_routes.router)


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "ok"}


@app.get("/")
async def index():
    """首页"""
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
