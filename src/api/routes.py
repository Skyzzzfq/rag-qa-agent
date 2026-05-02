import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.agents.qa_agent import QAAgent
from src.config import settings
from src.memory.conversation import get_memory, clear_memory, get_history
from src.rag.loader import load_document
from src.rag.splitter import split_documents
from src.rag.embeddings import get_embeddings
from src.rag.vectorstore import VectorStoreManager
from src.rag.retriever import HybridRetriever
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

qa_agent: QAAgent | None = None
vectorstore_manager: VectorStoreManager | None = None
hybrid_retriever: HybridRetriever | None = None
all_documents: list = []


def _rebuild_retriever() -> None:
    """根据当前向量存储和文档重建混合检索器"""
    global hybrid_retriever
    if vectorstore_manager and vectorstore_manager.vectorstore and all_documents:
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore_manager.vectorstore,
            documents=all_documents,
            embeddings=get_embeddings(),
        )
        qa_agent.update_retriever(hybrid_retriever, all_documents)
        logger.info("混合检索器已重建")


# ===== Pydantic 请求/响应模型 =====

class AskRequest(BaseModel):
    question: str
    session_id: str | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str

class UploadResponse(BaseModel):
    filename: str
    chunk_count: int
    index_status: str

class DocumentInfo(BaseModel):
    filename: str

class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]

class HistoryResponse(BaseModel):
    session_id: str
    history: list[dict]


# ===== API 路由 =====

@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """上传文档并自动构建索引"""
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in {".pdf", ".md", ".markdown"}:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    file_path = os.path.join(settings.data_dir, filename)
    os.makedirs(settings.data_dir, exist_ok=True)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"文件已保存: {file_path}")
    except Exception as e:
        logger.error(f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {e}")

    try:
        documents = load_document(file_path)
        chunks = split_documents(documents)
        all_documents.extend(chunks)

        embeddings = get_embeddings()
        if vectorstore_manager.vectorstore is None:
            vectorstore_manager.create_from_documents(chunks, embeddings)
        else:
            vectorstore_manager.add_documents(chunks)

        vectorstore_manager.save_index(settings.faiss_index_path)
        _rebuild_retriever()

        logger.info(f"文档处理完成: {filename}, 切分段数: {len(chunks)}")
        return UploadResponse(
            filename=filename,
            chunk_count=len(chunks),
            index_status="updated",
        )
    except Exception as e:
        logger.error(f"文档处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档处理失败: {e}")


@router.post("/qa/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """问答接口"""
    if qa_agent._chain is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化，请先上传文档")

    session_id = request.session_id or str(uuid.uuid4())

    memory = get_memory(session_id)
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    try:
        result = qa_agent.run(request.question, chat_history=chat_history)
        memory.save_context({"input": request.question}, {"output": result["answer"]})
        return AskResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"问答执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答执行失败: {e}")


@router.post("/qa/ask/stream")
async def ask_question_stream(request: AskRequest):
    """流式问答接口（SSE）"""
    if qa_agent._chain is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化，请先上传文档")

    session_id = request.session_id or str(uuid.uuid4())
    memory = get_memory(session_id)
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    async def event_generator():
        try:
            result = await qa_agent.arun(request.question, chat_history=chat_history)
            memory.save_context({"input": request.question}, {"output": result["answer"]})
            yield {"event": "final_answer", "data": result["answer"]}
        except Exception as e:
            logger.error(f"流式问答失败: {e}")
            yield {"event": "error", "data": str(e)}
        finally:
            yield {"event": "done", "data": ""}

    return EventSourceResponse(event_generator())


@router.get("/qa/history/{session_id}", response_model=HistoryResponse)
async def get_conversation_history(session_id: str):
    """获取对话历史"""
    history = get_history(session_id)
    return HistoryResponse(session_id=session_id, history=history)


@router.get("/documents/list", response_model=DocumentListResponse)
async def list_all_documents():
    """列出已加载的文档"""
    data_dir = settings.data_dir
    if not os.path.exists(data_dir):
        return DocumentListResponse(documents=[])
    files = os.listdir(data_dir)
    supported = [
        DocumentInfo(filename=f)
        for f in files
        if os.path.splitext(f)[1].lower() in {".pdf", ".md", ".markdown"}
    ]
    return DocumentListResponse(documents=supported)


@router.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """删除指定文档并更新索引"""
    file_path = os.path.join(settings.data_dir, doc_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_name}")

    try:
        os.remove(file_path)
        logger.info(f"文档已删除: {doc_name}")

        global all_documents
        all_documents = [doc for doc in all_documents if doc.metadata.get("source") != doc_name]

        if all_documents:
            embeddings = get_embeddings()
            vectorstore_manager.create_from_documents(all_documents, embeddings)
            vectorstore_manager.save_index(settings.faiss_index_path)
            _rebuild_retriever()
        else:
            vectorstore_manager.vectorstore = None
            index_path = settings.faiss_index_path
            if os.path.exists(index_path):
                for f in os.listdir(index_path):
                    os.remove(os.path.join(index_path, f))

        return {"message": f"文档 {doc_name} 已删除，索引已更新"}
    except Exception as e:
        logger.error(f"文档删除失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档删除失败: {e}")
