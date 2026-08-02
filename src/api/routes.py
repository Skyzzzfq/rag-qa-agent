import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.agents.qa_agent import QAAgent
from src.config import settings
from src.memory.conversation import get_memory, get_history
from src.rag.loader import load_document, SUPPORTED_EXTENSIONS
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
_index_lock = asyncio.Lock()


def _resolve_document_path(filename: str | None) -> Path:
    """校验文档名并确保最终路径位于数据目录内。"""
    if (
        not filename
        or filename.startswith(".")
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise HTTPException(status_code=400, detail="无效的文件名")

    data_dir = Path(settings.data_dir).resolve()
    file_path = (data_dir / filename).resolve()
    if file_path.parent != data_dir:
        raise HTTPException(status_code=400, detail="无效的文件名")
    return file_path


def _load_stored_documents(exclude: str | None = None) -> list:
    """从磁盘加载文档，作为索引的唯一事实来源。"""
    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        return []

    chunks = []
    excluded_name = exclude.casefold() if exclude else None
    for path in data_dir.iterdir():
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.name.casefold() == excluded_name
            or path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            continue
        chunks.extend(split_documents(load_document(str(path))))
    return chunks


def _rebuild_retriever() -> None:
    """根据当前向量存储和文档重建混合检索器"""
    global hybrid_retriever
    if vectorstore_manager and vectorstore_manager.vectorstore and all_documents:
        hybrid_retriever = HybridRetriever(
            vectorstore=vectorstore_manager.vectorstore,
            documents=all_documents,
            embeddings=get_embeddings(),
        )
        if qa_agent:
            qa_agent.update_retriever(hybrid_retriever, all_documents)
        logger.info("混合检索器已重建")
    else:
        hybrid_retriever = None
        if qa_agent:
            qa_agent.update_retriever(None, [])


# ===== Pydantic 请求/响应模型 =====

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
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
    global all_documents

    if vectorstore_manager is None:
        raise HTTPException(status_code=503, detail="索引服务尚未初始化")

    filename = file.filename or ""
    file_path = _resolve_document_path(filename)
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.parent / f".{uuid.uuid4().hex}.upload{ext}"
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    total_bytes = 0
    try:
        with temp_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件不能超过 {settings.max_upload_size_mb} MB",
                    )
                output.write(chunk)
        logger.info(f"文件已暂存: {filename}, 大小: {total_bytes} bytes")
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        logger.exception("文件保存失败")
        raise HTTPException(status_code=500, detail="文件保存失败") from e

    try:
        documents = await run_in_threadpool(load_document, str(temp_path))
        chunks = await run_in_threadpool(split_documents, documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="文档中没有可索引的文本")
        for chunk in chunks:
            chunk.metadata["source"] = filename

        async with _index_lock:
            existing_chunks = await run_in_threadpool(_load_stored_documents, filename)
            indexed_documents = existing_chunks + chunks
            replacement = VectorStoreManager()
            embeddings = get_embeddings()
            await run_in_threadpool(
                replacement.create_from_documents, indexed_documents, embeddings
            )

            os.replace(temp_path, file_path)
            vectorstore_manager.vectorstore = replacement.vectorstore
            all_documents = indexed_documents
            await run_in_threadpool(vectorstore_manager.save_index, settings.faiss_index_path)
            _rebuild_retriever()

        logger.info(f"文档处理完成: {filename}, 切分段数: {len(chunks)}")
        return UploadResponse(
            filename=filename,
            chunk_count=len(chunks),
            index_status="updated",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("文档处理失败")
        raise HTTPException(status_code=500, detail="文档处理失败") from e
    finally:
        temp_path.unlink(missing_ok=True)


@router.post("/qa/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """问答接口"""
    if qa_agent is None or qa_agent._executor is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化，请先上传文档")

    session_id = request.session_id or str(uuid.uuid4())

    memory = get_memory(session_id)
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    try:
        result = await run_in_threadpool(qa_agent.run, request.question, chat_history)
        memory.save_context({"input": request.question}, {"output": result["answer"]})
        return AskResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            session_id=session_id,
        )
    except Exception as e:
        logger.exception("问答执行失败")
        raise HTTPException(status_code=500, detail="问答执行失败") from e


@router.post("/qa/ask/stream")
async def ask_question_stream(request: AskRequest):
    """SSE 问答接口，在质量校验后分块返回答案和来源。"""
    if qa_agent is None or qa_agent._executor is None:
        raise HTTPException(status_code=503, detail="Agent 尚未初始化，请先上传文档")

    session_id = request.session_id or str(uuid.uuid4())
    memory = get_memory(session_id)
    chat_history = memory.load_memory_variables({}).get("chat_history", [])

    async def event_generator():
        full_answer = ""
        try:
            yield {"event": "session", "data": session_id}
            async for event in qa_agent.astream_events(request.question, chat_history=chat_history):
                event_type = event["type"]
                if event_type == "tool_start":
                    yield {"event": "tool_start", "data": json.dumps(event["data"], ensure_ascii=False)}
                elif event_type == "tool_end":
                    yield {"event": "tool_end", "data": json.dumps(event["data"], ensure_ascii=False)}
                elif event_type == "token":
                    full_answer += event["data"]
                    yield {"event": "token", "data": json.dumps(event["data"], ensure_ascii=False)}
                elif event_type == "sources":
                    yield {"event": "sources", "data": json.dumps(event["data"], ensure_ascii=False)}

            memory.save_context({"input": request.question}, {"output": full_answer})
            yield {"event": "done", "data": ""}
        except Exception as e:
            logger.exception("流式问答失败")
            yield {"event": "error", "data": "问答执行失败"}
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
        if not f.startswith(".") and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    return DocumentListResponse(documents=supported)


@router.delete("/documents/{doc_name}")
async def delete_document(doc_name: str):
    """删除指定文档并更新索引"""
    global all_documents

    if vectorstore_manager is None:
        raise HTTPException(status_code=503, detail="索引服务尚未初始化")

    file_path = _resolve_document_path(doc_name)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_name}")

    try:
        async with _index_lock:
            remaining_documents = await run_in_threadpool(_load_stored_documents, doc_name)
            replacement = VectorStoreManager()
            if remaining_documents:
                embeddings = get_embeddings()
                await run_in_threadpool(
                    replacement.create_from_documents, remaining_documents, embeddings
                )

            file_path.unlink()
            vectorstore_manager.vectorstore = replacement.vectorstore
            all_documents = remaining_documents
            if remaining_documents:
                await run_in_threadpool(vectorstore_manager.save_index, settings.faiss_index_path)
            else:
                index_path = Path(settings.faiss_index_path)
                for index_file in (index_path / "index.faiss", index_path / "index.pkl"):
                    index_file.unlink(missing_ok=True)
            _rebuild_retriever()
            logger.info(f"文档已删除: {doc_name}")

        return {"message": f"文档 {doc_name} 已删除，索引已更新"}
    except Exception as e:
        logger.exception("文档删除失败")
        raise HTTPException(status_code=500, detail="文档删除失败") from e
