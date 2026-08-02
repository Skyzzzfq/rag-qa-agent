"""API 集成测试：输入边界、SSE 会话和来源。"""

import re
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import routes


def create_client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_upload_rejects_path_traversal(monkeypatch):
    monkeypatch.setattr(routes, "vectorstore_manager", MagicMock())
    client = create_client()

    response = client.post(
        "/documents/upload",
        files={"file": ("../outside.txt", b"content", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "无效的文件名"


def test_upload_enforces_size_limit(monkeypatch, tmp_path):
    monkeypatch.setattr(routes, "vectorstore_manager", MagicMock())
    monkeypatch.setattr(routes.settings, "data_dir", str(tmp_path))
    monkeypatch.setattr(routes.settings, "max_upload_size_mb", 0)
    client = create_client()

    response = client.post(
        "/documents/upload",
        files={"file": ("large.txt", b"content", "text/plain")},
    )

    assert response.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_upload_replaces_same_named_document_without_duplicate_chunks(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "guide.txt").write_text("旧内容", encoding="utf-8")

    active_manager = MagicMock()
    replacement = MagicMock()

    def create_index(documents, embeddings):
        replacement.vectorstore = object()

    replacement.create_from_documents.side_effect = create_index
    monkeypatch.setattr(routes, "vectorstore_manager", active_manager)
    monkeypatch.setattr(routes, "all_documents", [])
    monkeypatch.setattr(routes, "VectorStoreManager", lambda: replacement)
    monkeypatch.setattr(routes, "get_embeddings", lambda: object())
    monkeypatch.setattr(routes, "_rebuild_retriever", MagicMock())
    monkeypatch.setattr(routes.settings, "data_dir", str(data_dir))
    monkeypatch.setattr(routes.settings, "faiss_index_path", str(tmp_path / "indexes"))
    client = create_client()

    response = client.post(
        "/documents/upload",
        files={"file": ("guide.txt", "新内容".encode(), "text/plain")},
    )

    assert response.status_code == 200
    assert (data_dir / "guide.txt").read_text(encoding="utf-8") == "新内容"
    indexed_documents = replacement.create_from_documents.call_args.args[0]
    assert "".join(doc.page_content for doc in indexed_documents) == "新内容"
    assert active_manager.vectorstore is replacement.vectorstore


def test_stream_returns_session_sources_and_answer(monkeypatch):
    fake_agent = MagicMock()
    fake_agent._executor = object()
    received_histories = []

    async def stream_events(question, chat_history=None):
        received_histories.append(chat_history)
        yield {"type": "sources", "data": ["guide.md"]}
        yield {"type": "token", "data": "最终回答"}

    fake_agent.astream_events = stream_events
    monkeypatch.setattr(routes, "qa_agent", fake_agent)

    with create_client() as client:
        response = client.post("/qa/ask/stream", json={"question": "测试问题"})

        assert response.status_code == 200
        assert "event: session" in response.text
        assert "event: sources" in response.text
        assert '["guide.md"]' in response.text
        assert "最终回答" in response.text

        session_match = re.search(r"event: session\r?\ndata: ([^\r\n]+)", response.text)
        assert session_match is not None
        second_response = client.post(
            "/qa/ask/stream",
            json={"question": "继续提问", "session_id": session_match.group(1)},
        )
        assert second_response.status_code == 200
        assert len(received_histories[1]) == 2
