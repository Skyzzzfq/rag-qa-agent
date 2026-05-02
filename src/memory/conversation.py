from langchain.memory import ConversationBufferWindowMemory

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 会话记忆存储，key 为 session_id
_memories: dict[str, ConversationBufferWindowMemory] = {}


def get_memory(session_id: str) -> ConversationBufferWindowMemory:
    """获取指定会话的记忆实例，不存在则自动创建"""
    if session_id not in _memories:
        _memories[session_id] = ConversationBufferWindowMemory(
            k=settings.conversation_window,
            return_messages=True,
            memory_key="chat_history",
            output_key="output",
            input_key="input",
        )
        logger.info(f"创建会话记忆: {session_id}")
    return _memories[session_id]


def clear_memory(session_id: str) -> None:
    """清除指定会话的记忆"""
    if session_id in _memories:
        _memories[session_id].clear()
        del _memories[session_id]
        logger.info(f"清除会话记忆: {session_id}")


def get_history(session_id: str) -> list[dict]:
    """获取指定会话的对话历史"""
    memory = get_memory(session_id)
    messages = memory.load_memory_variables({}).get("chat_history", [])
    history: list[dict] = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        history.append({"role": role, "content": msg.content})
    return history
