from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.agents.tools import set_retriever, set_documents
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

QA_PROMPT = """你是一个智能文档问答助手。请根据以下检索到的文档内容回答用户的问题。

约束条件:
- 仅根据检索到的文档内容回答，不确定时明确说明
- 不要编造信息
- 如果检索结果与问题无关，请如实说明知识库中没有相关内容
- 回答使用中文
- 在回答末尾标注信息来源

检索到的文档内容:
{context}

用户问题: {question}"""


class QAAgent:
    """问答 Agent：检索增强生成（RAG）"""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
            model=settings.llm_model_name,
            temperature=0,
        )
        self._chain = None
        self._context_docs: list[Document] = []

    def setup_agent(self, tools: list = None) -> None:
        """创建 RAG 问答链"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", QA_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}"),
        ])
        self._chain = (
            {
                "context": lambda x: self._retrieve(x["question"]),
                "question": RunnablePassthrough(),
                "chat_history": lambda x: x.get("chat_history", []),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("RAG 问答链已创建")

    def update_retriever(self, retriever, documents: list[Document]) -> None:
        """更新检索器和文档列表"""
        set_retriever(retriever)
        set_documents(documents)

    def _retrieve(self, question: str) -> str:
        """检索并格式化上下文"""
        from src.agents.tools import _hybrid_retriever
        if _hybrid_retriever is None:
            return "知识库尚未初始化，请先上传文档。"
        try:
            docs = _hybrid_retriever.retrieve(question)
            self._context_docs = docs
            if not docs:
                return "未找到相关文档内容。"
            parts = []
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page", "")
                page_info = f" (第{page}页)" if page else ""
                parts.append(f"[{i}] 来源: {source}{page_info}\n{doc.page_content}")
            return "\n\n---\n\n".join(parts)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return f"检索失败: {e}"

    def run(self, question: str, chat_history: list = None) -> dict:
        """执行问答"""
        if self._chain is None:
            raise ValueError("Agent 未初始化，请先调用 setup_agent")

        answer = self._chain.invoke({
            "question": question,
            "chat_history": chat_history or [],
        })
        sources = list({doc.metadata.get("source", "") for doc in self._context_docs if doc.metadata.get("source")})
        logger.info(f"问答完成: question='{question[:50]}...', sources={sources}")
        return {"answer": answer, "sources": sources}

    async def arun(self, question: str, chat_history: list = None) -> dict:
        """异步执行问答"""
        if self._chain is None:
            raise ValueError("Agent 未初始化，请先调用 setup_agent")

        answer = await self._chain.ainvoke({
            "question": question,
            "chat_history": chat_history or [],
        })
        sources = list({doc.metadata.get("source", "") for doc in self._context_docs if doc.metadata.get("source")})
        return {"answer": answer, "sources": sources}
