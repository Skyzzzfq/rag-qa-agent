from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

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

REFLECTION_PROMPT = """你是一个回答质量审查员。请检查以下回答是否存在问题。

用户问题: {question}

检索到的文档内容:
{context}

AI 的回答:
{answer}

请从以下三个维度审查，逐一判断是否通过：
1. 事实性：回答中的事实是否都来自检索到的文档内容，有没有编造信息？
2. 完整性：回答是否充分回应了用户问题，有没有遗漏关键点？
3. 一致性：回答内部是否存在自相矛盾的地方？

请用以下格式输出：
事实性: 通过/不通过 - 原因
完整性: 通过/不通过 - 原因
一致性: 通过/不通过 - 原因
最终结论: 通过/不通过

如果最终结论为"不通过"，请在下一行用一句话说明需要改进的方向。"""


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
                "context": lambda x: x["context"],
                "question": lambda x: x["question"],
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

    def _reflect(self, question: str, context: str, answer: str) -> tuple[bool, str]:
        """反思纠错：检查回答质量，返回 (是否通过, 改进方向)"""
        reflection_chain = ChatPromptTemplate.from_template(REFLECTION_PROMPT) | self.llm | StrOutputParser()
        result = reflection_chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
        })
        passed = "最终结论: 通过" in result
        suggestion = ""
        if not passed:
            for line in result.split("\n"):
                line = line.strip()
                if line and not line.startswith(("事实性", "完整性", "一致性", "最终结论")):
                    suggestion = line
                    break
        logger.info(f"反思结果: passed={passed}, suggestion='{suggestion[:50]}'")
        return passed, suggestion

    def _retrieve_context(self, question: str) -> str:
        """检索并返回格式化的上下文"""
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
        """执行问答（含反思纠错）"""
        if self._chain is None:
            raise ValueError("Agent 未初始化，请先调用 setup_agent")

        max_retries = settings.reflection_max_retries
        current_question = question

        for attempt in range(max_retries + 1):
            context = self._retrieve_context(current_question)
            answer = self._chain.invoke({
                "question": current_question,
                "context": context,
                "chat_history": chat_history or [],
            })

            if attempt < max_retries:
                passed, suggestion = self._reflect(current_question, context, answer)
                if passed:
                    logger.info(f"反思通过，第 {attempt + 1} 次生成即合格")
                    break
                logger.info(f"反思未通过，改进方向: {suggestion[:50]}，开始第 {attempt + 2} 次尝试")
                current_question = f"{question}\n\n[反思改进要求: {suggestion}]"
            else:
                logger.info(f"达到最大重试次数 {max_retries}，使用当前回答")

        sources = list({doc.metadata.get("source", "") for doc in self._context_docs if doc.metadata.get("source")})
        logger.info(f"问答完成: question='{question[:50]}...', sources={sources}")
        return {"answer": answer, "sources": sources}

    async def arun(self, question: str, chat_history: list = None) -> dict:
        """异步执行问答（含反思纠错）"""
        if self._chain is None:
            raise ValueError("Agent 未初始化，请先调用 setup_agent")

        max_retries = settings.reflection_max_retries
        current_question = question

        for attempt in range(max_retries + 1):
            context = self._retrieve_context(current_question)
            answer = await self._chain.ainvoke({
                "question": current_question,
                "context": context,
                "chat_history": chat_history or [],
            })

            if attempt < max_retries:
                passed, suggestion = self._reflect(current_question, context, answer)
                if passed:
                    logger.info(f"反思通过，第 {attempt + 1} 次生成即合格")
                    break
                logger.info(f"反思未通过，改进方向: {suggestion[:50]}，开始第 {attempt + 2} 次尝试")
                current_question = f"{question}\n\n[反思改进要求: {suggestion}]"
            else:
                logger.info(f"达到最大重试次数 {max_retries}，使用当前回答")

        sources = list({doc.metadata.get("source", "") for doc in self._context_docs if doc.metadata.get("source")})
        return {"answer": answer, "sources": sources}
