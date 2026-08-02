from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from src.agents.tools import get_all_tools, set_retriever, set_documents
from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是一个智能文档问答助手，可以使用工具来回答问题。

严格的约束条件:
- 必须使用 search_knowledge_base 工具检索知识库中的相关内容
- 只能根据工具返回的文档内容回答，禁止使用你自身的知识进行补充或扩展
- 文档中没提到的信息，即使你知道，也绝对不能添加
- 如果文档内容不足以完整回答问题，请明确说明"文档中未提及"相关内容
- 不要编造信息
- 如果检索结果与问题无关，请如实说明知识库中没有相关内容
- 回答使用中文
- 在回答末尾标注信息来源"""

REFLECTION_PROMPT = """你是一个回答质量审查员。请检查以下回答是否存在问题。

用户问题: {question}

检索到的文档内容:
{context}

AI 的回答:
{answer}

请从以下四个维度审查，逐一判断是否通过：
1. 事实性：回答中的事实是否都来自检索到的文档内容，有没有编造信息？
2. 来源纯度：回答是否使用了模型自身知识进行补充？文档未提及的内容出现在回答中即为不通过。
3. 完整性：回答是否充分回应了用户问题，有没有遗漏关键点？
4. 一致性：回答内部是否存在自相矛盾的地方？

请用以下格式输出：
事实性: 通过/不通过 - 原因
来源纯度: 通过/不通过 - 原因
完整性: 通过/不通过 - 原因
一致性: 通过/不通过 - 原因
最终结论: 通过/不通过

如果最终结论为"不通过"，请在下一行用一句话说明需要改进的方向。"""


class QAAgent:
    """问答 Agent：基于 Tool Calling 的 RAG 智能体"""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base,
            model=settings.llm_model_name,
            temperature=0,
        )
        self._executor: AgentExecutor | None = None

    def setup_agent(self, tools: list = None) -> None:
        """创建 Tool Calling Agent 和 AgentExecutor"""
        if tools is None:
            tools = get_all_tools()

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self._executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True,
        )
        logger.info(f"Tool Calling Agent 已创建，工具数: {len(tools)}")

    def update_retriever(self, retriever, documents) -> None:
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

    async def _areflect(self, question: str, context: str, answer: str) -> tuple[bool, str]:
        """异步检查回答质量，避免阻塞异步请求。"""
        reflection_chain = ChatPromptTemplate.from_template(REFLECTION_PROMPT) | self.llm | StrOutputParser()
        result = await reflection_chain.ainvoke({
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

    def _extract_sources(self, intermediate_steps: list) -> list[str]:
        """从 Agent 中间步骤中提取文档来源"""
        sources = set()
        for action, output in intermediate_steps:
            if getattr(action, "tool", None) == "search_knowledge_base":
                for line in output.split("\n"):
                    stripped = line.strip()
                    if "来源:" in stripped:
                        source = stripped.split("来源:")[-1].strip()
                        if "(" in source:
                            source = source.split("(")[0].strip()
                        if source:
                            sources.add(source)
        return sorted(sources)

    def _build_context_from_steps(self, intermediate_steps: list) -> str:
        """从 Agent 中间步骤中提取检索到的上下文文本"""
        parts = []
        for action, output in intermediate_steps:
            if getattr(action, "tool", None) == "search_knowledge_base":
                parts.append(output)
        return "\n\n---\n\n".join(parts) if parts else "（未使用知识库检索）"

    def run(self, question: str, chat_history: list = None) -> dict:
        """执行问答（含反思纠错）"""
        if self._executor is None:
            raise ValueError("Agent 未初始化，请先调用 setup_agent")

        max_retries = settings.reflection_max_retries
        current_question = question

        for attempt in range(max_retries + 1):
            result = self._executor.invoke({
                "input": current_question,
                "chat_history": chat_history or [],
            })
            answer = result["output"]
            context_str = self._build_context_from_steps(result.get("intermediate_steps", []))

            if attempt < max_retries:
                passed, suggestion = self._reflect(current_question, context_str, answer)
                if passed:
                    logger.info(f"反思通过，第 {attempt + 1} 次生成即合格")
                    break
                logger.info(f"反思未通过，改进方向: {suggestion[:50]}，开始第 {attempt + 2} 次尝试")
                current_question = f"{question}\n\n[反思改进要求: {suggestion}]"
            else:
                logger.info(f"达到最大重试次数 {max_retries}，使用当前回答")

        sources = self._extract_sources(result.get("intermediate_steps", []))
        logger.info(f"问答完成: question='{question[:50]}...', sources={sources}")
        return {"answer": answer, "sources": sources}

    async def arun(self, question: str, chat_history: list = None) -> dict:
        """异步执行问答（含反思纠错）"""
        if self._executor is None:
            raise ValueError("Agent 未初始化，请先调用 setup_agent")

        max_retries = settings.reflection_max_retries
        current_question = question

        for attempt in range(max_retries + 1):
            result = await self._executor.ainvoke({
                "input": current_question,
                "chat_history": chat_history or [],
            })
            answer = result["output"]
            context_str = self._build_context_from_steps(result.get("intermediate_steps", []))

            if attempt < max_retries:
                passed, suggestion = await self._areflect(current_question, context_str, answer)
                if passed:
                    logger.info(f"反思通过，第 {attempt + 1} 次生成即合格")
                    break
                logger.info(f"反思未通过，改进方向: {suggestion[:50]}，开始第 {attempt + 2} 次尝试")
                current_question = f"{question}\n\n[反思改进要求: {suggestion}]"
            else:
                logger.info(f"达到最大重试次数 {max_retries}，使用当前回答")

        sources = self._extract_sources(result.get("intermediate_steps", []))
        return {"answer": answer, "sources": sources}

    async def astream_events(self, question: str, chat_history: list = None):
        """执行完整反思流程后，以 SSE 友好的事件格式返回结果。"""
        result = await self.arun(question, chat_history=chat_history)
        yield {"type": "sources", "data": result["sources"]}
        answer = result["answer"]
        for start in range(0, len(answer), 12):
            yield {"type": "token", "data": answer[start:start + 12]}
