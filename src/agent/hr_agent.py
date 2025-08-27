# src/agent/hr_agent.py
import os
import logging
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor

from src.agent.tools import build_tools

logger = logging.getLogger(__name__)

AGENT_SYSTEM = """أنت hr_agent.

مهمتك اختيار الأداة المناسبة قبل توليد الإجابة:
- إذا كان السؤال عن سياسات وإجراءات الموارد البشرية الداخلية → استخدم أداة hr_search.
- إذا كان السؤال عن استخدام منصة جسر (مثل تسجيل الموظف، الإجازات عبر جسر، مسير الرواتب داخل جسر، طريقة رفع الطلب، التنقل داخل النظام) → استخدم أداة jisr_search.
- إذا كان السؤال يجمع بين سياسة داخلية وكيفية تنفيذها على جسر → استدعِ الأداتين بالترتيب المنطقي (أولاً HR ثم JISR غالباً) ثم ادمج النتائج.

قواعد صارمة:
1) يجب استدعاء أداة واحدة على الأقل قبل الإجابة. لا تكتب إجابة دون نتائج أدوات.
2) اعتمد فقط على "context" القادم من الأدوات. لا تُخمن ولا تضف معلومات خارج المصادر.
3) إذا لم تجد معلومة كافية في السياق، صرّح بذلك بوضوح واقترح على المستخدم تحديد سؤاله أو رفع ملف السياسة المناسب.
4) صُغ الإجابة بالعربية بشكل موجز وواضح وعملي.
5) اختم بقسم "المراجع" يذكر اسم المستند + رقم الجزء (chunk) لكل مصدر استندت إليه.
6) في السطر الأخير، ضع بلوك JSON داخل الوسمين التاليين حرفيًا:
<citations>{{"items":[...]}}</citations>
- "items" يجب أن تكون مصفوفة الدمج (بدون تكرار) لكل العناصر القادمة من الأدوات بالشكل:
  {{"doc_title": "string", "chunk": 0, "source": "string", "corpus": "hr"|"jisr"}}
- لا تضف مفاتيح أخرى غير المذكورة.
- لا تغيّر أسماء المفاتيح.

إرشادات لاستخدام نتائج الأدوات:
- كل أداة تُرجع نصًا بصيغة JSON يحوي:
  {{"context": "...", "citations": [{{...}}, ...]}}
- اقرأ هذا الـ JSON، ثم اعتمد على "context" حصراً لتأليف الإجابة.
- ادمج جميع "citations" من جميع الأدوات (وتخلّص من التكرارات) وضعها في البلوك النهائي أعلاه.
"""

AGENT_HUMAN = "{input}"

def _build_llm() -> ChatGroq:
    """Build the Groq LLM client locked to openai/gpt-oss-120b."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing")

    model = "openai/gpt-oss-120b"   # 🔒 fixed model
    try:
        temperature = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
    except Exception:
        temperature = 0.2

    logger.info(f"Starting hr_agent with model: {model}, temperature={temperature}")
    return ChatGroq(model_name=model, temperature=temperature, groq_api_key=api_key)

def build_hr_agent(retriever: Any, settings: Any) -> AgentExecutor:
    """
    Build the tool-calling agent that can pick hr_search / jisr_search (or both),
    then compose a final Arabic answer with citations.
    """
    tools = build_tools(retriever, settings.DEFAULT_TOP_K)
    llm = _build_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", AGENT_HUMAN),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
        early_stopping_method="force",   # ✅ FIXED (was "generate")
        return_intermediate_steps=False
    )
    return executor
