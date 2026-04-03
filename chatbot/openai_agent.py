import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except (ImportError, OSError):
    OPENAI_AVAILABLE = False


class OpenAIAgent:
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI library is not available or failed to initialize.")

        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))
        self.history: List[Dict[str, str]] = []

    def reset_history(self) -> None:
        self.history = []

    def _build_system_prompt(self, system_state: Optional[Dict[str, Any]], memory_context: Any) -> str:
        state = system_state or {}
        context_text = str(memory_context or "").strip()

        return f"""
You are Meta AI Assistant, a helpful and accurate AI chatbot.
You can answer ANY kind of user question.

When report context is available, use it first for report-related questions.
If the answer is not present in report context, answer using your general knowledge.
Never invent report numbers or metrics that are not explicitly present.

REPORT CONTEXT:
- Final Decision: {state.get('final_decision', 'N/A')}
- Primary Reason: {state.get('reason', 'N/A')}
- Risk Analysis: {state.get('risks', 'N/A')}
- Agent Deliberation Trace: {state.get('agent_trace', 'N/A')}

RETRIEVED MEMORY:
{context_text if context_text else 'No additional report memory provided.'}
""".strip()

    def _extract_text(self, response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts).strip()
        return str(content).strip()

    def answer(
        self,
        question: str,
        system_state: Optional[Dict[str, Any]] = None,
        memory_context: Any = None,
    ) -> str:
        if not question or not question.strip():
            return "Please ask a question."

        system_prompt = self._build_system_prompt(system_state, memory_context)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.history[-10:])
        messages.append({"role": "user", "content": question.strip()})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        answer_text = self._extract_text(response) or "I could not generate a response."

        self.history.append({"role": "user", "content": question.strip()})
        self.history.append({"role": "assistant", "content": answer_text})

        return answer_text
