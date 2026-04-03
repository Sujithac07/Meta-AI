import os
from typing import Dict, Any

class AIArchitectAssistant:
    """
    Conversational agent that explains the system's logic and reasoning.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = None
        if self.api_key:
            try:
                # Lazy import to avoid Torch/DLL errors at startup
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key)
            except Exception as e:
                print(f"⚠️ ChatAgent Warning: Failed to load advanced LLM components: {e}")
                self.llm = None

    def answer_question(self, question: str, context: Dict[str, Any]) -> str:
        """
        Generates an architectural response based on current metrics and logic.
        """
        if not self.llm:
            return "Architecture Assistant: Advanced LLM logic is currently offline (environmental issues). However, looking at your metrics, focusing on top features and data cleaning is recommended."

        prompt = f"""
        You are a Senior AI Architect and this system's builder.
        Current Context:
        - Model: {context.get('model_name', 'N/A')}
        - Metrics: {context.get('metrics', 'N/A')}
        
        Question: {question}
        
        Provide a concise, professional, and architectural answer.
        """
        
        try:
            response = self.llm.predict(prompt)
            return response
        except Exception as e:
            return f"Error communicating with AI: {e}"
