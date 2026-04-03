import os
import json
from typing import Dict, Any

class ModelSelectorLLM:
    """
    Uses LLM intelligence to suggest the best model architecture.
    """
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def suggest_strategy(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Connects to LLM to get strategic advice for the AutoML run.
        """
        if not self.api_key:
            return {
                "suggested_model": "RandomForest",
                "reasoning": "OpenAI API Key missing, defaulting to robust RandomForest BASELINE.",
                "preprocessing": "Standard Scaling"
            }
            
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model="gpt-4o", api_key=self.api_key, temperature=0.1)
            
            prompt = f"""
            Analyze this dataset summary and suggest the best ML strategy:
            {json.dumps(data_summary, indent=2)}
            
            Return ONLY a JSON object with:
            {{
              "suggested_model": "RandomForest|XGBoost|LightGBM",
              "reasoning": "string",
              "preprocessing": "string"
            }}
            """
            response = llm.predict(prompt)
            # Find JSON block
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"suggested_model": "RandomForest", "reasoning": "Fallback to baseline.", "preprocessing": "Default"}
        except Exception as e:
            return {
                "suggested_model": "RandomForest",
                "reasoning": f"Strategy logic error ( {str(e)} ), falling back.",
                "preprocessing": "Standard Scaling"
            }
