import os
import json
from typing import Any, Dict, Optional

def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)

def generate_llm_insights(
    profile: Dict[str, Any],
    risks: Dict[str, Any],
    strategy: Dict[str, Any],
    metrics: Dict[str, Any],
    validation: Dict[str, Any],
    stability: Dict[str, Any],
    failure_analysis: Dict[str, Any]
) -> Optional[str]:
    """
    Uses OpenAI to generate a concise expert summary and recommendations.
    Returns None if no API key is configured or if the call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except (ImportError, OSError) as e:
        print(f"Warning: OpenAI not available or failed to initialize: {e}")
        return None

    system_msg = (
        "You are a senior ML architect. Produce a crisp executive summary, "
        "key risks, and actionable improvements. Keep it concise and practical."
    )

    user_payload = {
        "profile": profile,
        "risks": risks,
        "strategy": strategy,
        "metrics": metrics,
        "validation": validation,
        "stability": stability,
        "failure_analysis": failure_analysis
    }

    prompt = (
        "Analyze this ML system context and provide:\n"
        "1) Executive summary (3-5 bullets)\n"
        "2) Top risks (3 bullets)\n"
        "3) Improvements (3 bullets)\n\n"
        f"CONTEXT:\n{_safe_json(user_payload)}"
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"), # Use gpt-4o as default
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except Exception:
        return None
