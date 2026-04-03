import json
import os
from datetime import datetime
from typing import Dict, List, Any

class MemoryManager:
    """
    Persistent JSON-based memory for agentic decision history.
    Allows the orchestrator to learn from past runs.
    """
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Default to project root
            storage_path = os.path.join(os.getcwd(), "agents", "decision_history.json")
        self.path = storage_path
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, 'w') as f:
                json.dump([], f)

    def save_decision(self, dataset_name: str, model_name: str, metrics: Dict[str, float], decision_logic: str):
        """Register a new decision in the ledger."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "selected_model": model_name,
            "performance": metrics,
            "logic": decision_logic
        }
        
        current_data = self.load_history()
        current_data.append(record)
        
        with open(self.path, 'w') as f:
            json.dump(current_data, f, indent=4)

    def load_history(self) -> List[Dict[str, Any]]:
        """Load all previous decisions."""
        try:
            with open(self.path, 'r') as f:
                return json.load(f)
        except Exception:
            return []

    def get_summary(self) -> str:
        """Returns a string summary of the last 3 decisions for prompt context."""
        history = self.load_history()[-3:]
        if not history:
            return "No previous decisions found."
        
        summary = "Past Decision Context:\n"
        for item in history:
            summary += f"- Dataset '{item['dataset']}': Used {item['selected_model']} (Acc: {item['performance'].get('accuracy', 0):.2f})\n"
        return summary
