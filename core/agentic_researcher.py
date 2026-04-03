"""
🏛️ High-Level Agentic Research Orchestrator
Uses modular sub-agents to achieve autonomous ML optimization.
"""

import pandas as pd
from typing import Dict, Any
from agents.orchestrator_agent import OrchestratorAgent
from agents.memory_manager import MemoryManager

class AgenticResearchOrchestrator:
    """
    Orchestrates the research cycle between specialized agents.
    Replaces old monolithic logic with the Senior Architect's modular flow.
    """
    def __init__(self, target_metric: str = "f1", threshold: float = 0.85):
        self.target_metric = target_metric
        self.threshold = threshold
        self.orchestrator = OrchestratorAgent()
        self.memory = MemoryManager()
        self.history = []

    def run_research_cycle(self, df: pd.DataFrame, target_col: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        Runs iterative optimization cycles until the target metric is met.
        """
        print(f"🚀 [Orchestrator] Starting cycle for {target_col}...")
        
        best_overall_result = None
        
        for i in range(max_iterations):
            print(f"🧪 Cycle {i+1}/{max_iterations}")
            
            # Execute modular workflow
            try:
                result = self.orchestrator.run_workflow(df, target_col)
            except Exception as e:
                print(f"❌ Cycle {i+1} Failed: {e}")
                continue
            
            if "error" in result:
                print(f"⚠️ Agentic Error: {result['error']}")
                continue
            
            # Track history
            self.history.append({
                "iteration": i+1,
                "metrics": result["metrics"],
                "model": result["model_name"]
            })
            
            # Check milestone
            current_score = result["metrics"].get(self.target_metric, 0)
            if current_score >= self.threshold:
                print(f"✅ Success! Threshold {self.threshold} reached with {current_score:.4f}")
                result["status"] = "Target Reached"
                result["history"] = self.history
                return result
                
            # Update best
            if best_overall_result is None or current_score > best_overall_result["metrics"].get(self.target_metric, 0):
                best_overall_result = result
        
        # If we reach here, we exhausted budget
        if best_overall_result:
            best_overall_result["status"] = "Max Cycles Reached"
            best_overall_result["history"] = self.history
            return best_overall_result
        
        return {"status": "Research Failed", "history": self.history}
