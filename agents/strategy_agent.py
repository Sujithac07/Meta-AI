from core.strategy_output import generate_strategy
from agents.memory_agent import MemoryAgent

class StrategyAgent:
    def __init__(self):
        self.memory_agent = MemoryAgent()

    def run(self, risks):
        strategy = generate_strategy(risks)

        knowledge = self.memory_agent.consult(
            "What models are recommended for imbalanced or small datasets?"
        )

        return {
            "strategy": strategy,
            "knowledge_used": knowledge,
            "comment": "StrategyAgent used memory-based reasoning."
        }
