import os
import re
from typing import Optional

class ResultsChatbot:
    def __init__(self, report_path="meta_ai_report.txt"):
        self.report_path = report_path
        self.report_content = self._load_report()
        self.context = self._parse_report_context()
        self.agent = self._init_agent()

    def _init_agent(self) -> Optional[object]:
        try:
            from chatbot.openai_agent import OpenAIAgent

            return OpenAIAgent()
        except Exception:
            return None

    def _load_report(self):
        if not os.path.exists(self.report_path):
            return "Report not found. Please run main.py first to generate results."
        with open(self.report_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _parse_report_context(self):
        """Simple parser to extract key sections for context"""
        context = {}
        # Extract Accuracy/decision roughly
        if "ACCEPT" in self.report_content:
            context['decision'] = "ACCEPT"
        elif "REJECT" in self.report_content:
            context['decision'] = "REJECT"
        else:
            context['decision'] = "Unknown"
        
        return context

    def answer(self, question):
        """
        Main entry point. Tries OpenAI first, then falls back to rule-based parsing.
        """
        # 1) Try OpenAI agent
        try:
            if self.agent is None:
                self.agent = self._init_agent()

            if self.agent is None:
                raise RuntimeError("OpenAI agent is unavailable.")

            system_state = {
                'final_decision': self.context.get('decision', 'N/A'),
                'reason': "extracted from report",
                'agent_trace': "N/A",
                'risks': "See report details"
            }
            response = self.agent.answer(question, system_state, self.report_content)
            return f"[Neural Interface]: {response}"

        except Exception:
            # 2) Fallback to robust report parser
            return self.rule_based_response(question)

    def rule_based_response(self, question):
        """
        Robust fallback that scans the report for keywords.
        """
        if "Report not found" in self.report_content:
            return (
                "[Fallback]: I can answer any question once OPENAI_API_KEY is available. "
                "Right now, no report was found and OpenAI is unavailable."
            )

        q = question.lower()
        
        response = "[Report Reader]: "
        found_something = False

        if "accuracy" in q or "performance" in q or "score" in q:
            # Extract evaluation results section
            match = re.search(r"4\. MODEL EVALUATION RESULTS\n(.*?)\n\n", self.report_content, re.DOTALL)
            if match:
                response += f"Here are the model performance metrics:\n{match.group(1)}\n"
                found_something = True

        if "risk" in q or "warning" in q:
            match = re.search(r"2\. IDENTIFIED RISKS\n(.*?)\n\n", self.report_content, re.DOTALL)
            if match:
                response += f"Identified Risks:\n{match.group(1)}\n"
                found_something = True
                
        if "decision" in q or "result" in q:
            match = re.search(r"6\. SELF-AUDIT & FINAL DECISION\n(.*?)\n", self.report_content, re.DOTALL)
            if match:
                response += f"Final System Decision:\n{match.group(1)}\n"
                found_something = True

        if not found_something:
            response += "I searched the report but couldn't find a direct answer to that specific question. You can ask about 'accuracy', 'risks', or 'decision'.\n\nFull Report Summary:\n" + self.report_content[:500] + "..."

        return response
