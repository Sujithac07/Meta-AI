from core.risk_reasoning import analyze_risks

class RiskAgent:
    def run(self, profile):
        risks = analyze_risks(profile)
        return {
            "risks": risks,
            "comment": "RiskAgent identified potential modeling risks."
        }
