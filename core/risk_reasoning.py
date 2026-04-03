import os
import json

# Pre-define names to avoid NameError if imports fail
ChatOpenAI = None

try:
    from langchain_openai import ChatOpenAI
    try:
        # Legacy import path (older LangChain versions)
        from langchain.prompts import PromptTemplate
    except Exception:
        # Modern import path
        from langchain_core.prompts import PromptTemplate
    LLM_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: LLM components (langchain/openai) not available or failed to initialize: {e}")
    LLM_AVAILABLE = False

def analyze_risks(profile):
    """
    Analyzes risks using LLM if available, otherwise falls back to heuristics.
    """
    risks = {}
    reasoning = []
    
    # 1. Try LLM-based Reasoning
    if LLM_AVAILABLE and os.getenv("OPENAI_API_KEY") and ChatOpenAI:
        try:
            llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
            
            prompt = PromptTemplate.from_template("""
            You are an expert AI Risk Officer. Analyze the following dataset profile and identify critical risks.
            
            DATA PROFILE:
            {profile}
            
            Output valid JSON only with keys: 
            "class_imbalance" (bool), 
            "imbalance_severity" (low/moderate/high), 
            "overfitting_risk" (low/moderate/high),
            "recommended_metric" (str),
            "reasoning" (list of strings explaining the risks).
            """)
            
            chain = prompt | llm
            response = chain.invoke({"profile": str(profile)})
            
            # Clean and parse JSON
            content = response.content.strip()
            if "```json" in content: 
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            llm_risks = json.loads(content)
            return llm_risks
            
        except Exception as e:
            print(f"LLM Risk Analysis failed: {e}. Falling back to heuristics.")
            reasoning.append(f"LLM Reasoning failed ({str(e)}), reverted to rule-based logic.")
    
    # 2. Heuristic Fallback (Original Logic)
    target_dist = profile.get("target_distribution", {})
    if target_dist:
        vals = list(target_dist.values())
        total = sum(vals)
        minority_ratio = min(vals) / total if total > 0 else 0
        
        if minority_ratio < 0.2:
            risks["class_imbalance"] = True
            risks["imbalance_severity"] = "high" if minority_ratio < 0.1 else "moderate"
            risks["recommended_metric"] = "F1-score"
            reasoning.append(f"Class imbalance detected (Minority class: {minority_ratio:.1%}). Accuracy will be misleading.")
        else:
            risks["class_imbalance"] = False
            risks["imbalance_severity"] = "low"
            risks["recommended_metric"] = "accuracy"
            reasoning.append("Dataset is balanced. Accuracy is a valid metric.")
    else:
         # Fallback for regression or unknown target
         risks["class_imbalance"] = False
         risks["recommended_metric"] = "RMSE"

    # Overfitting risk
    rows = profile.get("rows", 0)
    cols = profile.get("columns", 0)
    
    if rows < 1000:
        risks["overfitting_risk"] = "high"
        reasoning.append(f"Small dataset ({rows} rows). highly susceptible to overfitting.")
    elif rows < 10 * cols:
        risks["overfitting_risk"] = "moderate"
        reasoning.append("Dataset size is small relative to feature count (Curse of Dimensionality).")
    else:
        risks["overfitting_risk"] = "low"
        reasoning.append("Sufficient data volume for stable training.")

    risks["reasoning"] = reasoning
    return risks
