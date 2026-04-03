"""
Agentic Insight Generator
LLM-powered strategic advice for model improvement
"""

import os
from typing import Dict
from datetime import datetime

# Try importing LLM libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class AgentInsightGenerator:
    """
    LLM-powered strategic advice generator.
    Analyzes anomalies, feature importance, and accuracy to provide improvement recommendations.
    """
    
    SYSTEM_PROMPT = """You are a Senior ML Strategist providing actionable advice to improve a machine learning model.
Based on the pipeline data provided, write a 3-paragraph strategic advice report:

**Paragraph 1: Data Quality Assessment**
Analyze the anomaly scores and cleaning results. Explain what the anomaly patterns reveal about data quality,
whether the anomalies are problematic or contain valuable edge cases, and specific data collection improvements needed.

**Paragraph 2: Feature Strategy**
Analyze the feature importance rankings. Identify which features dominate predictions (potential overfitting risk),
which features contribute little (candidates for removal), and suggest new feature engineering opportunities
based on the top features.

**Paragraph 3: Model Improvement Roadmap**
Based on the accuracy metrics and model tournament results, provide a concrete 3-step action plan for the next iteration.
Include specific hyperparameter adjustments, alternative algorithms to try, or ensemble strategies to explore.

Keep each paragraph focused and actionable. Use bullet points where helpful.
Write in a professional but accessible tone suitable for a technical presentation."""

    def __init__(self, provider: str = "auto"):
        """
        Initialize with LLM provider.
        
        Args:
            provider: 'openai', 'gemini', 'groq', or 'auto' (auto-detect)
        """
        self.provider = self._detect_provider(provider)
        self.client = None
        self._initialize_client()
    
    def _detect_provider(self, provider: str) -> str:
        """Auto-detect available LLM provider."""
        if provider != "auto":
            return provider
        
        if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
            return "openai"
        elif os.getenv("GOOGLE_API_KEY") and GEMINI_AVAILABLE:
            return "gemini"
        elif os.getenv("GROQ_API_KEY") and GROQ_AVAILABLE:
            return "groq"
        else:
            return "template"  # Fallback to template-based
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        if self.provider == "openai" and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
        
        elif self.provider == "gemini" and GEMINI_AVAILABLE:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel('gemini-pro')
        
        elif self.provider == "groq" and GROQ_AVAILABLE:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.client = Groq(api_key=api_key)
    
    def generate_insight(self,
                        cleaning_report: Dict,
                        training_report: Dict,
                        xai_report: Dict,
                        feature_report: Dict = None) -> str:
        """
        Generate strategic advice from pipeline data.
        
        Args:
            cleaning_report: Output from ForensicCleaner (Step 2)
            training_report: Output from EliteTrainer (Step 4)
            xai_report: Output from BlackBoxBreaker (Step 5)
            feature_report: Optional output from AutoFeatureEngineer
        
        Returns:
            3-paragraph strategic advice report
        """
        # Compile context
        context = self._compile_insight_context(
            cleaning_report, training_report, xai_report, feature_report
        )
        
        # Try LLM generation
        if self.client:
            try:
                return self._generate_with_llm(context)
            except Exception as e:
                print(f"LLM generation failed: {e}, using template fallback")
        
        # Fallback to template
        return self._generate_template_insight(
            cleaning_report, training_report, xai_report, feature_report
        )
    
    def _compile_insight_context(self,
                                 cleaning_report: Dict,
                                 training_report: Dict,
                                 xai_report: Dict,
                                 feature_report: Dict) -> str:
        """Compile all data into context string for LLM."""
        
        sections = []
        
        # Anomaly Scores from Step 2
        sections.append("=== ANOMALY ANALYSIS (Step 2) ===")
        anomaly = cleaning_report.get('anomaly_detection', {})
        sections.append(f"Detection Method: {anomaly.get('method', 'IsolationForest')}")
        sections.append(f"Anomalies Detected: {anomaly.get('anomalies_detected', 0)}")
        sections.append(f"Anomaly Percentage: {anomaly.get('anomaly_percentage', 0):.1f}%")
        sections.append(f"Contamination Rate: {anomaly.get('contamination_rate', 0)}")
        
        score_range = anomaly.get('score_range', {})
        if score_range:
            sections.append(f"Anomaly Score Range: {score_range.get('min', 0):.3f} - {score_range.get('max', 1):.3f}")
            sections.append(f"Mean Anomaly Score: {score_range.get('mean', 0):.3f}")
        
        # Stability from cleaning
        stability = cleaning_report.get('stability', {})
        sections.append(f"\nDistribution Stability Score: {stability.get('stability_score', 100)}/100")
        sections.append(f"Unstable Columns: {stability.get('unstable_columns', 0)}")
        
        if stability.get('flags'):
            sections.append("Stability Issues:")
            for flag in stability['flags'][:5]:
                sections.append(f"  - {flag['column']}: {flag['mean_shift']:.1f}% shift")
        
        # Imputation info
        imputation = cleaning_report.get('imputation', {})
        sections.append(f"\nValues Imputed: {imputation.get('total_imputed', 0)}")
        if imputation.get('columns_affected'):
            sections.append(f"Columns with Missing Data: {', '.join(imputation['columns_affected'][:10])}")
        
        # Accuracy Metrics from Step 4
        sections.append("\n=== ACCURACY METRICS (Step 4) ===")
        sections.append(f"Task Type: {training_report.get('task_type', 'classification')}")
        sections.append(f"Trials per Model: {training_report.get('n_trials_per_model', 0)}")
        
        sections.append("\nTournament Rankings:")
        for rank in training_report.get('rankings', []):
            sections.append(f"  #{rank['rank']} {rank['model']}: {rank['score']:.4f}")
        
        # Best model params
        tournament_results = training_report.get('tournament_results', {})
        for model, result in list(tournament_results.items())[:3]:
            sections.append(f"\n{model} Best Parameters:")
            for param, value in list(result.get('best_params', {}).items())[:5]:
                sections.append(f"  {param}: {value}")
            sections.append(f"  Pruned Trials: {result.get('trials_pruned', 0)}/{result.get('trials_completed', 0) + result.get('trials_pruned', 0)}")
        
        # Super model
        super_model = training_report.get('super_model', {})
        if super_model.get('status') == 'success':
            sections.append("\nSuper-Model (Stacking Ensemble):")
            sections.append(f"  Score: {super_model.get('super_model_score', 0):.4f}")
            sections.append(f"  Best Single: {super_model.get('best_single_score', 0):.4f}")
            sections.append(f"  Improvement: {super_model.get('improvement_pct', 0):.2f}%")
            sections.append(f"  Base Models: {', '.join(super_model.get('base_models', []))}")
        
        # Feature Importance from Step 5
        sections.append("\n=== FEATURE IMPORTANCE (Step 5) ===")
        if xai_report.get('status') == 'success':
            sections.append(f"Samples Analyzed: {xai_report.get('samples_analyzed', 0)}")
            sections.append("\nTop 10 Features by SHAP Impact:")
            for i, feat in enumerate(xai_report.get('top_10_features', []), 1):
                sections.append(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")
            
            # Calculate concentration
            top_feats = xai_report.get('top_10_features', [])
            if len(top_feats) >= 3:
                total_importance = sum(f['importance'] for f in top_feats)
                top3_importance = sum(f['importance'] for f in top_feats[:3])
                concentration = top3_importance / total_importance if total_importance > 0 else 0
                sections.append(f"\nTop 3 Feature Concentration: {concentration*100:.1f}%")
        
        # Feature Engineering info
        if feature_report:
            sections.append("\nFeature Engineering Summary:")
            sections.append(f"  Original Features: {feature_report.get('original_features', 0)}")
            sections.append(f"  New Features Created: {feature_report.get('new_features_created', 0)}")
            sections.append(f"  Features Dropped (low MI): {feature_report.get('features_dropped', 0)}")
            
            interactions = feature_report.get('interaction_discovery', {})
            if interactions.get('top_interactions'):
                sections.append("  Top Engineered Features:")
                for feat in interactions['top_interactions'][:3]:
                    sections.append(f"    - {feat['name']}: corr={feat['correlation']:.4f}")
        
        return '\n'.join(sections)
    
    def _generate_with_llm(self, context: str) -> str:
        """Generate insight using LLM."""
        
        prompt = f"""Based on the following ML pipeline analysis data, generate a 3-paragraph strategic advice report:

{context}

Remember to structure your response as:
1. Data Quality Assessment (anomaly analysis)
2. Feature Strategy (importance analysis)
3. Model Improvement Roadmap (concrete action plan)"""

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        
        elif self.provider == "gemini":
            full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
            response = self.client.generate_content(full_prompt)
            return response.text
        
        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        
        raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_template_insight(self,
                                   cleaning_report: Dict,
                                   training_report: Dict,
                                   xai_report: Dict,
                                   feature_report: Dict) -> str:
        """Generate insight using templates (no LLM)."""
        
        paragraphs = []
        
        # Paragraph 1: Data Quality Assessment
        anomaly = cleaning_report.get('anomaly_detection', {})
        anomaly_pct = anomaly.get('anomaly_percentage', 0)
        imputation = cleaning_report.get('imputation', {})
        imputed = imputation.get('total_imputed', 0)
        stability = cleaning_report.get('stability', {})
        
        para1 = "**Data Quality Assessment:** "
        
        if anomaly_pct > 15:
            para1 += f"Your dataset shows a high anomaly rate of {anomaly_pct:.1f}%, which warrants investigation. "
            para1 += "These anomalies could represent genuine edge cases that the model should learn, or data quality issues that need correction. "
            para1 += "Recommendation: Sample 50-100 flagged anomalies and manually review them to determine if they represent valid business cases or data errors. "
        elif anomaly_pct > 5:
            para1 += f"A moderate {anomaly_pct:.1f}% of samples were flagged as anomalies. "
            para1 += "This is within acceptable bounds but worth monitoring. "
        else:
            para1 += f"Your data quality is excellent with only {anomaly_pct:.1f}% anomalies detected. "
        
        if imputed > 0:
            affected = imputation.get('columns_affected', [])
            para1 += f"Bayesian imputation was applied to {imputed} missing values across {len(affected)} columns. "
            if len(affected) <= 3:
                para1 += f"Focus data collection efforts on: {', '.join(affected)}. "
        
        if not stability.get('overall_stable', True):
            para1 += "Warning: Some distributions shifted significantly after cleaning - verify data transformations are appropriate. "
        
        paragraphs.append(para1)
        
        # Paragraph 2: Feature Strategy
        para2 = "**Feature Strategy:** "
        
        if xai_report.get('status') == 'success':
            top_feats = xai_report.get('top_10_features', [])
            
            if len(top_feats) >= 3:
                total_imp = sum(f['importance'] for f in top_feats)
                top3_imp = sum(f['importance'] for f in top_feats[:3])
                concentration = top3_imp / total_imp if total_imp > 0 else 0
                
                para2 += f"The top 3 features account for {concentration*100:.0f}% of total SHAP importance. "
                
                if concentration > 0.7:
                    para2 += f"This high concentration on {top_feats[0]['feature']}, {top_feats[1]['feature']}, and {top_feats[2]['feature']} suggests potential overfitting risk. "
                    para2 += "Consider: (1) Adding regularization, (2) Engineering interaction features from lower-ranked variables, (3) Collecting additional features to diversify signal. "
                elif concentration > 0.5:
                    para2 += "This is a healthy distribution with clear signal from key features. "
                    para2 += f"Focus feature engineering on interactions between {top_feats[0]['feature']} and mid-tier features. "
                else:
                    para2 += "Feature importance is well-distributed, indicating a robust multi-signal model. "
                
                # Low importance features
                low_feats = [f for f in top_feats if f['importance'] < 0.01]
                if low_feats:
                    para2 += f"Consider removing {len(low_feats)} low-impact features to reduce dimensionality. "
        else:
            para2 += "SHAP analysis was not completed. Run explainability analysis to understand feature contributions. "
        
        if feature_report:
            new_feats = feature_report.get('new_features_created', 0)
            if new_feats > 0:
                para2 += f"Auto-engineering created {new_feats} new features - review their SHAP contributions to identify valuable interactions. "
        
        paragraphs.append(para2)
        
        # Paragraph 3: Model Improvement Roadmap
        para3 = "**Model Improvement Roadmap:** "
        
        rankings = training_report.get('rankings', [])
        super_model = training_report.get('super_model', {})
        
        if rankings:
            best_model = rankings[0]['model']
            best_score = rankings[0]['score']
            
            para3 += f"Current best performance is {best_score:.4f} using {best_model}. "
            
            improvement = super_model.get('improvement_pct', 0) if super_model.get('status') == 'success' else 0
            
            if improvement > 1:
                para3 += f"Stacking provided +{improvement:.2f}% improvement - continue using ensemble approaches. "
            elif improvement < 0.5:
                para3 += "The stacking ensemble showed minimal improvement over single models - consider simplifying to the best single model for production. "
            
            para3 += "**Next Iteration Plan:** "
            para3 += "(1) Increase Optuna trials to 50-100 for finer hyperparameter tuning. "
            
            # Model-specific advice
            if 'XGBoost' in best_model or 'LightGBM' in best_model:
                para3 += "(2) Experiment with higher max_depth (12-15) and lower learning_rate (0.01-0.05) with more trees. "
            elif 'RandomForest' in best_model:
                para3 += "(2) Try increasing n_estimators to 500+ and experiment with max_features='sqrt' vs 'log2'. "
            else:
                para3 += "(2) Explore gradient boosting variants (XGBoost, CatBoost) if not already tested. "
            
            if anomaly_pct > 10:
                para3 += "(3) Create separate models for normal vs edge cases, or use anomaly_score as a feature. "
            else:
                para3 += "(3) Implement k-fold cross-validation with stratification to improve generalization estimates. "
        
        paragraphs.append(para3)
        
        # Format final output
        header = "=" * 60 + "\n"
        header += "STRATEGIC ADVICE FOR MODEL IMPROVEMENT\n"
        header += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        header += "=" * 60 + "\n\n"
        
        return header + "\n\n".join(paragraphs)


def generate_agent_insight(cleaning_report: Dict,
                          training_report: Dict,
                          xai_report: Dict,
                          feature_report: Dict = None,
                          provider: str = "auto") -> str:
    """
    Generate strategic advice for model improvement.
    
    Gathers:
    - Anomaly Scores from Step 2 (ForensicCleaner)
    - Feature Importance from Step 5 (BlackBoxBreaker/SHAP)
    - Accuracy Metrics from Step 4 (EliteTrainer)
    
    Args:
        cleaning_report: Output from ForensicCleaner
        training_report: Output from EliteTrainer
        xai_report: Output from BlackBoxBreaker
        feature_report: Optional output from AutoFeatureEngineer
        provider: LLM provider ('openai', 'gemini', 'groq', 'auto')
    
    Returns:
        3-paragraph strategic advice report
    """
    generator = AgentInsightGenerator(provider=provider)
    return generator.generate_insight(
        cleaning_report=cleaning_report,
        training_report=training_report,
        xai_report=xai_report,
        feature_report=feature_report
    )
