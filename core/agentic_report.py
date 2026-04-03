"""
Agentic Report Generator
LLM-powered synthesis of pipeline logs into Technical Executive Summary
"""

from typing import Dict
from datetime import datetime
import os

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


class AgenticReportGenerator:
    """
    LLM-powered report generator that synthesizes pipeline logs
    into a Technical Executive Summary.
    """
    
    SYSTEM_PROMPT = """You are an expert ML Engineer and Data Scientist writing a Technical Executive Summary 
for a stakeholder presentation. Your task is to synthesize the provided pipeline logs into a clear, 
professional report that explains the ML pipeline decisions in business-friendly terms while maintaining 
technical accuracy.

Structure your response EXACTLY as follows:

## TECHNICAL EXECUTIVE SUMMARY

### 1. Data Preparation Rationale
Explain why certain data was imputed, what imputation method was used, and why it was appropriate.
Include the number of values imputed and which columns were affected.

### 2. Model Selection Justification
Explain why the specific ensemble/stacking model was chosen. Reference the tournament results,
which models competed, and why the winning combination outperformed alternatives.

### 3. Potential Bias & Risk Assessment
Identify any potential biases found in the analysis, anomaly patterns, data distribution issues,
or areas of concern that stakeholders should be aware of.

### 4. Key Recommendations
Provide 3-5 actionable recommendations based on the analysis.

Keep the language professional but accessible. Use bullet points where appropriate.
Include specific numbers and metrics from the logs."""

    def __init__(self, api_key: str = None, provider: str = "openai"):
        """
        Initialize report generator.
        
        Args:
            api_key: API key for LLM provider (or uses env var)
            provider: 'openai', 'groq', or 'local'
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        self.client = None
        
        if provider == "openai" and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        elif provider == "groq" and GROQ_AVAILABLE:
            self.client = Groq(api_key=self.api_key) if self.api_key else None
    
    def generate_report(self, 
                       cleaning_report: Dict,
                       training_report: Dict,
                       feature_report: Dict = None,
                       xai_report: Dict = None,
                       ingestion_report: Dict = None) -> str:
        """
        Generate Technical Executive Summary from pipeline logs.
        
        Args:
            cleaning_report: Output from ForensicCleaner
            training_report: Output from EliteTrainer
            feature_report: Optional output from AutoFeatureEngineer
            xai_report: Optional output from BlackBoxBreaker
            ingestion_report: Optional output from SmartIngestion
        
        Returns:
            Technical Executive Summary as string
        """
        # Compile context for LLM
        context = self._compile_context(
            cleaning_report, training_report, 
            feature_report, xai_report, ingestion_report
        )
        
        # Try LLM generation first
        if self.client:
            try:
                return self._generate_with_llm(context)
            except Exception as e:
                print(f"LLM generation failed: {e}, falling back to template")
        
        # Fallback to template-based generation
        return self._generate_template_report(
            cleaning_report, training_report,
            feature_report, xai_report, ingestion_report
        )
    
    def _compile_context(self, cleaning_report: Dict, training_report: Dict,
                        feature_report: Dict, xai_report: Dict,
                        ingestion_report: Dict) -> str:
        """Compile all reports into context string for LLM."""
        
        sections = []
        
        # Data Ingestion
        if ingestion_report:
            sections.append("=== DATA INGESTION LOG ===")
            quality = ingestion_report.get('quality_report', {})
            sections.append(f"Quality Score: {quality.get('overall_score', 'N/A')}/100")
            sections.append(f"Missing Data: {quality.get('missing_percentage', 0):.1f}%")
            sections.append(f"Duplicate Rows: {quality.get('duplicate_rows', 0)}")
            
            domain = ingestion_report.get('detected_domain', {})
            sections.append(f"Detected Domain: {domain.get('primary_domain', 'UNKNOWN')}")
            
            if quality.get('issues'):
                sections.append("Critical Issues:")
                for issue in quality['issues']:
                    sections.append(f"  - {issue['message']}")
            
            if quality.get('warnings'):
                sections.append("Warnings:")
                for warning in quality['warnings'][:5]:
                    sections.append(f"  - {warning['message']}")
        
        # Forensic Cleaning
        sections.append("\n=== FORENSIC CLEANING LOG ===")
        
        imputation = cleaning_report.get('imputation', {})
        sections.append(f"Imputation Method: {imputation.get('method', 'N/A')}")
        sections.append(f"Estimator Used: {imputation.get('estimator', 'N/A')}")
        sections.append(f"Total Values Imputed: {imputation.get('total_imputed', 0)}")
        
        if imputation.get('columns_affected'):
            sections.append(f"Columns Affected: {', '.join(imputation['columns_affected'][:10])}")
        
        anomaly = cleaning_report.get('anomaly_detection', {})
        sections.append(f"\nAnomaly Detection Method: {anomaly.get('method', 'N/A')}")
        sections.append(f"Anomalies Detected: {anomaly.get('anomalies_detected', 0)} ({anomaly.get('anomaly_percentage', 0)}%)")
        sections.append(f"Contamination Rate: {anomaly.get('contamination_rate', 0)}")
        
        stability = cleaning_report.get('stability', {})
        sections.append(f"\nDistribution Stability Score: {stability.get('stability_score', 100)}/100")
        sections.append(f"Columns Stable: {stability.get('stable_columns', 0)}/{stability.get('total_columns', 0)}")
        
        if stability.get('flags'):
            sections.append("Stability Warnings:")
            for flag in stability['flags'][:5]:
                sections.append(f"  - {flag['column']}: {flag['message']}")
        
        # Feature Engineering
        if feature_report:
            sections.append("\n=== FEATURE ENGINEERING LOG ===")
            sections.append(f"Original Features: {feature_report.get('original_features', 0)}")
            sections.append(f"New Features Created: {feature_report.get('new_features_created', 0)}")
            sections.append(f"Features Dropped: {feature_report.get('features_dropped', 0)}")
            sections.append(f"Final Feature Count: {feature_report.get('final_features', 0)}")
            
            interactions = feature_report.get('interaction_discovery', {})
            if interactions.get('top_interactions'):
                sections.append("\nTop Engineered Features:")
                for feat in interactions['top_interactions'][:5]:
                    sections.append(f"  - {feat['name']}: correlation={feat['correlation']:.4f} ({feat['type']})")
            
            mi_filter = feature_report.get('information_filter', {})
            if mi_filter.get('top_mi_scores'):
                sections.append("\nTop Mutual Information Scores:")
                for feat in mi_filter['top_mi_scores'][:5]:
                    sections.append(f"  - {feat['feature']}: MI={feat['mi_score']:.4f}")
        
        # Training Tournament
        sections.append("\n=== ELITE TRAINING TOURNAMENT LOG ===")
        sections.append(f"Task Type: {training_report.get('task_type', 'N/A')}")
        sections.append(f"Trials per Model: {training_report.get('n_trials_per_model', 0)}")
        sections.append(f"Competitors: {', '.join(training_report.get('competitors', []))}")
        
        sections.append("\nTournament Rankings:")
        for rank in training_report.get('rankings', []):
            medal = "WINNER" if rank['rank'] == 1 else f"#{rank['rank']}"
            sections.append(f"  {medal}: {rank['model']} (Score: {rank['score']:.4f})")
        
        # Tournament details
        sections.append("\nOptimization Details:")
        for model, result in training_report.get('tournament_results', {}).items():
            sections.append(f"  {model}:")
            sections.append(f"    Best Score: {result['best_score']:.4f}")
            sections.append(f"    Trials Completed: {result['trials_completed']}")
            sections.append(f"    Trials Pruned: {result['trials_pruned']}")
        
        # Super Model
        super_model = training_report.get('super_model', {})
        if super_model.get('status') == 'success':
            sections.append("\nSuper-Model (Stacking Ensemble):")
            sections.append(f"  Base Models: {', '.join(super_model.get('base_models', []))}")
            sections.append(f"  Meta-Learner: {super_model.get('meta_learner', 'N/A')}")
            sections.append(f"  Super-Model Score: {super_model.get('super_model_score', 0):.4f}")
            sections.append(f"  Best Single Score: {super_model.get('best_single_score', 0):.4f}")
            sections.append(f"  Improvement: {super_model.get('improvement', 0):.4f} ({super_model.get('improvement_pct', 0):.2f}%)")
        
        # XAI
        if xai_report and xai_report.get('status') == 'success':
            sections.append("\n=== EXPLAINABILITY ANALYSIS LOG ===")
            sections.append(f"Samples Analyzed: {xai_report.get('samples_analyzed', 0)}")
            
            if xai_report.get('top_10_features'):
                sections.append("\nTop Feature Impacts (SHAP):")
                for feat in xai_report['top_10_features']:
                    sections.append(f"  - {feat['feature']}: {feat['importance']:.4f}")
        
        return '\n'.join(sections)
    
    def _generate_with_llm(self, context: str) -> str:
        """Generate report using LLM."""
        
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a Technical Executive Summary from these pipeline logs:\n\n{context}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
            
        elif self.provider == "groq":
            response = self.client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Generate a Technical Executive Summary from these pipeline logs:\n\n{context}"}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        
        raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_template_report(self, cleaning_report: Dict, training_report: Dict,
                                  feature_report: Dict, xai_report: Dict,
                                  ingestion_report: Dict) -> str:
        """Generate report using templates (fallback when no LLM)."""
        
        lines = []
        lines.append("=" * 70)
        lines.append("TECHNICAL EXECUTIVE SUMMARY")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        # Section 1: Data Preparation Rationale
        lines.append("\n## 1. DATA PREPARATION RATIONALE")
        lines.append("-" * 50)
        
        imputation = cleaning_report.get('imputation', {})
        total_imputed = imputation.get('total_imputed', 0)
        method = imputation.get('method', 'Unknown')
        estimator = imputation.get('estimator', 'Unknown')
        affected_cols = imputation.get('columns_affected', [])
        
        if total_imputed > 0:
            lines.append(f"\nImputation was performed on {total_imputed} missing values using {method}.")
            lines.append(f"The {estimator} estimator was chosen because it models missing values")
            lines.append("as a function of all other features, providing more accurate predictions")
            lines.append("than simple mean/median imputation.")
            
            if affected_cols:
                lines.append(f"\nAffected columns ({len(affected_cols)}):")
                for col in affected_cols[:10]:
                    missing_count = imputation.get('missing_before', {}).get(col, 0)
                    lines.append(f"  - {col}: {missing_count} values imputed")
        else:
            lines.append("\nNo missing values were detected in the dataset.")
            lines.append("Data completeness was verified before proceeding to analysis.")
        
        # Stability info
        stability = cleaning_report.get('stability', {})
        if stability.get('flags'):
            lines.append("\nDistribution Stability Concerns:")
            for flag in stability['flags'][:3]:
                lines.append(f"  - {flag['column']}: Mean shifted by {flag['mean_shift']:.1f}%")
        
        # Section 2: Model Selection Justification
        lines.append("\n\n## 2. MODEL SELECTION JUSTIFICATION")
        lines.append("-" * 50)
        
        rankings = training_report.get('rankings', [])
        super_model = training_report.get('super_model', {})
        competitors = training_report.get('competitors', [])
        
        lines.append(f"\nA tournament was conducted among {len(competitors)} model architectures:")
        for comp in competitors:
            lines.append(f"  - {comp}")
        
        if rankings:
            lines.append("\nTournament Results:")
            for rank in rankings[:3]:
                lines.append(f"  #{rank['rank']} {rank['model']}: {rank['score']:.4f}")
        
        if super_model.get('status') == 'success':
            base_models = super_model.get('base_models', [])
            meta = super_model.get('meta_learner', 'LogisticRegression')
            super_score = super_model.get('super_model_score', 0)
            best_single = super_model.get('best_single_score', 0)
            improvement = super_model.get('improvement_pct', 0)
            
            lines.append("\nFinal Model Selection: STACKING ENSEMBLE")
            lines.append(f"  Base Models: {', '.join(base_models)}")
            lines.append(f"  Meta-Learner: {meta}")
            lines.append("\nRationale:")
            lines.append(f"  The stacking ensemble was chosen because it achieved {super_score:.4f} accuracy,")
            if improvement > 0:
                lines.append(f"  outperforming the best single model ({best_single:.4f}) by {improvement:.2f}%.")
            else:
                lines.append("  matching the best single model performance while providing ensemble robustness.")
            lines.append(f"  By combining predictions from {len(base_models)} diverse algorithms,")
            lines.append("  the ensemble reduces variance and captures complementary patterns.")
        
        # Section 3: Bias & Risk Assessment
        lines.append("\n\n## 3. POTENTIAL BIAS & RISK ASSESSMENT")
        lines.append("-" * 50)
        
        risks_found = []
        
        # Check anomaly rate
        anomaly = cleaning_report.get('anomaly_detection', {})
        anomaly_pct = anomaly.get('anomaly_percentage', 0)
        if anomaly_pct > 15:
            risks_found.append(f"High anomaly rate ({anomaly_pct}%) may indicate data quality issues or outlier populations")
        
        # Check stability
        if stability.get('flags'):
            risks_found.append(f"Data distribution shifted during cleaning for {len(stability['flags'])} columns")
        
        # Check feature importance concentration
        if xai_report and xai_report.get('top_10_features'):
            top_feats = xai_report['top_10_features']
            if len(top_feats) >= 2:
                top_importance = top_feats[0]['importance']
                second_importance = top_feats[1]['importance']
                if top_importance > second_importance * 3:
                    risks_found.append(f"Model heavily relies on single feature '{top_feats[0]['feature']}' - potential overfitting risk")
        
        # Check pruned trials
        for model, result in training_report.get('tournament_results', {}).items():
            pruned = result.get('trials_pruned', 0)
            total = result.get('trials_completed', 0) + pruned
            if total > 0 and pruned / total > 0.5:
                risks_found.append(f"{model} had {pruned}/{total} trials pruned - hyperparameter space may need refinement")
        
        # Check data quality issues from ingestion
        if ingestion_report:
            quality = ingestion_report.get('quality_report', {})
            if quality.get('overall_score', 100) < 70:
                risks_found.append(f"Initial data quality score was low ({quality.get('overall_score')})")
            
            for issue in quality.get('issues', []):
                risks_found.append(issue['message'])
        
        if risks_found:
            lines.append("\nIdentified Risks:")
            for i, risk in enumerate(risks_found, 1):
                lines.append(f"  {i}. {risk}")
        else:
            lines.append("\nNo significant biases or risks were identified in the analysis.")
            lines.append("The model appears robust based on the available data.")
        
        lines.append("\nGeneral Considerations:")
        lines.append("  - Ensure training data is representative of production distribution")
        lines.append("  - Monitor for data drift in production deployment")
        lines.append("  - Validate model fairness across relevant demographic groups")
        
        # Section 4: Recommendations
        lines.append("\n\n## 4. KEY RECOMMENDATIONS")
        lines.append("-" * 50)
        
        recommendations = []
        
        # Based on analysis
        if anomaly_pct > 10:
            recommendations.append("Investigate anomalous samples before deployment - they may represent edge cases or data errors")
        
        if imputation.get('total_imputed', 0) > 0:
            recommendations.append("Monitor for increased missing data in production - may indicate upstream data issues")
        
        if super_model.get('improvement_pct', 0) < 1:
            recommendations.append("Consider using the simpler single model in production for faster inference")
        else:
            recommendations.append("Use the stacking ensemble for maximum accuracy; optimize inference if latency is critical")
        
        if feature_report:
            new_features = feature_report.get('new_features_created', 0)
            if new_features > 0:
                recommendations.append(f"Document the {new_features} engineered features for reproducibility")
        
        recommendations.append("Set up automated drift detection using the DeploymentGuard module")
        recommendations.append("Establish retraining triggers based on performance degradation thresholds")
        
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("\n" + "=" * 70)
        lines.append("END OF TECHNICAL EXECUTIVE SUMMARY")
        lines.append("=" * 70)
        
        return '\n'.join(lines)


def agent_report_generator(cleaning_report: Dict,
                          training_report: Dict,
                          feature_report: Dict = None,
                          xai_report: Dict = None,
                          ingestion_report: Dict = None,
                          api_key: str = None,
                          provider: str = "openai") -> str:
    """
    Convenience function to generate Technical Executive Summary.
    
    Args:
        cleaning_report: Output from ForensicCleaner
        training_report: Output from EliteTrainer
        feature_report: Optional output from AutoFeatureEngineer
        xai_report: Optional output from BlackBoxBreaker
        ingestion_report: Optional output from SmartIngestion
        api_key: Optional API key for LLM
        provider: 'openai', 'groq', or 'local'
    
    Returns:
        Technical Executive Summary string
    """
    generator = AgenticReportGenerator(api_key=api_key, provider=provider)
    return generator.generate_report(
        cleaning_report=cleaning_report,
        training_report=training_report,
        feature_report=feature_report,
        xai_report=xai_report,
        ingestion_report=ingestion_report
    )
