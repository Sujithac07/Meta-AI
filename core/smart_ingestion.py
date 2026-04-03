"""
Universal Smart Ingestion Engine
Domain-agnostic data analysis with heuristic semantic detection
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


class SmartIngestionEngine:
    """
    Universal data ingestion with semantic type detection.
    Works across Finance, Retail, Healthcare, Science, etc.
    """
    
    # Heuristic patterns for domain detection
    DOMAIN_PATTERNS = {
        'FINANCE': {
            'keywords': ['price', 'amount', 'revenue', 'cost', 'profit', 'loss', 'balance', 
                        'payment', 'transaction', 'invoice', 'salary', 'income', 'expense',
                        'stock', 'share', 'dividend', 'interest', 'loan', 'credit', 'debit'],
            'currency_symbols': ['$', '€', '£', '¥', '₹']
        },
        'RETAIL': {
            'keywords': ['product', 'item', 'quantity', 'order', 'customer', 'purchase',
                        'cart', 'sku', 'inventory', 'discount', 'sale', 'return', 'shipping']
        },
        'HEALTHCARE': {
            'keywords': ['patient', 'diagnosis', 'symptom', 'treatment', 'medication',
                        'blood', 'pressure', 'heart', 'bmi', 'cholesterol', 'glucose']
        },
        'TEMPORAL': {
            'keywords': ['date', 'time', 'timestamp', 'created', 'updated', 'modified',
                        'year', 'month', 'day', 'hour', 'minute', 'period', 'duration']
        },
        'GEOSPATIAL': {
            'keywords': ['latitude', 'longitude', 'lat', 'lng', 'lon', 'city', 'country',
                        'state', 'region', 'zip', 'postal', 'address', 'location']
        },
        'IDENTITY': {
            'keywords': ['id', 'uuid', 'key', 'code', 'number', 'index', 'serial']
        }
    }
    
    # Semantic type patterns
    SEMANTIC_PATTERNS = {
        'EMAIL': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'PHONE': r'^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$',
        'URL': r'^https?://[^\s]+$',
        'IP_ADDRESS': r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        'CREDIT_CARD': r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
        'SSN': r'^\d{3}-\d{2}-\d{4}$',
        'ZIP_CODE': r'^\d{5}(-\d{4})?$',
        'DATE_ISO': r'^\d{4}-\d{2}-\d{2}$',
        'PERCENTAGE': r'^-?\d+\.?\d*%$'
    }

    def __init__(self):
        self.analysis_results = {}
        self.detected_domain = None
        
    def smart_ingest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main entry point for universal smart ingestion.
        Returns comprehensive analysis with quality report.
        """
        if df is None or df.empty:
            return {"error": "Empty or None DataFrame provided"}
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "detected_domain": self._detect_domain(df),
            "column_analysis": self._analyze_all_columns(df),
            "quality_report": self._generate_quality_report(df),
            "recommendations": []
        }
        
        # Generate recommendations based on analysis
        result["recommendations"] = self._generate_recommendations(result)
        self.analysis_results = result
        
        return result
    
    def _detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect the likely domain of the dataset using heuristics."""
        column_names = ' '.join(df.columns.str.lower())
        domain_scores = {}
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            score = 0
            matched_keywords = []
            for keyword in patterns.get('keywords', []):
                if keyword in column_names:
                    score += 1
                    matched_keywords.append(keyword)
            domain_scores[domain] = {
                'score': score,
                'matched_keywords': matched_keywords
            }
        
        # Find top domain
        top_domain = max(domain_scores.items(), key=lambda x: x[1]['score'])
        self.detected_domain = top_domain[0] if top_domain[1]['score'] > 0 else 'GENERAL'
        
        return {
            'primary_domain': self.detected_domain,
            'confidence': min(top_domain[1]['score'] / 3.0, 1.0) if top_domain[1]['score'] > 0 else 0.0,
            'domain_scores': domain_scores
        }
    
    def _analyze_all_columns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze each column for semantic type and statistics."""
        columns_analysis = {}
        
        for col in df.columns:
            columns_analysis[col] = self._analyze_column(df, col)
        
        return columns_analysis
    
    def _analyze_column(self, df: pd.DataFrame, col: str) -> Dict[str, Any]:
        """Deep analysis of a single column."""
        series = df[col]
        total_rows = len(df)
        
        analysis = {
            'original_dtype': str(series.dtype),
            'semantic_type': 'UNKNOWN',
            'category': 'UNKNOWN',
            'statistics': {},
            'quality': {},
            'domain_hint': None
        }
        
        # Basic stats
        null_count = series.isna().sum()
        unique_count = series.nunique()
        
        analysis['quality'] = {
            'null_count': int(null_count),
            'null_percentage': round(null_count / total_rows * 100, 2),
            'unique_count': int(unique_count),
            'unique_percentage': round(unique_count / total_rows * 100, 2),
            'is_constant': unique_count <= 1
        }
        
        # Determine category using statistical heuristics
        analysis['category'] = self._categorize_column(series, col, total_rows, unique_count)
        
        # Detect semantic type
        analysis['semantic_type'] = self._detect_semantic_type(series, col)
        
        # Domain hint from column name
        analysis['domain_hint'] = self._get_domain_hint(col)
        
        # Type-specific statistics
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                analysis['statistics'] = {
                    'min': float(non_null.min()),
                    'max': float(non_null.max()),
                    'mean': float(non_null.mean()),
                    'median': float(non_null.median()),
                    'std': float(non_null.std()) if len(non_null) > 1 else 0,
                    'skewness': float(non_null.skew()) if len(non_null) > 2 else 0,
                    'kurtosis': float(non_null.kurtosis()) if len(non_null) > 3 else 0
                }
        elif analysis['category'] == 'CATEGORICAL':
            value_counts = series.value_counts().head(10)
            analysis['statistics'] = {
                'top_values': value_counts.to_dict(),
                'mode': str(series.mode().iloc[0]) if len(series.mode()) > 0 else None
            }
        
        return analysis
    
    def _categorize_column(self, series: pd.Series, col: str, total_rows: int, unique_count: int) -> str:
        """
        Categorize column into ID, Target, Temporal, Categorical, or Continuous.
        Uses statistical distributions and heuristics.
        """
        col_lower = col.lower()
        
        # Check for ID columns
        id_patterns = ['id', 'uuid', 'key', '_id', 'index', 'serial', 'code']
        if any(pattern in col_lower for pattern in id_patterns):
            if unique_count > total_rows * 0.9:  # >90% unique = likely ID
                return 'ID'
        
        # Check for Target columns
        target_patterns = ['target', 'label', 'class', 'outcome', 'result', 'y_', 'is_', 'has_']
        if any(pattern in col_lower for pattern in target_patterns):
            if unique_count <= 10:  # Few unique values = likely target
                return 'TARGET'
        
        # Check for Temporal columns
        temporal_patterns = ['date', 'time', 'timestamp', 'created', 'updated', 'year', 'month', 'day']
        if any(pattern in col_lower for pattern in temporal_patterns):
            return 'TEMPORAL'
        
        # Try to detect datetime
        if series.dtype == 'object':
            sample = series.dropna().head(100)
            try:
                pd.to_datetime(sample)
                return 'TEMPORAL'
            except Exception:
                pass  # nosec B110
        
        # Statistical categorization
        if pd.api.types.is_numeric_dtype(series):
            # If very few unique values relative to rows, treat as categorical
            if unique_count < total_rows * 0.05 or unique_count <= 20:
                return 'CATEGORICAL'
            else:
                return 'CONTINUOUS'
        else:
            # String columns
            if unique_count < total_rows * 0.05 or unique_count <= 50:
                return 'CATEGORICAL'
            elif unique_count > total_rows * 0.9:
                return 'ID'  # High cardinality text = likely ID
            else:
                return 'CATEGORICAL'
        
        return 'UNKNOWN'
    
    def _detect_semantic_type(self, series: pd.Series, col: str) -> str:
        """Detect specific semantic type using regex patterns and heuristics."""
        col_lower = col.lower()
        
        # Check column name heuristics first
        name_hints = {
            'email': 'EMAIL',
            'phone': 'PHONE',
            'url': 'URL',
            'website': 'URL',
            'ip': 'IP_ADDRESS',
            'credit': 'CREDIT_CARD',
            'ssn': 'SSN',
            'zip': 'ZIP_CODE',
            'postal': 'ZIP_CODE',
            'price': 'CURRENCY',
            'amount': 'CURRENCY',
            'cost': 'CURRENCY',
            'revenue': 'CURRENCY',
            'salary': 'CURRENCY',
            'age': 'AGE',
            'percentage': 'PERCENTAGE',
            'ratio': 'RATIO',
            'rate': 'RATE',
            'latitude': 'LATITUDE',
            'longitude': 'LONGITUDE',
            'lat': 'LATITUDE',
            'lng': 'LONGITUDE',
            'lon': 'LONGITUDE',
            'name': 'NAME',
            'first_name': 'FIRST_NAME',
            'last_name': 'LAST_NAME',
            'gender': 'GENDER',
            'sex': 'GENDER',
            'country': 'COUNTRY',
            'city': 'CITY',
            'state': 'STATE',
            'address': 'ADDRESS'
        }
        
        for pattern, semantic_type in name_hints.items():
            if pattern in col_lower:
                return semantic_type
        
        # Try regex pattern matching on sample values
        if series.dtype == 'object':
            sample = series.dropna().astype(str).head(50)
            for pattern_name, regex in self.SEMANTIC_PATTERNS.items():
                matches = sample.str.match(regex, na=False).sum()
                if matches > len(sample) * 0.8:  # 80% match threshold
                    return pattern_name
        
        # Numeric type detection
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                min_val, max_val = non_null.min(), non_null.max()
                
                # Age detection
                if 0 <= min_val and max_val <= 120 and 'age' in col_lower:
                    return 'AGE'
                
                # Percentage detection
                if 0 <= min_val and max_val <= 100:
                    return 'PERCENTAGE_VALUE'
                
                # Binary detection
                if set(non_null.unique()) <= {0, 1}:
                    return 'BINARY'
                
                # Year detection
                if 1900 <= min_val <= 2100 and 1900 <= max_val <= 2100:
                    if non_null.dtype in ['int64', 'int32']:
                        return 'YEAR'
        
        return 'GENERIC'
    
    def _get_domain_hint(self, col: str) -> Optional[str]:
        """Get domain hint from column name."""
        col_lower = col.lower()
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for keyword in patterns.get('keywords', []):
                if keyword in col_lower:
                    return domain
        
        return None
    
    def _generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        total_rows = len(df)
        total_cols = len(df.columns)
        total_cells = total_rows * total_cols
        
        # Calculate issues
        issues = []
        warnings = []
        
        # All-null columns
        null_cols = df.columns[df.isna().all()].tolist()
        if null_cols:
            issues.append({
                'type': 'ALL_NULL_COLUMNS',
                'severity': 'CRITICAL',
                'columns': null_cols,
                'message': f'{len(null_cols)} column(s) are completely empty'
            })
        
        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            warnings.append({
                'type': 'CONSTANT_COLUMNS',
                'severity': 'WARNING',
                'columns': constant_cols,
                'message': f'{len(constant_cols)} column(s) have constant values'
            })
        
        # High cardinality IDs
        high_card_cols = []
        for col in df.columns:
            if df[col].nunique() > total_rows * 0.95 and df[col].dtype == 'object':
                high_card_cols.append(col)
        if high_card_cols:
            warnings.append({
                'type': 'HIGH_CARDINALITY_TEXT',
                'severity': 'INFO',
                'columns': high_card_cols,
                'message': f'{len(high_card_cols)} text column(s) have very high cardinality (likely IDs)'
            })
        
        # Duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append({
                'type': 'DUPLICATE_ROWS',
                'severity': 'WARNING',
                'count': int(dup_count),
                'message': f'{dup_count} duplicate rows detected ({dup_count/total_rows*100:.1f}%)'
            })
        
        # Missing value summary
        total_missing = df.isna().sum().sum()
        missing_pct = total_missing / total_cells * 100
        
        # High missing columns (>50%)
        high_missing_cols = df.columns[df.isna().sum() / total_rows > 0.5].tolist()
        if high_missing_cols:
            warnings.append({
                'type': 'HIGH_MISSING_COLUMNS',
                'severity': 'WARNING',
                'columns': high_missing_cols,
                'message': f'{len(high_missing_cols)} column(s) have >50% missing values'
            })
        
        # Calculate overall score
        score = 100
        score -= len(issues) * 20  # Critical issues
        score -= len(warnings) * 5  # Warnings
        score -= min(missing_pct, 30)  # Missing data penalty (max 30)
        score = max(0, min(100, score))
        
        return {
            'overall_score': round(score, 1),
            'grade': self._score_to_grade(score),
            'total_cells': total_cells,
            'total_missing': int(total_missing),
            'missing_percentage': round(missing_pct, 2),
            'duplicate_rows': int(dup_count),
            'issues': issues,
            'warnings': warnings,
            'summary': {
                'critical_issues': len(issues),
                'warnings': len(warnings),
                'columns_analyzed': total_cols,
                'rows_analyzed': total_rows
            }
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        quality = analysis.get('quality_report', {})
        
        # Missing data recommendations
        if quality.get('missing_percentage', 0) > 5:
            recommendations.append(
                "Consider using Bayesian Iterative Imputation for missing values"
            )
        
        # Duplicate recommendations
        if quality.get('duplicate_rows', 0) > 0:
            recommendations.append(
                "Remove duplicate rows before training to prevent data leakage"
            )
        
        # Issue-based recommendations
        for issue in quality.get('issues', []):
            if issue['type'] == 'ALL_NULL_COLUMNS':
                recommendations.append(
                    f"Drop empty columns: {', '.join(issue['columns'])}"
                )
        
        for warning in quality.get('warnings', []):
            if warning['type'] == 'CONSTANT_COLUMNS':
                recommendations.append(
                    f"Consider dropping constant columns: {', '.join(warning['columns'])}"
                )
            elif warning['type'] == 'HIGH_CARDINALITY_TEXT':
                recommendations.append(
                    f"High cardinality columns may be IDs - exclude from features: {', '.join(warning['columns'])}"
                )
        
        # Column type recommendations
        col_analysis = analysis.get('column_analysis', {})
        id_cols = [col for col, info in col_analysis.items() if info.get('category') == 'ID']
        if id_cols:
            recommendations.append(
                f"Detected ID columns (exclude from training): {', '.join(id_cols)}"
            )
        
        target_cols = [col for col, info in col_analysis.items() if info.get('category') == 'TARGET']
        if target_cols:
            recommendations.append(
                f"Potential target columns detected: {', '.join(target_cols)}"
            )
        elif not target_cols:
            recommendations.append(
                "No target column auto-detected - please specify manually"
            )
        
        return recommendations


def smart_ingest(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convenience function for smart ingestion.
    Returns comprehensive analysis with quality report.
    """
    engine = SmartIngestionEngine()
    return engine.smart_ingest(df)


def format_ingestion_report(result: Dict[str, Any]) -> str:
    """Format ingestion result as readable text for Gradio display."""
    if 'error' in result:
        return f"Error: {result['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("SMART INGESTION REPORT")
    lines.append("=" * 60)
    
    # Dataset Overview
    shape = result.get('shape', {})
    lines.append(f"\nDataset: {shape.get('rows', 0):,} rows x {shape.get('columns', 0)} columns")
    
    # Domain Detection
    domain = result.get('detected_domain', {})
    lines.append(f"\nDetected Domain: {domain.get('primary_domain', 'UNKNOWN')}")
    lines.append(f"Confidence: {domain.get('confidence', 0)*100:.0f}%")
    
    # Quality Score
    quality = result.get('quality_report', {})
    lines.append(f"\nData Quality Score: {quality.get('overall_score', 0)}/100 (Grade: {quality.get('grade', 'N/A')})")
    lines.append(f"Missing Data: {quality.get('missing_percentage', 0):.1f}%")
    lines.append(f"Duplicate Rows: {quality.get('duplicate_rows', 0):,}")
    
    # Column Categories Summary
    col_analysis = result.get('column_analysis', {})
    categories = {}
    for col, info in col_analysis.items():
        cat = info.get('category', 'UNKNOWN')
        categories[cat] = categories.get(cat, []) + [col]
    
    lines.append("\nColumn Categories:")
    for cat, cols in categories.items():
        lines.append(f"  {cat}: {len(cols)} columns")
        if len(cols) <= 5:
            lines.append(f"    -> {', '.join(cols)}")
    
    # Issues & Warnings
    if quality.get('issues'):
        lines.append("\nCRITICAL ISSUES:")
        for issue in quality['issues']:
            lines.append(f"  ! {issue['message']}")
    
    if quality.get('warnings'):
        lines.append("\nWARNINGS:")
        for warning in quality['warnings']:
            lines.append(f"  * {warning['message']}")
    
    # Recommendations
    recommendations = result.get('recommendations', [])
    if recommendations:
        lines.append("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. {rec}")
    
    lines.append("\n" + "=" * 60)
    
    return '\n'.join(lines)
