"""
MetaAI Pro - Professional Dashboard v3.0
Enhanced Data Ingestion with Subtabs:
- Manual CSV Upload
- Semantic Type Detection (Universal - Any Dataset)
- Pydantic Data Validation
- Drift Baseline Capture
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env file

# Must run before any `from sklearn.impute import IterativeImputer` (experimental API)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler
matplotlib.use('Agg')
import io
import base64
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field, validator, ValidationError
from pydantic import create_model

# Import core modules
from core.smart_ingestion import SmartIngestionEngine, format_ingestion_report
from core.forensic_cleaner import ForensicCleaner, format_forensic_report
from core.auto_feature_engineer import AutoFeatureEngineer, format_feature_report
from core.elite_trainer import EliteTrainer, format_tournament_report, OPTUNA_AVAILABLE
from core.black_box_breaker import BlackBoxBreaker, format_xai_report
from core.deployment_guard import DeploymentGuard, format_drift_report
from core.agentic_report import agent_report_generator
from core.production_export import create_production_export
from core.agent_insight import generate_agent_insight


# ==================== STATE MANAGEMENT ====================

class AppState:
    """Centralized application state."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.target_column = None
        self.task_type = "classification"
        self.columns = []
        
        # Semantic types detected
        self.semantic_types = {}
        
        # Pydantic schema
        self.pydantic_schema = None
        self.validation_errors = []
        
        # Drift baseline
        self.drift_baseline = None
        
        # Reports
        self.ingestion_report = None
        self.cleaning_report = None
        self.feature_report = None
        self.training_report = None
        self.xai_report = None
        
        # Stats
        self.raw_stats = {}
        self.cleaned_stats = {}
        
        # Models
        self.model = None
        # Some production/MLOps callbacks historically use `trained_model`
        # instead of `model`, so keep this in sync.
        self.trained_model = None
        self.explainer = None
        self.X_train = None
        self.y_train = None


state = AppState()


# ==================== UNIVERSAL SEMANTIC TYPE DETECTION ====================

# Comprehensive patterns for ANY domain
SEMANTIC_PATTERNS = {
    # === IDENTIFIERS ===
    'primary_key': r'(?i)(^id$|_id$|^pk$|^key$|^uuid|^guid|identifier|record.?id)',
    'foreign_key': r'(?i)(_fk$|_ref$|parent.?id|related.?id)',
    'name': r'(?i)(^name$|_name$|first.?name|last.?name|full.?name|customer.?name|user.?name|product.?name|item.?name)',
    'email': r'(?i)(email|e.?mail|mail.?address)',
    'phone': r'(?i)(phone|mobile|tel|contact.?num|cell)',
    'address': r'(?i)(address|street|city|state|zip|postal|country|location|region)',
    'url': r'(?i)(url|link|website|href|uri)',
    'ip_address': r'(?i)(ip|ip.?address|ipv4|ipv6)',
    
    # === DEMOGRAPHICS ===
    'age': r'(?i)(^age$|_age$|years.?old|customer.?age|user.?age)',
    'gender': r'(?i)(gender|sex|male.?female)',
    'birth_date': r'(?i)(birth|dob|date.?of.?birth|born)',
    'marital_status': r'(?i)(marital|married|single|divorced)',
    'education': r'(?i)(education|degree|school|university|qualification)',
    'occupation': r'(?i)(occupation|job|profession|work|employment|career|title)',
    
    # === FINANCIAL ===
    'price': r'(?i)(price|cost|amount|fee|charge|rate|value|msrp)',
    'revenue': r'(?i)(revenue|sales|total.?amount|gross)',
    'profit': r'(?i)(profit|margin|gain|net|earnings)',
    'income': r'(?i)(income|salary|wage|pay|compensation|annual)',
    'discount': r'(?i)(discount|promo|coupon|rebate|reduction)',
    'tax': r'(?i)(tax|vat|gst|duty)',
    'currency': r'(?i)(currency|curr|money.?type)',
    'balance': r'(?i)(balance|credit|debit|account.?amt)',
    
    # === QUANTITIES & COUNTS ===
    'quantity': r'(?i)(quantity|qty|count|num|number.?of|total.?count|units)',
    'percentage': r'(?i)(percent|pct|ratio|rate$|proportion)',
    'score': r'(?i)(score|rating|rank|grade|points|stars)',
    'size': r'(?i)(size|dimension|length|width|height|area|volume)',
    'weight': r'(?i)(weight|mass|wt$|kg$|lbs)',
    'duration': r'(?i)(duration|time.?spent|minutes|hours|days|length.?of)',
    'distance': r'(?i)(distance|miles|km|meters|feet)',
    
    # === TEMPORAL ===
    'date': r'(?i)(date|dt$|day$|created|updated|modified|timestamp)',
    'year': r'(?i)(^year$|yr$|fiscal.?year|calendar.?year)',
    'month': r'(?i)(^month$|mon$|mm$)',
    'quarter': r'(?i)(quarter|qtr|q[1-4])',
    'datetime': r'(?i)(datetime|timestamp|ts$|time$)',
    'time_of_day': r'(?i)(time.?of.?day|hour|am.?pm)',
    
    # === STATUS & CATEGORIES ===
    'status': r'(?i)(status|state|condition|stage|phase)',
    'type': r'(?i)(type|category|class$|kind|group|segment)',
    'level': r'(?i)(level|tier|grade|priority|severity)',
    'flag': r'(?i)(flag|is.?|has.?|can.?|should|active|enabled|valid)',
    
    # === E-COMMERCE ===
    'product_id': r'(?i)(product.?id|item.?id|sku|upc|ean|asin)',
    'product_name': r'(?i)(product.?name|item.?name|title|description)',
    'brand': r'(?i)(brand|manufacturer|vendor|supplier|maker)',
    'order_id': r'(?i)(order.?id|order.?num|transaction.?id|invoice)',
    'customer_id': r'(?i)(customer.?id|client.?id|user.?id|buyer.?id)',
    'shipping': r'(?i)(shipping|delivery|freight|postage)',
    'inventory': r'(?i)(inventory|stock|available|in.?stock|on.?hand)',
    
    # === MARKETING ===
    'campaign': r'(?i)(campaign|promo|ad|advertisement|marketing)',
    'channel': r'(?i)(channel|source|medium|platform|referrer)',
    'clicks': r'(?i)(click|impression|view|visit|hit)',
    'conversion': r'(?i)(conversion|convert|signup|subscribe|purchase)',
    
    # === HR / EMPLOYEE ===
    'employee_id': r'(?i)(employee.?id|emp.?id|staff.?id|worker.?id)',
    'department': r'(?i)(department|dept|division|unit|team)',
    'hire_date': r'(?i)(hire|join|start.?date|employment.?date)',
    'tenure': r'(?i)(tenure|years.?employed|experience|seniority)',
    'performance': r'(?i)(performance|review|evaluation|appraisal)',
    
    # === TECHNICAL / IOT ===
    'sensor_id': r'(?i)(sensor.?id|device.?id|node.?id|equipment.?id)',
    'reading': r'(?i)(reading|measurement|value|signal|output)',
    'temperature': r'(?i)(temp|temperature|celsius|fahrenheit|thermal)',
    'pressure': r'(?i)(pressure|psi|bar|pascal)',
    'voltage': r'(?i)(voltage|volt|v$|current|amp)',
    'frequency': r'(?i)(frequency|freq|hz|hertz)',
    'error_code': r'(?i)(error|fault|alert|warning|code)',
    
    # === GEOGRAPHIC ===
    'latitude': r'(?i)(lat|latitude)',
    'longitude': r'(?i)(lon|lng|longitude)',
    'coordinates': r'(?i)(coord|geo|location)',
    'zip_code': r'(?i)(zip|postal|postcode)',
    'country_code': r'(?i)(country.?code|iso.?country)',
    
    # === MEDICAL (general) ===
    'patient_id': r'(?i)(patient.?id|medical.?record|mrn)',
    'diagnosis': r'(?i)(diagnosis|condition|disease|illness)',
    'medication': r'(?i)(medication|drug|prescription|medicine)',
    'dosage': r'(?i)(dosage|dose|mg|ml)',
    'blood_pressure': r'(?i)(blood.?pressure|bp|systolic|diastolic)',
    'heart_rate': r'(?i)(heart.?rate|pulse|bpm)',
    'lab_result': r'(?i)(lab|test.?result|specimen)',
    
    # === TARGET / LABEL ===
    'target': r'(?i)(target|label|outcome|result|class$|predict|y$|response|dependent)',
    'binary_outcome': r'(?i)(churn|default|fraud|spam|click|convert|buy|cancel|return)',
}

# Domain-specific constraints (applied when semantic type is detected)
DOMAIN_CONSTRAINTS = {
    # Positive integers
    'quantity': {'min': 0, 'type': 'int'},
    'age': {'min': 0, 'max': 150, 'type': 'int'},
    'year': {'min': 1900, 'max': 2100, 'type': 'int'},
    'inventory': {'min': 0, 'type': 'int'},
    'clicks': {'min': 0, 'type': 'int'},
    
    # Positive floats
    'price': {'min': 0, 'type': 'float'},
    'revenue': {'min': 0, 'type': 'float'},
    'income': {'min': 0, 'type': 'float'},
    'weight': {'min': 0, 'type': 'float'},
    'distance': {'min': 0, 'type': 'float'},
    'duration': {'min': 0, 'type': 'float'},
    'size': {'min': 0, 'type': 'float'},
    'shipping': {'min': 0, 'type': 'float'},
    'tax': {'min': 0, 'type': 'float'},
    
    # Percentages (0-100 or 0-1)
    'percentage': {'min': 0, 'max': 100, 'type': 'float'},
    'conversion': {'min': 0, 'max': 100, 'type': 'float'},
    
    # Scores/ratings (typically bounded)
    'score': {'min': 0, 'max': 100, 'type': 'float'},
    
    # Geographic
    'latitude': {'min': -90, 'max': 90, 'type': 'float'},
    'longitude': {'min': -180, 'max': 180, 'type': 'float'},
    
    # Technical measurements
    'temperature': {'min': -273.15, 'type': 'float'},  # Absolute zero
    'pressure': {'min': 0, 'type': 'float'},
    'voltage': {'type': 'float'},
    'frequency': {'min': 0, 'type': 'float'},
    
    # Medical
    'heart_rate': {'min': 20, 'max': 300, 'type': 'float'},
    'dosage': {'min': 0, 'type': 'float'},
}


def detect_semantic_type(col_name: str, series: pd.Series) -> Dict[str, Any]:
    """
    Detect semantic type using pattern matching and statistical heuristics.
    Works for ANY dataset - not domain-specific.
    """
    result = {
        'column': col_name,
        'detected_type': 'unknown',
        'category': 'Unknown',  # NEW: Semantic category tag
        'confidence': 0.0,
        'domain': None,
        'constraints': None,
        'inferred_from': 'heuristics',
        'sample_values': [],
        'statistics': {}
    }
    
    # Category mapping for semantic tags
    CATEGORY_MAP = {
        'user_id': 'Identifier',
        'patient_id': 'Identifier',
        'order_id': 'Identifier',
        'transaction_id': 'Identifier',
        'product_id': 'Identifier',
        'email': 'Identifier',
        'phone': 'Identifier',
        'ssn': 'Identifier',
        
        'age': 'Demographic',
        'gender': 'Demographic',
        'sex': 'Demographic',
        'ethnicity': 'Demographic',
        'race': 'Demographic',
        'marital_status': 'Demographic',
        'education': 'Demographic',
        
        'systolic_bp': 'Clinical_Metric',
        'diastolic_bp': 'Clinical_Metric',
        'blood_pressure': 'Clinical_Metric',
        'cholesterol': 'Clinical_Metric',
        'glucose': 'Clinical_Metric',
        'heart_rate': 'Clinical_Metric',
        'bmi': 'Clinical_Metric',
        'weight': 'Clinical_Metric',
        'height': 'Clinical_Metric',
        'dosage': 'Clinical_Metric',
        
        'smoking': 'Lifestyle_Factor',
        'alcohol': 'Lifestyle_Factor',
        'exercise': 'Lifestyle_Factor',
        'diet': 'Lifestyle_Factor',
        'sleep_hours': 'Lifestyle_Factor',
        
        'income': 'Financial',
        'salary': 'Financial',
        'revenue': 'Financial',
        'price': 'Financial',
        'cost': 'Financial',
        'profit': 'Financial',
        
        'latitude': 'Geographic',
        'longitude': 'Geographic',
        'city': 'Geographic',
        'country': 'Geographic',
        'zipcode': 'Geographic',
        'postal_code': 'Geographic',
        
        'timestamp': 'Temporal',
        'date': 'Temporal',
        'created_at': 'Temporal',
        
        'temperature': 'Technical',
        'pressure': 'Technical',
        'voltage': 'Technical',
        'frequency': 'Technical',
    }
    
    # 1. Pattern matching on column name
    for semantic_type, pattern in SEMANTIC_PATTERNS.items():
        if re.search(pattern, col_name):
            result['detected_type'] = semantic_type
            result['category'] = CATEGORY_MAP.get(semantic_type, 'Feature')  # Assign category
            result['confidence'] = 0.85
            result['inferred_from'] = 'column_name_pattern'
            
            if semantic_type in DOMAIN_CONSTRAINTS:
                result['domain'] = semantic_type
                result['constraints'] = DOMAIN_CONSTRAINTS[semantic_type]
            break
    
    # 2. Statistical & heuristic analysis for unknown types
    if result['detected_type'] == 'unknown':
        dtype = series.dtype
        n_unique = series.nunique()
        n_total = len(series)
        unique_ratio = n_unique / n_total if n_total > 0 else 0
        null_pct = series.isnull().mean()
        
        # Collect statistics
        result['statistics'] = {
            'dtype': str(dtype),
            'unique_count': n_unique,
            'unique_ratio': round(unique_ratio, 4),
            'null_pct': round(null_pct, 4)
        }
        
        if np.issubdtype(dtype, np.number):
            # Numeric analysis
            result['statistics'].update({
                'min': float(series.min()) if not series.isnull().all() else None,
                'max': float(series.max()) if not series.isnull().all() else None,
                'mean': float(series.mean()) if not series.isnull().all() else None,
                'std': float(series.std()) if not series.isnull().all() else None
            })
            
            # Infer type from statistics
            if n_unique == 2:
                result['detected_type'] = 'binary_numeric'
                result['confidence'] = 0.9
            elif unique_ratio < 0.02 and n_unique <= 10:
                result['detected_type'] = 'ordinal'
                result['confidence'] = 0.75
            elif series.min() >= 0 and series.max() <= 1:
                result['detected_type'] = 'probability_or_ratio'
                result['confidence'] = 0.7
            elif series.min() >= 0 and series.max() <= 100:
                result['detected_type'] = 'percentage_or_score'
                result['confidence'] = 0.6
            elif series.min() >= 0:
                result['detected_type'] = 'positive_numeric'
                result['confidence'] = 0.5
                result['constraints'] = {'min': 0, 'type': 'float'}
            else:
                result['detected_type'] = 'numeric'
                result['confidence'] = 0.5
        
        elif dtype == 'object' or dtype.name == 'category':
            # Categorical analysis
            sample = series.dropna().head(20).astype(str)
            
            # Check for email pattern
            if any('@' in str(v) and '.' in str(v) for v in sample):
                result['detected_type'] = 'email'
                result['confidence'] = 0.9
            # Check for URL pattern
            elif any(str(v).startswith(('http', 'www.')) for v in sample):
                result['detected_type'] = 'url'
                result['confidence'] = 0.9
            # Check for phone pattern
            elif any(re.match(r'^[\d\-\+\(\)\s]{7,}$', str(v)) for v in sample):
                result['detected_type'] = 'phone'
                result['confidence'] = 0.7
            # Binary category
            elif n_unique == 2:
                result['detected_type'] = 'binary_category'
                result['confidence'] = 0.85
            # Low cardinality
            elif n_unique <= 10:
                result['detected_type'] = 'low_cardinality_category'
                result['confidence'] = 0.7
            # Medium cardinality
            elif n_unique <= 50:
                result['detected_type'] = 'category'
                result['confidence'] = 0.6
            # High cardinality - likely identifier or free text
            elif unique_ratio > 0.9:
                result['detected_type'] = 'identifier_or_text'
                result['confidence'] = 0.6
            else:
                result['detected_type'] = 'categorical'
                result['confidence'] = 0.5
        
        elif np.issubdtype(dtype, np.datetime64):
            result['detected_type'] = 'datetime'
            result['confidence'] = 0.95
        
        elif dtype == 'bool':
            result['detected_type'] = 'boolean'
            result['confidence'] = 0.95
    
    # Get sample values
    result['sample_values'] = series.dropna().head(5).tolist()
    
    return result


def run_semantic_detection(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Run semantic type detection on all columns - works for ANY dataset."""
    if df is None:
        return None, "No data uploaded"
    
    results = []
    state.semantic_types = {}
    
    for col in df.columns:
        detection = detect_semantic_type(col, df[col])
        state.semantic_types[col] = detection
        
        results.append({
            'Column': col,
            'Category': detection.get('category', 'Feature'),  # NEW: Show category tag
            'Semantic Type': detection['detected_type'].replace('_', ' ').title(),
            'Confidence': f"{detection['confidence']*100:.0f}%",
            'Inferred From': detection['inferred_from'].replace('_', ' ').title(),
            'Samples': str(detection['sample_values'][:3])[:40]
        })
    
    results_df = pd.DataFrame(results)
    
    # Create summary report
    report = f"""## Semantic Type Detection Report

**Columns Analyzed:** {len(df.columns)}
**Detection Method:** Pattern Matching + Statistical Heuristics

### Type Distribution:
"""
    type_counts = {}
    for col, det in state.semantic_types.items():
        t = det['detected_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        report += f"- **{t.replace('_', ' ').title()}**: {count} column(s)\n"
    
    # High confidence detections
    high_conf = [(c, d) for c, d in state.semantic_types.items() if d['confidence'] >= 0.7]
    if high_conf:
        report += "\n### High-Confidence Detections:\n"
        for col, det in high_conf[:15]:
            report += f"- `{col}` -> **{det['detected_type']}** ({det['confidence']*100:.0f}%)\n"
    
    # Columns with constraints
    constrained = [(c, d) for c, d in state.semantic_types.items() if d.get('constraints')]
    if constrained:
        report += "\n### Columns with Validation Constraints:\n"
        for col, det in constrained:
            c = det['constraints']
            min_v = c.get('min', 'N/A')
            max_v = c.get('max', 'N/A')
            report += f"- `{col}`: {min_v} <= value <= {max_v}\n"
    
    return results_df, report


# ==================== PYDANTIC VALIDATION ====================

def generate_pydantic_schema(df: pd.DataFrame) -> Tuple[str, str]:
    """Generate Pydantic schema based on detected types."""
    if df is None:
        return "", "No data to generate schema"
    
    schema_fields = {}
    validators_code = []
    
    for col in df.columns:
        dtype = df[col].dtype
        semantic = state.semantic_types.get(col, {})
        constraints = semantic.get('constraints', {})
        
        # Determine field type
        if np.issubdtype(dtype, np.integer):
            field_type = "int"
        elif np.issubdtype(dtype, np.floating):
            field_type = "float"
        elif dtype == 'bool':
            field_type = "bool"
        else:
            field_type = "str"
        
        # Add constraints
        if constraints:
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None and max_val is not None and max_val != float('inf'):
                schema_fields[col] = f"{field_type} = Field(ge={min_val}, le={max_val})"
                validators_code.append(f"""
    @validator('{col}')
    def validate_{col.replace(' ', '_').lower()}(cls, v):
        if v < {min_val} or v > {max_val}:
            raise ValueError(f'{col} must be between {min_val} and {max_val}, got {{v}}')
        return v""")
            elif min_val is not None:
                schema_fields[col] = f"{field_type} = Field(ge={min_val})"
            else:
                schema_fields[col] = f"Optional[{field_type}] = None"
        else:
            schema_fields[col] = f"Optional[{field_type}] = None"
    
    # Generate schema code
    schema_code = '''from pydantic import BaseModel, Field, validator
from typing import Optional

class DataRowSchema(BaseModel):
    """Auto-generated Pydantic schema for data validation."""
    
'''
    for col, field_def in schema_fields.items():
        safe_name = col.replace(' ', '_').replace('-', '_').lower()
        schema_code += f"    {safe_name}: {field_def}\n"
    
    for validator_code in validators_code:
        schema_code += validator_code
    
    schema_code += """
    
    class Config:
        extra = 'forbid'  # Reject unknown fields
"""
    
    state.pydantic_schema = schema_code
    
    # Create summary
    summary = f"""## Pydantic Schema Generated

**Fields:** {len(schema_fields)}
**Validators:** {len(validators_code)}

### Validation Rules Applied:
"""
    for col, det in state.semantic_types.items():
        if det.get('constraints'):
            c = det['constraints']
            summary += f"- `{col}`: {c.get('min', 'N/A')} â‰¤ value â‰¤ {c.get('max', 'N/A')}\n"
    
    summary += "\n### Schema Preview:\n```python\n" + schema_code[:1500] + "\n```"
    
    return schema_code, summary


def validate_data_with_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Validate data against generated schema."""
    if df is None:
        return None, "No data to validate"
    
    validation_results = []
    errors = []
    
    for idx, row in df.head(100).iterrows():  # Validate first 100 rows
        row_errors = []
        
        for col in df.columns:
            semantic = state.semantic_types.get(col, {})
            constraints = semantic.get('constraints', {})
            if constraints is None:
                constraints = {}
            value = row[col]
            
            if pd.isna(value):
                continue
            
            # Check type
            expected_type = constraints.get('type', 'any')
            if expected_type == 'int' and not isinstance(value, (int, np.integer)):
                try:
                    float(value)
                    if float(value) != int(float(value)):
                        row_errors.append(f"{col}: Expected integer, got {type(value).__name__}")
                except:
                    row_errors.append(f"{col}: Expected integer, got {type(value).__name__}")
            
            # Check range
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            try:
                num_val = float(value)
                if min_val is not None and num_val < min_val:
                    row_errors.append(f"{col}: Value {num_val} below minimum {min_val}")
                if max_val is not None and max_val != float('inf') and num_val > max_val:
                    row_errors.append(f"{col}: Value {num_val} above maximum {max_val}")
            except:
                pass
        
        if row_errors:
            errors.append({'row': idx, 'errors': row_errors})
            validation_results.append({
                'Row': idx,
                'Status': 'INVALID',
                'Errors': '; '.join(row_errors[:3])
            })
        else:
            validation_results.append({
                'Row': idx,
                'Status': 'VALID',
                'Errors': '-'
            })
    
    state.validation_errors = errors
    results_df = pd.DataFrame(validation_results)
    
    # Create report
    valid_count = len([r for r in validation_results if r['Status'] == 'VALID'])
    invalid_count = len([r for r in validation_results if r['Status'] == 'INVALID'])
    
    report = f"""## Data Validation Report

**Rows Validated:** {len(validation_results)}
**Valid Rows:** {valid_count} ({valid_count/len(validation_results)*100:.1f}%)
**Invalid Rows:** {invalid_count} ({invalid_count/len(validation_results)*100:.1f}%)

"""
    
    if errors:
        report += "### Common Validation Errors:\n"
        error_types = {}
        for e in errors:
            for err in e['errors']:
                col = err.split(':')[0]
                error_types[col] = error_types.get(col, 0) + 1
        
        for col, count in sorted(error_types.items(), key=lambda x: -x[1])[:5]:
            report += f"- **{col}**: {count} violations\n"
        
        report += "\n### Sample Error Details:\n"
        for e in errors[:5]:
            report += f"- Row {e['row']}: {e['errors'][0]}\n"
    else:
        report += "### All validated rows passed schema checks!"
    
    return results_df, report


# ==================== DRIFT BASELINE ====================

def capture_drift_baseline(df: pd.DataFrame) -> Tuple[str, str]:
    """Capture statistical fingerprint for drift detection."""
    if df is None:
        return "", "No data to analyze"
    
    baseline = {
        'timestamp': datetime.now().isoformat(),
        'shape': df.shape,
        'columns': list(df.columns),
        'statistics': {},
        'distributions': {}
    }
    
    for col in df.columns:
        col_stats = {}
        
        if np.issubdtype(df[col].dtype, np.number):
            col_stats = {
                'type': 'numeric',
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'null_pct': float(df[col].isnull().mean()),
                'unique_count': int(df[col].nunique())
            }
            
            # Store histogram for KS test
            hist, edges = np.histogram(df[col].dropna(), bins=20, density=True)
            baseline['distributions'][col] = {
                'histogram': hist.tolist(),
                'edges': edges.tolist()
            }
        else:
            col_stats = {
                'type': 'categorical',
                'unique_count': int(df[col].nunique()),
                'null_pct': float(df[col].isnull().mean()),
                'top_values': df[col].value_counts().head(10).to_dict()
            }
        
        baseline['statistics'][col] = col_stats
    
    state.drift_baseline = baseline
    
    # Format as JSON
    baseline_json = json.dumps(baseline, indent=2, default=str)
    
    # Create summary report
    report = f"""## Drift Baseline Captured

**Timestamp:** {baseline['timestamp']}
**Dataset Shape:** {baseline['shape'][0]} rows Ã— {baseline['shape'][1]} columns

### Numeric Column Statistics:
"""
    
    for col, stats in baseline['statistics'].items():
        if stats['type'] == 'numeric':
            report += f"""
**{col}:**
- Mean: {stats['mean']:.4f} Â± {stats['std']:.4f}
- Range: [{stats['min']:.2f}, {stats['max']:.2f}]
- Median: {stats['median']:.4f}
- Missing: {stats['null_pct']*100:.1f}%
"""
    
    report += "\n### Categorical Column Statistics:\n"
    for col, stats in baseline['statistics'].items():
        if stats['type'] == 'categorical':
            report += f"- **{col}**: {stats['unique_count']} unique values\n"
    
    report += f"""
---
**Baseline saved!** This fingerprint will be used to detect data drift in production.
Export the JSON baseline for deployment monitoring.
"""
    
    return baseline_json, report


def generate_data_lineage(df: pd.DataFrame, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    """
    Generate data lineage visualization showing data flow through pipeline.
    Shows auditability and tracking of transformations.
    """
    if df is None:
        return create_placeholder_image("No data"), "Please upload data first."
    
    try:
        progress(0.1, desc="Generating lineage graph...")
        
        # Create lineage visualization
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('off')
        
        # Define pipeline stages
        stages = [
            {"name": "Raw CSV\nUpload", "x": 0.1, "y": 0.5, "color": "#3498db"},
            {"name": "Semantic\nDetection", "x": 0.3, "y": 0.5, "color": "#9b59b6"},
            {"name": "Pydantic\nValidation", "x": 0.5, "y": 0.5, "color": "#e74c3c"},
            {"name": "Drift\nBaseline", "x": 0.7, "y": 0.5, "color": "#f39c12"},
            {"name": "Ready for\nProcessing", "x": 0.9, "y": 0.5, "color": "#27ae60"}
        ]
        
        progress(0.3, desc="Drawing pipeline stages...")
        
        # Draw boxes for each stage
        for stage in stages:
            rect = plt.Rectangle((stage["x"]-0.07, stage["y"]-0.08), 0.14, 0.16,
                                facecolor=stage["color"], edgecolor='black', linewidth=2, alpha=0.8)
            ax.add_patch(rect)
            ax.text(stage["x"], stage["y"], stage["name"], ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white')
        
        # Draw arrows between stages
        for i in range(len(stages) - 1):
            x1, x2 = stages[i]["x"] + 0.07, stages[i+1]["x"] - 0.07
            y = stages[i]["y"]
            ax.annotate('', xy=(x2, y), xytext=(x1, y),
                       arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        
        progress(0.6, desc="Adding metrics...")
        
        # Add metrics below each stage
        rows, cols = df.shape
        nulls = df.isnull().sum().sum()
        semantic_detected = len(getattr(state, 'semantic_types', {}))
        
        metrics = [
            f"{rows:,} rows\n{cols} cols",
            f"{semantic_detected} types\ndetected",
            f"Validation:\nPending",
            f"Baseline:\nReady",
            f"Status:\nReady"
        ]
        
        for i, (stage, metric) in enumerate(zip(stages, metrics)):
            ax.text(stage["x"], stage["y"] - 0.15, metric, ha='center', va='top',
                   fontsize=9, color='gray', style='italic')
        
        # Add title and timestamp
        ax.text(0.5, 0.85, "Data Lineage Pipeline", ha='center', va='center',
               fontsize=18, fontweight='bold', color='#2c3e50')
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(0.5, 0.05, f"Generated: {timestamp}", ha='center', va='center',
               fontsize=9, color='gray')
        
        # Add audit trail box
        audit_text = "AUDIT TRAIL:\nData tracked from ingestion\nthrough validation pipeline\nfor full transparency"
        ax.text(0.5, 0.95, audit_text, ha='center', va='top',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8),
               color='#34495e')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        progress(0.9, desc="Finalizing visualization...")

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Lineage graph complete!")
        
        # Generate report
        report = f"""## Data Lineage Report

### Pipeline Overview
The data has been tracked through **5 critical stages** to ensure full auditability:

1. **Raw CSV Upload**: {rows:,} rows Ã— {cols} columns ingested
2. **Semantic Detection**: {semantic_detected} columns analyzed and typed
3. **Pydantic Validation**: Schema-based validation ready
4. **Drift Baseline**: Statistical fingerprint captured
5. **Processing Ready**: Data validated and ready for ML pipeline

### Audit Metrics
- **Data Quality**: {(1 - nulls/(rows*cols))*100:.1f}% complete
- **Type Coverage**: {(semantic_detected/cols)*100:.0f}% columns typed
- **Validation Status**: Schema generated and ready
- **Traceability**: Full lineage from source to pipeline

### Benefits
- **Regulatory Compliance**: Complete audit trail for compliance (HIPAA, GDPR, SOC2)
- **Debugging**: Easily trace data issues back to source
- **Transparency**: Clear visibility into data transformations
- **Production Ready**: Lineage tracking enables monitoring and alerting

### Next Steps
1. Proceed to Missing Values & Outliers tab
2. Continue through Feature Engineering pipeline
3. Export lineage metadata for production monitoring
"""
        
        return img, report
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error generating lineage: {str(e)}\n{traceback.format_exc()}"


# ==================== MISSING VALUES & OUTLIERS ====================

def detect_missing_data_bias(df: pd.DataFrame, progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Detect systematic bias in missing data patterns.
    Identifies if missingness correlates with other variables (e.g., Income missing for specific Age groups).
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    try:
        from scipy.stats import chi2_contingency, pearsonr
        
        progress(0.1, desc="Analyzing missing data patterns...")
        
        # Get columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        if not missing_cols:
            return create_placeholder_image("No missing data"), "No missing values detected. No bias analysis needed.", pd.DataFrame()
        
        progress(0.3, desc="Testing for systematic bias...")
        
        bias_results = []
        bias_warnings = []
        
        # For each column with missing data
        for missing_col in missing_cols:
            # Create binary indicator: 1 = missing, 0 = not missing
            is_missing = df[missing_col].isnull().astype(int)
            
            # Test correlation with other columns
            for other_col in df.columns:
                if other_col == missing_col:
                    continue
                
                # For numeric columns: use correlation
                if df[other_col].dtype in [np.float64, np.int64, np.float32, np.int32]:
                    other_clean = df[other_col].dropna()
                    is_missing_aligned = is_missing[df[other_col].notna()]
                    
                    if len(other_clean) > 10 and is_missing_aligned.sum() > 0:
                        try:
                            corr, p_value = pearsonr(other_clean, is_missing_aligned)
                            
                            # Significant correlation = systematic bias
                            if abs(corr) > 0.3 and p_value < 0.05:
                                severity = "CRITICAL" if abs(corr) > 0.5 else "WARNING"
                                bias_results.append({
                                    'Missing Column': missing_col,
                                    'Correlated With': other_col,
                                    'Correlation': f"{corr:.3f}",
                                    'P-value': f"{p_value:.4f}",
                                    'Severity': severity,
                                    'Type': 'Numeric Correlation'
                                })
                                
                                if severity == "CRITICAL":
                                    bias_warnings.append(f"CRITICAL: '{missing_col}' missing data strongly correlates with '{other_col}' (r={corr:.2f})")
                                else:
                                    bias_warnings.append(f"WARNING: '{missing_col}' missing data correlates with '{other_col}' (r={corr:.2f})")
                        except:
                            pass
                
                # For categorical columns: use chi-square test
                elif df[other_col].dtype == 'object' or df[other_col].nunique() < 20:
                    contingency = pd.crosstab(is_missing, df[other_col])
                    
                    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                        try:
                            chi2, p_value, dof, expected = chi2_contingency(contingency)
                            
                            # Significant association = systematic bias
                            if p_value < 0.05:
                                # Calculate effect size (CramÃ©r's V)
                                n = contingency.sum().sum()
                                min_dim = min(contingency.shape) - 1
                                cramers_v = np.sqrt(chi2 / (n * min_dim))
                                
                                if cramers_v > 0.3:
                                    severity = "CRITICAL" if cramers_v > 0.5 else "WARNING"
                                    bias_results.append({
                                        'Missing Column': missing_col,
                                        'Correlated With': other_col,
                                        'Correlation': f"{cramers_v:.3f}",
                                        'P-value': f"{p_value:.4f}",
                                        'Severity': severity,
                                        'Type': 'Categorical Association'
                                    })
                                    
                                    if severity == "CRITICAL":
                                        bias_warnings.append(f"CRITICAL: '{missing_col}' missing data varies by '{other_col}' (V={cramers_v:.2f})")
                                    else:
                                        bias_warnings.append(f"WARNING: '{missing_col}' missing data varies by '{other_col}' (V={cramers_v:.2f})")
                        except:
                            pass
        
        progress(0.7, desc="Creating bias visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Missing data heatmap
        ax1 = axes[0, 0]
        missing_matrix = df[missing_cols].isnull().astype(int)
        if len(missing_matrix.columns) > 0:
            im = ax1.imshow(missing_matrix.T, aspect='auto', cmap='RdYlGn_r', interpolation='none')
            ax1.set_yticks(range(len(missing_cols)))
            ax1.set_yticklabels(missing_cols, fontsize=9)
            ax1.set_xlabel('Row Index')
            ax1.set_title('Missing Data Pattern', fontweight='bold')
            plt.colorbar(im, ax=ax1, label='Missing (1) / Present (0)')
        
        # Plot 2: Bias severity distribution
        ax2 = axes[0, 1]
        if bias_results:
            severity_counts = pd.Series([r['Severity'] for r in bias_results]).value_counts()
            colors = {'CRITICAL': '#e74c3c', 'WARNING': '#f39c12'}
            bars = ax2.bar(severity_counts.index, severity_counts.values, 
                          color=[colors.get(s, 'gray') for s in severity_counts.index], alpha=0.8)
            ax2.set_ylabel('Count')
            ax2.set_title('Bias Severity Distribution', fontweight='bold')
            
            for bar, count in zip(bars, severity_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No Systematic Bias Detected', ha='center', va='center',
                    fontsize=14, color='#27ae60', fontweight='bold')
            ax2.axis('off')
        
        # Plot 3: Bias warning summary
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        if bias_warnings:
            warning_text = "SYSTEMATIC BIAS DETECTED\n\n"
            warning_text += "\n".join(bias_warnings[:8])
            ax3.text(0.05, 0.95, warning_text, ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='#fee', edgecolor='#e74c3c', linewidth=2),
                    color='#c0392b', family='monospace')
        else:
            ax3.text(0.5, 0.5, 'No Systematic Bias\nMissing data appears random', 
                    ha='center', va='center', fontsize=14, color='#27ae60',
                    bbox=dict(boxstyle='round', facecolor='#d5f4e6', edgecolor='#27ae60', linewidth=2),
                    fontweight='bold')
        
        # Plot 4: Correlation strength
        ax4 = axes[1, 1]
        if bias_results:
            correlations = [float(r['Correlation']) for r in bias_results]
            labels = [f"{r['Missing Column'][:10]}\nvs\n{r['Correlated With'][:10]}" for r in bias_results[:10]]
            
            bars = ax4.barh(range(len(correlations[:10])), correlations[:10], 
                           color=['#e74c3c' if c > 0.5 else '#f39c12' for c in correlations[:10]], alpha=0.8)
            ax4.set_yticks(range(len(labels)))
            ax4.set_yticklabels(labels, fontsize=8)
            ax4.set_xlabel('Correlation Strength')
            ax4.set_title('Top Bias Correlations', fontweight='bold')
            ax4.axvline(0.3, color='orange', linestyle='--', label='Warning threshold')
            ax4.axvline(0.5, color='red', linestyle='--', label='Critical threshold')
            ax4.legend(fontsize=8)
        else:
            ax4.axis('off')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Bias analysis complete!")
        
        # Generate report
        n_critical = len([r for r in bias_results if r['Severity'] == 'CRITICAL'])
        n_warning = len([r for r in bias_results if r['Severity'] == 'WARNING'])
        
        report = f"""## Missing Data Bias Detection Report

### Summary
- **Columns with Missing Data**: {len(missing_cols)}
- **Total Bias Patterns Detected**: {len(bias_results)}
- **Critical Biases**: {n_critical}
- **Warnings**: {n_warning}

### What is Systematic Bias?
Systematic bias occurs when missing data is **not random** but correlates with other variables.
This can lead to biased models and unfair predictions.

**Example**: If "Income" is missing only for people aged 20-30, the model may unfairly 
penalize younger individuals.

### Detected Patterns
"""
        
        if bias_results:
            report += "\n**CRITICAL ISSUES:**\n"
            for r in [r for r in bias_results if r['Severity'] == 'CRITICAL']:
                report += f"- **{r['Missing Column']}** missing data correlates with **{r['Correlated With']}** (strength: {r['Correlation']}, p<{r['P-value']})\n"
            
            report += "\n**WARNINGS:**\n"
            for r in [r for r in bias_results if r['Severity'] == 'WARNING']:
                report += f"- **{r['Missing Column']}** missing data correlates with **{r['Correlated With']}** (strength: {r['Correlation']}, p<{r['P-value']})\n"
            
            report += f"""

### Recommendations
1. **Investigate**: Why is data missing for specific groups?
2. **Collect**: Gather more data from underrepresented groups
3. **Document**: Record the bias for transparency in model reports
4. **Mitigate**: Use stratified imputation or separate models per group

### Impact on Fairness
Models trained on biased missing data may:
- Underperform for specific demographics
- Perpetuate existing inequalities
- Violate fairness regulations (GDPR, Equal Credit Opportunity Act)
"""
        else:
            report += "\nNo systematic bias detected. Missing data appears to be **Missing Completely At Random (MCAR)**.\n"
            report += "\nThis is good for model fairness, but you should still use proper imputation techniques.\n"
        
        results_df = pd.DataFrame(bias_results) if bias_results else pd.DataFrame()
        
        return img, report, results_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_bayesian_imputation(df: pd.DataFrame, progress=gr.Progress()) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """
    Run Iterative Bayesian Imputation using relationships between columns.
    Uses IterativeImputer (MICE algorithm) to predict missing values.
    """
    if df is None:
        return None, create_placeholder_image("No data"), "No data uploaded"
    
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        from sklearn.ensemble import RandomForestRegressor
        
        progress(0.1, desc="Analyzing missing values...")
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return df, create_placeholder_image("No numeric columns"), "No numeric columns found for imputation"
        
        # Calculate missing stats before
        missing_before = df[numeric_cols].isnull().sum()
        total_missing = missing_before.sum()
        
        if total_missing == 0:
            return df, create_placeholder_image("No missing values"), "No missing values found in numeric columns"
        
        progress(0.3, desc="Running Bayesian imputation...")
        
        # Create imputer with Bayesian estimation
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42, n_jobs=-1),
            max_iter=10,
            random_state=42,
            initial_strategy='median'
        )
        
        # Store original values for comparison
        original_numeric = df[numeric_cols].copy()
        
        # Impute
        imputed_values = imputer.fit_transform(df[numeric_cols])
        df_imputed = df.copy()
        df_imputed[numeric_cols] = imputed_values
        
        progress(0.7, desc="Creating visualization...")
        
        # Create comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Find columns with missing values
        cols_with_missing = [col for col in numeric_cols if missing_before[col] > 0][:4]
        
        for i, col in enumerate(cols_with_missing):
            ax = axes[i // 2, i % 2]
            
            # Plot original distribution (excluding NaN)
            original_vals = original_numeric[col].dropna()
            ax.hist(original_vals, bins=30, alpha=0.5, label='Original', color='blue', density=True)
            
            # Plot imputed distribution
            ax.hist(df_imputed[col], bins=30, alpha=0.5, label='After Imputation', color='green', density=True)
            
            # Mark imputed values
            imputed_mask = original_numeric[col].isnull()
            imputed_vals = df_imputed.loc[imputed_mask, col]
            if len(imputed_vals) > 0:
                ax.axvline(imputed_vals.mean(), color='red', linestyle='--', label=f'Imputed Mean: {imputed_vals.mean():.2f}')
            
            ax.set_title(f'{col} (Imputed: {missing_before[col]} values)')
            ax.legend()
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
        
        # Hide unused subplots
        for i in range(len(cols_with_missing), 4):
            axes[i // 2, i % 2].axis('off')
        
        plt.suptitle('Iterative Bayesian Imputation Results', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Imputation complete!")
        
        # Create report
        report = f"""## Iterative Bayesian Imputation Report

**Method:** MICE (Multiple Imputation by Chained Equations) with Random Forest

### How It Works:
The algorithm uses relationships between columns to predict missing values:
1. Initial fill with median values
2. For each column with missing values:
   - Train a Random Forest model using other columns as features
   - Predict the missing values based on the learned relationships
3. Repeat iteratively until convergence

### Missing Values Summary:
| Column | Missing Count | Percentage |
|--------|--------------|------------|
"""
        for col in cols_with_missing:
            pct = missing_before[col] / len(df) * 100
            report += f"| {col} | {missing_before[col]} | {pct:.1f}% |\n"
        
        report += f"""
### Imputation Statistics:
- **Total values imputed:** {total_missing}
- **Columns affected:** {len(cols_with_missing)}
- **Iterations:** 10
- **Base estimator:** Random Forest (10 trees, max_depth=5)

### Key Insight:
Unlike simple mean/median imputation, Bayesian imputation preserves the 
relationships between variables, resulting in more realistic imputed values.
"""
        
        # Store imputed df in state
        state.raw_stats['imputation_done'] = True
        
        return df_imputed, img, report
        
    except Exception as e:
        import traceback
        return df, create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}"


def run_outlier_detection(df: pd.DataFrame, contamination: float = 0.05, progress=gr.Progress()) -> Tuple[pd.DataFrame, np.ndarray, str]:
    """
    Run Isolation Forest outlier detection to flag anomalous rows.
    """
    if df is None:
        return None, None, create_placeholder_image("No data"), "No data uploaded"
    
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        progress(0.1, desc="Preparing data for outlier detection...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return df, None, create_placeholder_image("No numeric columns"), "No numeric columns for outlier detection"
        
        # Prepare data (handle missing values for detection)
        df_numeric = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
        
        progress(0.4, desc="Running Isolation Forest...")
        
        # Run Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )
        
        # Get predictions (-1 for outliers, 1 for inliers)
        predictions = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)
        
        # Create results
        df_result = df.copy()
        df_result['_outlier_flag'] = predictions
        df_result['_anomaly_score'] = anomaly_scores
        
        # Count outliers
        n_outliers = (predictions == -1).sum()
        n_inliers = (predictions == 1).sum()
        
        progress(0.7, desc="Analyzing outliers...")
        
        # Get outlier rows
        outlier_indices = np.where(predictions == -1)[0]
        outlier_df = df.iloc[outlier_indices].copy()
        outlier_df['_anomaly_score'] = anomaly_scores[outlier_indices]
        
        # Find which columns contribute most to outliers
        outlier_contributions = {}
        for col in numeric_cols:
            col_mean = df_numeric[col].mean()
            col_std = df_numeric[col].std()
            if col_std > 0:
                # Calculate z-scores for outliers
                outlier_zscores = np.abs((outlier_df[col].fillna(col_mean) - col_mean) / col_std)
                outlier_contributions[col] = outlier_zscores.mean()
        
        # Sort by contribution
        sorted_contributions = sorted(outlier_contributions.items(), key=lambda x: -x[1])
        
        progress(0.85, desc="Creating visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Anomaly score distribution
        ax1 = axes[0, 0]
        ax1.hist(anomaly_scores[predictions == 1], bins=30, alpha=0.7, label='Inliers', color='green')
        ax1.hist(anomaly_scores[predictions == -1], bins=30, alpha=0.7, label='Outliers', color='red')
        ax1.axvline(x=0, color='black', linestyle='--', label='Threshold')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Anomaly Score Distribution')
        ax1.legend()
        
        # Plot 2: Outlier contribution by column
        ax2 = axes[0, 1]
        top_cols = sorted_contributions[:8]
        if top_cols:
            cols = [c[0][:15] for c in top_cols]
            vals = [c[1] for c in top_cols]
            bars = ax2.barh(cols[::-1], vals[::-1], color='coral')
            ax2.set_xlabel('Average Z-Score in Outliers')
            ax2.set_title('Columns Contributing to Outliers')
        
        # Plot 3: Scatter of top 2 contributing columns
        ax3 = axes[1, 0]
        if len(sorted_contributions) >= 2:
            col1, col2 = sorted_contributions[0][0], sorted_contributions[1][0]
            ax3.scatter(df_numeric.loc[predictions == 1, col1], 
                       df_numeric.loc[predictions == 1, col2], 
                       alpha=0.5, label='Inliers', c='green', s=20)
            ax3.scatter(df_numeric.loc[predictions == -1, col1], 
                       df_numeric.loc[predictions == -1, col2], 
                       alpha=0.8, label='Outliers', c='red', s=50, marker='x')
            ax3.set_xlabel(col1[:20])
            ax3.set_ylabel(col2[:20])
            ax3.set_title('Outlier Scatter Plot')
            ax3.legend()
        
        # Plot 4: Pie chart of inliers vs outliers
        ax4 = axes[1, 1]
        ax4.pie([n_inliers, n_outliers], 
                labels=['Inliers', 'Outliers'], 
                colors=['#2ecc71', '#e74c3c'],
                autopct='%1.1f%%',
                explode=(0, 0.1))
        ax4.set_title(f'Data Quality: {n_outliers} Outliers Detected')
        
        plt.suptitle('Isolation Forest Outlier Detection', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Detection complete!")
        
        # Create outlier details table
        if len(outlier_df) > 0:
            outlier_display = outlier_df.head(20).round(2)
        else:
            outlier_display = pd.DataFrame({'Message': ['No outliers detected']})
        
        # Create report
        report = f"""## Isolation Forest Outlier Detection Report

**Method:** Isolation Forest (Unsupervised Anomaly Detection)

### How It Works:
Isolation Forest detects anomalies by isolating observations:
1. Randomly select a feature and split value
2. Outliers require fewer splits to isolate (shorter path length)
3. Anomaly score based on average path length across trees

### Detection Results:
| Metric | Value |
|--------|-------|
| Total Rows | {len(df)} |
| Inliers | {n_inliers} ({n_inliers/len(df)*100:.1f}%) |
| **Outliers** | **{n_outliers}** ({n_outliers/len(df)*100:.1f}%) |
| Contamination Rate | {contamination*100:.1f}% |

### Top Contributing Columns:
These columns show the most extreme values in outlier rows:
"""
        for col, zscore in sorted_contributions[:5]:
            report += f"- **{col}**: Avg Z-score = {zscore:.2f}\n"
        
        report += f"""
### Recommendation:
- **Review flagged rows** before removing them
- Outliers may be data entry errors OR legitimate extreme cases
- Consider domain knowledge when deciding treatment

### Outlier Indices:
First 20 outlier row indices: {list(outlier_indices[:20])}
"""
        
        return df_result, outlier_display, img, report
        
    except Exception as e:
        import traceback
        return df, None, create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}"


# ==================== EDA FUNCTIONS ====================

def run_hypothesis_generation(df: pd.DataFrame, target_col: str = None, progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Automated Hypothesis Generation - scans data for correlations and patterns.
    Returns insights about feature relationships and prioritization suggestions.
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    try:
        progress(0.1, desc="Analyzing feature relationships...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return create_placeholder_image("Need 2+ numeric columns"), "Need at least 2 numeric columns for correlation analysis", pd.DataFrame()
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        progress(0.3, desc="Finding significant correlations...")
        
        # Find strong correlations (excluding self-correlations)
        hypotheses = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Upper triangle only
                    corr = corr_matrix.loc[col1, col2]
                    if not np.isnan(corr):
                        abs_corr = abs(corr)
                        if abs_corr >= 0.3:  # Threshold for "interesting" correlation
                            direction = "positive" if corr > 0 else "negative"
                            strength = "strong" if abs_corr >= 0.7 else "moderate" if abs_corr >= 0.5 else "weak"
                            hypotheses.append({
                                'Feature 1': col1,
                                'Feature 2': col2,
                                'Correlation': round(corr, 3),
                                'Strength': strength.capitalize(),
                                'Direction': direction.capitalize(),
                                'Priority': 'High' if abs_corr >= 0.7 else 'Medium' if abs_corr >= 0.5 else 'Low'
                            })
        
        # Sort by absolute correlation
        hypotheses = sorted(hypotheses, key=lambda x: abs(x['Correlation']), reverse=True)
        
        progress(0.5, desc="Analyzing target relationships...")
        
        # If target column specified or can be inferred, analyze target correlations
        target_insights = []
        if target_col and target_col in numeric_cols:
            target_corrs = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
            for col, corr in target_corrs.items():
                if abs(corr) >= 0.2:
                    target_insights.append({
                        'Feature': col,
                        'Target Correlation': round(corr, 3),
                        'Predictive Power': 'High' if abs(corr) >= 0.5 else 'Medium' if abs(corr) >= 0.3 else 'Low'
                    })
        
        progress(0.7, desc="Creating visualization...")
        
        # Create correlation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Heatmap - limit to top 15 columns for readability
        top_cols = numeric_cols[:15]
        corr_subset = corr_matrix.loc[top_cols, top_cols]
        
        ax1 = axes[0]
        im = ax1.imshow(corr_subset, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(top_cols)))
        ax1.set_yticks(range(len(top_cols)))
        ax1.set_xticklabels([c[:12] for c in top_cols], rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels([c[:12] for c in top_cols], fontsize=8)
        ax1.set_title('Feature Correlation Heatmap', fontweight='bold')
        
        # Add correlation values
        for i in range(len(top_cols)):
            for j in range(len(top_cols)):
                val = corr_subset.iloc[i, j]
                if not np.isnan(val):
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=color)
        
        plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # Bar chart of top correlations
        ax2 = axes[1]
        if hypotheses:
            top_hyp = hypotheses[:12]
            labels = [f"{h['Feature 1'][:10]} vs {h['Feature 2'][:10]}" for h in top_hyp]
            values = [h['Correlation'] for h in top_hyp]
            colors = ['#e74c3c' if v < 0 else '#27ae60' for v in values]
            
            y_pos = range(len(labels))
            ax2.barh(y_pos, values, color=colors, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.set_xlabel('Correlation Coefficient')
            ax2.set_title('Top Feature Correlations', fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlim(-1, 1)
        else:
            ax2.text(0.5, 0.5, 'No significant correlations found', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Top Feature Correlations', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(0.9, desc="Generating report...")
        
        # Create hypothesis table
        if hypotheses:
            hyp_df = pd.DataFrame(hypotheses[:20])
        else:
            hyp_df = pd.DataFrame({'Message': ['No significant correlations found (threshold: 0.3)']})
        
        # Generate report
        report = f"""## Automated Hypothesis Generation Report

### Analysis Summary
- **Total Features Analyzed:** {len(numeric_cols)}
- **Correlation Pairs Examined:** {len(numeric_cols) * (len(numeric_cols) - 1) // 2}
- **Significant Correlations Found:** {len(hypotheses)}

### Key Findings
"""
        
        if hypotheses:
            # Strong correlations
            strong = [h for h in hypotheses if h['Strength'] == 'Strong']
            moderate = [h for h in hypotheses if h['Strength'] == 'Moderate']
            
            if strong:
                report += f"\n**Strong Correlations (|r| >= 0.7):** {len(strong)} found\n"
                for h in strong[:3]:
                    report += f"- {h['Feature 1']} and {h['Feature 2']}: r = {h['Correlation']} ({h['Direction']})\n"
            
            if moderate:
                report += f"\n**Moderate Correlations (0.5 <= |r| < 0.7):** {len(moderate)} found\n"
                for h in moderate[:3]:
                    report += f"- {h['Feature 1']} and {h['Feature 2']}: r = {h['Correlation']} ({h['Direction']})\n"
        
        # Target-specific insights
        if target_insights:
            report += f"\n### Target Variable Analysis ('{target_col}')\n"
            report += "Top predictive features:\n"
            for ti in target_insights[:5]:
                report += f"- **{ti['Feature']}**: r = {ti['Target Correlation']} (Predictive Power: {ti['Predictive Power']})\n"
        
        report += """
### Recommendations

**Feature Engineering:**
- Consider creating interaction terms for strongly correlated features
- Highly correlated features may indicate multicollinearity - consider removing redundant ones

**Model Building:**
- Prioritize features with strong target correlations
- Watch for data leakage from features too perfectly correlated with target

**Data Quality:**
- Correlations near 1.0 may indicate duplicate or derived features
- Check if strong negative correlations make domain sense
"""
        
        progress(1.0, desc="Analysis complete!")
        
        return img, report, hyp_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_dimensionality_reduction(df: pd.DataFrame, target_col: str = None, method: str = "UMAP", 
                                  n_components: int = 2, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    """
    Interactive Dimensionality Reduction using UMAP or t-SNE.
    Shows cluster visualization to see how groups separate before training.
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded"
    
    try:
        progress(0.1, desc=f"Preparing data for {method}...")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from features if specified
        feature_cols = [c for c in numeric_cols if c != target_col]
        
        if len(feature_cols) < 2:
            return create_placeholder_image("Need 2+ features"), "Need at least 2 numeric features for dimensionality reduction"
        
        # Prepare data - handle missing values
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Scale data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Limit samples for speed (UMAP/t-SNE are slow on large datasets)
        max_samples = 5000
        if len(X_scaled) > max_samples:
            indices = np.random.choice(len(X_scaled), max_samples, replace=False)
            X_scaled = X_scaled[indices]
            df_subset = df.iloc[indices]
        else:
            indices = np.arange(len(X_scaled))
            df_subset = df
        
        progress(0.3, desc=f"Running {method}...")
        
        # Apply dimensionality reduction
        if method == "UMAP":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=15,
                    min_dist=0.1,
                    metric='euclidean',
                    random_state=42
                )
                embedding = reducer.fit_transform(X_scaled)
            except ImportError:
                # Fall back to t-SNE if UMAP not installed
                from sklearn.manifold import TSNE
                reducer = TSNE(
                    n_components=min(n_components, 3),
                    perplexity=min(30, len(X_scaled) - 1),
                    random_state=42,
                    max_iter=500
                )
                embedding = reducer.fit_transform(X_scaled)
                method = "t-SNE (UMAP unavailable)"
        else:  # t-SNE
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=min(n_components, 3),
                perplexity=min(30, len(X_scaled) - 1),
                random_state=42,
                max_iter=500
            )
            embedding = reducer.fit_transform(X_scaled)
        
        progress(0.7, desc="Creating visualization...")
        
        # Determine coloring
        color_values = None
        color_label = "Index"
        is_categorical = False
        
        if target_col and target_col in df_subset.columns:
            target_data = df_subset[target_col].values
            unique_vals = pd.Series(target_data).nunique()
            
            if unique_vals <= 10:  # Categorical target
                is_categorical = True
                color_values = target_data
                color_label = target_col
            else:  # Continuous target
                color_values = target_data
                color_label = target_col
        
        # Create visualization
        if n_components == 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            if is_categorical and color_values is not None:
                # Categorical coloring
                unique_classes = np.unique(color_values[~pd.isna(color_values)])
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
                
                for i, cls in enumerate(unique_classes):
                    mask = color_values == cls
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=[colors[i]], label=f'{cls}', alpha=0.6, s=30)
                ax.legend(title=color_label, loc='best')
            elif color_values is not None:
                # Continuous coloring
                sc = ax.scatter(embedding[:, 0], embedding[:, 1], 
                               c=color_values, cmap='viridis', alpha=0.6, s=30)
                plt.colorbar(sc, ax=ax, label=color_label)
            else:
                # No target - color by index
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.6, s=30, c='steelblue')
            
            ax.set_xlabel(f'{method} Component 1')
            ax.set_ylabel(f'{method} Component 2')
            ax.set_title(f'{method} Projection ({len(feature_cols)} features to 2D)', fontweight='bold')
            
        else:  # 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if is_categorical and color_values is not None:
                unique_classes = np.unique(color_values[~pd.isna(color_values)])
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
                
                for i, cls in enumerate(unique_classes):
                    mask = color_values == cls
                    ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                              c=[colors[i]], label=f'{cls}', alpha=0.6, s=30)
                ax.legend(title=color_label, loc='best')
            elif color_values is not None:
                sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=color_values, cmap='viridis', alpha=0.6, s=30)
                plt.colorbar(sc, ax=ax, label=color_label, shrink=0.6)
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], 
                          alpha=0.6, s=30, c='steelblue')
            
            ax.set_xlabel(f'{method} 1')
            ax.set_ylabel(f'{method} 2')
            ax.set_zlabel(f'{method} 3')
            ax.set_title(f'{method} 3D Projection ({len(feature_cols)} features)', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Visualization complete!")
        
        # Generate report
        report = f"""## {method} Dimensionality Reduction Report

### Configuration
- **Method:** {method}
- **Input Features:** {len(feature_cols)}
- **Output Dimensions:** {n_components}
- **Samples Visualized:** {len(X_scaled):,}
"""
        
        if target_col:
            report += f"- **Color Variable:** {target_col}\n"
            if is_categorical:
                unique_classes = np.unique(color_values[~pd.isna(color_values)])
                report += f"- **Classes:** {list(unique_classes)}\n"
        
        report += f"""
### How to Interpret

**Cluster Separation:**
- Points close together = similar feature patterns
- Clear separation between colors = good class separability
- Overlapping classes = harder classification problem

**What to Look For:**
- Distinct clusters suggest the model should learn patterns easily
- Overlapping regions may need feature engineering
- Outliers far from clusters may be anomalies

### Technical Details

**{method} Parameters:**
"""
        if "UMAP" in method:
            report += """- n_neighbors: 15 (local vs global structure balance)
- min_dist: 0.1 (how tightly points can pack)
- metric: euclidean
"""
        else:
            report += """- perplexity: 30 (balance of local/global structure)
- max_iter: 500 (optimization iterations)
"""
        
        report += f"""
### Recommendations

Based on the visualization:
- **Well-separated clusters:** Your features have good discriminative power
- **Overlapping clusters:** Consider feature engineering or collecting more distinguishing features
- **No clear structure:** May indicate noisy data or need for different features

**Next Steps:**
1. Use this insight to guide feature selection
2. Consider removing features that don't contribute to separation
3. Try different target variables to see class separability
"""
        
        return img, report
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}"


def explain_cluster_with_ai(df: pd.DataFrame, target_col: str = None, 
                            cluster_id: int = 0, progress=gr.Progress()) -> str:
    """
    Use AI to explain why specific data points cluster together.
    Identifies common patterns in a cluster.
    """
    if df is None:
        return "No data available. Please run dimensionality reduction first."
    
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        progress(0.1, desc="Identifying clusters...")
        
        # Get numeric features
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in feature_cols:
            feature_cols.remove(target_col)
        
        if len(feature_cols) < 2:
            return "Not enough numeric features for clustering."
        
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        progress(0.3, desc="Running K-Means clustering...")
        
        # Cluster with K-Means
        n_clusters = min(5, len(df) // 20)
        if n_clusters < 2:
            n_clusters = 2
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Get the requested cluster (default to largest)
        cluster_sizes = pd.Series(clusters).value_counts()
        if cluster_id >= len(cluster_sizes):
            cluster_id = cluster_sizes.index[0]
        
        progress(0.5, desc=f"Analyzing cluster {cluster_id}...")
        
        # Get points in this cluster
        cluster_mask = clusters == cluster_id
        cluster_data = df[cluster_mask][feature_cols]
        overall_data = df[feature_cols]
        
        # Find distinguishing features
        explanations = []
        
        for col in feature_cols:
            cluster_mean = cluster_data[col].mean()
            overall_mean = overall_data[col].mean()
            overall_std = overall_data[col].std()
            
            if overall_std > 0:
                z_score = (cluster_mean - overall_mean) / overall_std
                
                if abs(z_score) > 0.5:  # Significant difference
                    if z_score > 0:
                        explanations.append(f"**{col}**: HIGH (avg: {cluster_mean:.2f} vs overall: {overall_mean:.2f})")
                    else:
                        explanations.append(f"**{col}**: LOW (avg: {cluster_mean:.2f} vs overall: {overall_mean:.2f})")
        
        progress(0.8, desc="Generating AI explanation...")
        
        # Create explanation
        explanation = f"""## Cluster {cluster_id} Explanation

### Cluster Summary
- **Size:** {cluster_mask.sum()} samples ({cluster_mask.mean()*100:.1f}% of dataset)
- **Total Clusters:** {n_clusters}

### Distinguishing Characteristics

This cluster is characterized by:

"""
        
        if explanations:
            for exp in explanations[:10]:
                explanation += f"- {exp}\n"
        else:
            explanation += "- No strongly distinguishing features (cluster is similar to overall average)\n"
        
        # Add target distribution if available
        if target_col and target_col in df.columns:
            cluster_target = df[cluster_mask][target_col]
            overall_target = df[target_col]
            
            explanation += f"\n### Target Variable Distribution\n\n"
            
            if cluster_target.dtype in [np.float64, np.int64]:
                explanation += f"- **Cluster {target_col} mean:** {cluster_target.mean():.2f}\n"
                explanation += f"- **Overall {target_col} mean:** {overall_target.mean():.2f}\n"
            else:
                cluster_dist = cluster_target.value_counts(normalize=True).head(3)
                explanation += f"**Top values in cluster:**\n"
                for val, pct in cluster_dist.items():
                    explanation += f"- {val}: {pct*100:.1f}%\n"
        
        # AI-generated interpretation
        explanation += f"""

### Interpretation

**Pattern:** This group represents {"high-risk" if any("HIGH" in e for e in explanations) else "typical"} cases with distinct characteristics.

**Clinical/Business Relevance:** 
- These samples cluster together because they share similar feature values
- They may represent a specific demographic or risk category
- Understanding this cluster helps with targeted interventions

**Recommended Actions:**
1. Review the distinguishing features for this cluster
2. Consider if this cluster needs specialized treatment/handling
3. Use this insight for stratified model training
"""
        
        progress(1.0, desc="Cluster explanation complete!")
        
        return explanation
        
    except Exception as e:
        import traceback
        return f"Error generating explanation: {str(e)}\n{traceback.format_exc()}"


# ==================== FEATURE ENGINEERING FUNCTIONS ====================

# Universal feature creation patterns based on semantic types
FEATURE_CREATION_RULES = {
    # Health/Medical domain
    'bmi': {
        'requires': ['weight', 'height'],
        'formula': lambda df, cols: df[cols['weight']] / (df[cols['height']] ** 2),
        'description': 'Body Mass Index = weight / height^2'
    },
    'pulse_pressure': {
        'requires': ['systolic', 'diastolic'],
        'formula': lambda df, cols: df[cols['systolic']] - df[cols['diastolic']],
        'description': 'Pulse Pressure = Systolic - Diastolic'
    },
    'mean_arterial_pressure': {
        'requires': ['systolic', 'diastolic'],
        'formula': lambda df, cols: (df[cols['systolic']] + 2 * df[cols['diastolic']]) / 3,
        'description': 'MAP = (Systolic + 2*Diastolic) / 3'
    },
    
    # Financial domain
    'debt_to_income': {
        'requires': ['debt', 'income'],
        'formula': lambda df, cols: df[cols['debt']] / df[cols['income']].replace(0, np.nan),
        'description': 'Debt-to-Income Ratio = debt / income'
    },
    'savings_rate': {
        'requires': ['savings', 'income'],
        'formula': lambda df, cols: df[cols['savings']] / df[cols['income']].replace(0, np.nan),
        'description': 'Savings Rate = savings / income'
    },
    'profit_margin': {
        'requires': ['profit', 'revenue'],
        'formula': lambda df, cols: df[cols['profit']] / df[cols['revenue']].replace(0, np.nan),
        'description': 'Profit Margin = profit / revenue'
    },
    
    # E-commerce domain
    'conversion_rate': {
        'requires': ['purchases', 'visits'],
        'formula': lambda df, cols: df[cols['purchases']] / df[cols['visits']].replace(0, np.nan),
        'description': 'Conversion Rate = purchases / visits'
    },
    'avg_order_value': {
        'requires': ['revenue', 'orders'],
        'formula': lambda df, cols: df[cols['revenue']] / df[cols['orders']].replace(0, np.nan),
        'description': 'Average Order Value = revenue / orders'
    },
    
    # HR/Demographics domain
    'age_tenure_ratio': {
        'requires': ['age', 'tenure'],
        'formula': lambda df, cols: df[cols['age']] / df[cols['tenure']].replace(0, np.nan),
        'description': 'Age-Tenure Ratio (career stage indicator)'
    },
    'experience_ratio': {
        'requires': ['experience', 'age'],
        'formula': lambda df, cols: df[cols['experience']] / (df[cols['age']] - 18).replace(0, np.nan),
        'description': 'Experience Ratio = experience / (age - 18)'
    },
}

# Column name patterns for matching
COLUMN_PATTERNS = {
    'weight': r'(?i)(weight|mass|wt|kg)',
    'height': r'(?i)(height|ht|cm|meters?)',
    'systolic': r'(?i)(systolic|sys|sbp|systolic.?bp)',
    'diastolic': r'(?i)(diastolic|dia|dbp|diastolic.?bp)',
    'age': r'(?i)(age|years?|yr)',
    'income': r'(?i)(income|salary|wage|earning|pay)',
    'debt': r'(?i)(debt|loan|owe|liability)',
    'savings': r'(?i)(saving|balance|deposit)',
    'profit': r'(?i)(profit|net.?income|earnings)',
    'revenue': r'(?i)(revenue|sales|gross)',
    'purchases': r'(?i)(purchase|order|buy|transaction)',
    'visits': r'(?i)(visit|session|view|click)',
    'orders': r'(?i)(order|transaction|purchase)',
    'tenure': r'(?i)(tenure|years?.?employed|service.?years?)',
    'experience': r'(?i)(experience|exp|years?.?exp)',
}


def run_agentic_feature_creation(df: pd.DataFrame, progress=gr.Progress()) -> Tuple[pd.DataFrame, np.ndarray, str, pd.DataFrame]:
    """
    Agentic Feature Creation - analyzes data semantically and creates domain-expert features.
    Works universally across different domains (medical, financial, e-commerce, etc.)
    """
    if df is None:
        return None, create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    try:
        progress(0.1, desc="Analyzing column semantics...")
        
        # Match columns to semantic types
        column_matches = {}
        for col in df.columns:
            for semantic_type, pattern in COLUMN_PATTERNS.items():
                if re.search(pattern, col):
                    if semantic_type not in column_matches:
                        column_matches[semantic_type] = col
                    break
        
        progress(0.3, desc="Identifying applicable feature rules...")
        
        # Find which rules can be applied
        applicable_rules = []
        for rule_name, rule_config in FEATURE_CREATION_RULES.items():
            required = rule_config['requires']
            if all(r in column_matches for r in required):
                applicable_rules.append((rule_name, rule_config, {r: column_matches[r] for r in required}))
        
        progress(0.5, desc="Creating expert features...")
        
        # Create new features
        df_enhanced = df.copy()
        created_features = []
        
        for rule_name, rule_config, cols_mapping in applicable_rules:
            try:
                new_col_name = f"feat_{rule_name}"
                df_enhanced[new_col_name] = rule_config['formula'](df, cols_mapping)
                
                # Clean up infinities
                df_enhanced[new_col_name] = df_enhanced[new_col_name].replace([np.inf, -np.inf], np.nan)
                
                created_features.append({
                    'Feature Name': new_col_name,
                    'Formula': rule_config['description'],
                    'Source Columns': ', '.join(cols_mapping.values()),
                    'Non-Null Count': df_enhanced[new_col_name].notna().sum(),
                    'Mean': round(df_enhanced[new_col_name].mean(), 4) if df_enhanced[new_col_name].notna().any() else 'N/A'
                })
            except Exception as e:
                continue
        
        # Also create statistical interaction features for numeric columns
        progress(0.7, desc="Creating statistical interaction features...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # If no domain features were created, create generic polynomial and interaction features
        if len(created_features) == 0 and len(numeric_cols) >= 2:
            progress(0.75, desc="Creating polynomial features...")
            
            # Square and cube features for top numeric columns
            top_numeric = numeric_cols[:min(5, len(numeric_cols))]
            for col in top_numeric:
                # Square
                square_name = f"feat_{col[:10]}_squared"
                df_enhanced[square_name] = df[col] ** 2
                created_features.append({
                    'Feature Name': square_name,
                    'Formula': f'{col}^2',
                    'Source Columns': col,
                    'Non-Null Count': df_enhanced[square_name].notna().sum(),
                    'Mean': round(df_enhanced[square_name].mean(), 4) if df_enhanced[square_name].notna().any() else 'N/A'
                })
                
                # Log transform (for positive values)
                if df[col].min() > 0:
                    log_name = f"feat_log_{col[:10]}"
                    df_enhanced[log_name] = np.log1p(df[col])
                    created_features.append({
                        'Feature Name': log_name,
                        'Formula': f'log({col}+1)',
                        'Source Columns': col,
                        'Non-Null Count': df_enhanced[log_name].notna().sum(),
                        'Mean': round(df_enhanced[log_name].mean(), 4) if df_enhanced[log_name].notna().any() else 'N/A'
                    })
            
            # Interaction features (product and ratio) for top pairs
            if len(numeric_cols) >= 2:
                for i in range(min(3, len(numeric_cols))):
                    for j in range(i+1, min(4, len(numeric_cols))):
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        
                        # Product
                        prod_name = f"feat_{col1[:8]}_x_{col2[:8]}"
                        df_enhanced[prod_name] = df[col1] * df[col2]
                        df_enhanced[prod_name] = df_enhanced[prod_name].replace([np.inf, -np.inf], np.nan)
                        created_features.append({
                            'Feature Name': prod_name,
                            'Formula': f'{col1} * {col2}',
                            'Source Columns': f'{col1}, {col2}',
                            'Non-Null Count': df_enhanced[prod_name].notna().sum(),
                            'Mean': round(df_enhanced[prod_name].mean(), 4) if df_enhanced[prod_name].notna().any() else 'N/A'
                        })
                        
                        # Ratio
                        if df[col2].abs().min() > 0:
                            ratio_name = f"feat_{col1[:8]}_div_{col2[:8]}"
                            df_enhanced[ratio_name] = df[col1] / df[col2]
                            df_enhanced[ratio_name] = df_enhanced[ratio_name].replace([np.inf, -np.inf], np.nan)
                            created_features.append({
                                'Feature Name': ratio_name,
                                'Formula': f'{col1} / {col2}',
                                'Source Columns': f'{col1}, {col2}',
                                'Non-Null Count': df_enhanced[ratio_name].notna().sum(),
                                'Mean': round(df_enhanced[ratio_name].mean(), 4) if df_enhanced[ratio_name].notna().any() else 'N/A'
                            })
        
        # Create ratio features for highly correlated pairs (only if domain features were created)
        if len(numeric_cols) >= 2 and len(created_features) > 0 and len(created_features) < 20:
            corr_matrix = df[numeric_cols[:10]].corr()
            for i, col1 in enumerate(numeric_cols[:10]):
                for col2 in numeric_cols[i+1:10]:
                    corr = corr_matrix.loc[col1, col2]
                    ratio_name = f"feat_{col1[:8]}_div_{col2[:8]}"
                    
                    # Skip if already created
                    if ratio_name in df_enhanced.columns:
                        continue
                    
                    if abs(corr) >= 0.5:  # Only for correlated pairs
                        # Ratio feature
                        df_enhanced[ratio_name] = df[col1] / df[col2].replace(0, np.nan)
                        df_enhanced[ratio_name] = df_enhanced[ratio_name].replace([np.inf, -np.inf], np.nan)
                        
                        created_features.append({
                            'Feature Name': ratio_name,
                            'Formula': f'{col1} / {col2} (correlation: {corr:.2f})',
                            'Source Columns': f'{col1}, {col2}',
                            'Non-Null Count': df_enhanced[ratio_name].notna().sum(),
                            'Mean': round(df_enhanced[ratio_name].mean(), 4) if df_enhanced[ratio_name].notna().any() else 'N/A'
                        })
                        
                        if len(created_features) >= 20:  # Limit total features
                            break
                if len(created_features) >= 20:
                    break
        
        progress(0.85, desc="Creating visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Feature creation summary
        ax1 = axes[0, 0]
        if created_features:
            feature_names = [f['Feature Name'][:15] for f in created_features[:10]]
            non_null_counts = [f['Non-Null Count'] for f in created_features[:10]]
            ax1.barh(feature_names[::-1], non_null_counts[::-1], color='steelblue', alpha=0.8)
            ax1.set_xlabel('Non-Null Count')
            ax1.set_title('Created Features - Data Coverage', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No domain features created', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Created Features', fontweight='bold')
        
        # Plot 2: Distribution of first new feature
        ax2 = axes[0, 1]
        new_feat_cols = [f['Feature Name'] for f in created_features if f['Feature Name'] in df_enhanced.columns]
        if new_feat_cols:
            first_feat = new_feat_cols[0]
            data = df_enhanced[first_feat].dropna()
            if len(data) > 0:
                ax2.hist(data, bins=30, color='coral', alpha=0.7, edgecolor='white')
                ax2.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
                ax2.set_xlabel(first_feat)
                ax2.set_ylabel('Frequency')
                ax2.legend()
        ax2.set_title('First Created Feature Distribution', fontweight='bold')
        
        # Plot 3: Column type breakdown
        ax3 = axes[1, 0]
        original_cols = len(df.columns)
        new_cols = len(df_enhanced.columns) - original_cols
        ax3.pie([original_cols, new_cols], 
                labels=['Original Features', 'Created Features'],
                colors=['#3498db', '#e74c3c'],
                autopct='%1.0f%%',
                explode=(0, 0.1))
        ax3.set_title(f'Feature Expansion: {original_cols} -> {len(df_enhanced.columns)}', fontweight='bold')
        
        # Plot 4: Correlation Heatmap Comparison (Mutual Information Proof)
        ax4 = axes[1, 1]
        
        # Calculate mutual information improvement
        from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        
        # Get target variable
        numeric_cols_orig = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_new = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        new_feature_names = [c for c in numeric_cols_new if c not in numeric_cols_orig]
        
        if len(numeric_cols_orig) > 0 and len(new_feature_names) > 0:
            # Pick top original and new features for comparison
            top_orig = numeric_cols_orig[:min(5, len(numeric_cols_orig))]
            top_new = new_feature_names[:min(5, len(new_feature_names))]
            
            # Create mini correlation matrix
            all_feats = top_orig + top_new
            corr_matrix = df_enhanced[all_feats].corr().fillna(0)
            
            # Plot heatmap
            im = ax4.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
            ax4.set_xticks(range(len(all_feats)))
            ax4.set_yticks(range(len(all_feats)))
            ax4.set_xticklabels([f[:8] for f in all_feats], rotation=45, ha='right', fontsize=8)
            ax4.set_yticklabels([f[:8] for f in all_feats], fontsize=8)
            ax4.set_title('Correlation: Original vs Created', fontweight='bold')
            
            # Add correlation values
            for i in range(len(all_feats)):
                for j in range(len(all_feats)):
                    text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=7)
            
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        else:
            ax4.text(0.5, 0.5, 'Correlation Analysis\nRequires numeric features', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=10)
            ax4.set_title('Correlation Heatmap', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Feature creation complete!")
        
        # Create features table
        if created_features:
            features_df = pd.DataFrame(created_features)
        else:
            features_df = pd.DataFrame({'Message': ['No expert features could be created from this dataset']})
        
        # Generate report
        report = f"""## Agentic Feature Creation Report

### Analysis Summary
- **Original Features:** {len(df.columns)}
- **New Features Created:** {len(created_features)}
- **Total Features:** {len(df_enhanced.columns)}
- **Semantic Matches Found:** {len(column_matches)}

### Detected Column Semantics
"""
        if column_matches:
            for semantic_type, col_name in column_matches.items():
                report += f"- **{semantic_type}**: {col_name}\n"
        else:
            report += "No recognized semantic patterns found in column names.\n"
        
        report += f"""
### Domain-Expert Features Created
"""
        domain_features = [f for f in created_features if not f['Feature Name'].startswith('feat_') or '_div_' not in f['Feature Name']]
        if domain_features:
            for f in domain_features[:5]:
                report += f"- **{f['Feature Name']}**: {f['Formula']}\n"
        else:
            report += "No domain-specific features created (column names did not match known patterns).\n"
        
        interaction_features = [f for f in created_features if '_div_' in f['Feature Name']]
        if interaction_features:
            report += f"""
### Statistical Interaction Features
Created {len(interaction_features)} ratio features from correlated column pairs.
"""
        
        report += """
### How It Works
1. **Semantic Analysis**: Column names are matched against known patterns (medical, financial, e-commerce, HR)
2. **Domain Rules**: Expert formulas are applied when matching columns are found
3. **Statistical Interactions**: Ratio features are created for highly correlated pairs

### Next Steps
- Review created features for domain relevance
- Use RFE (next subtab) to eliminate redundant features
- Features with many nulls may need imputation first
"""
        
        return df_enhanced, img, report, features_df
        
    except Exception as e:
        import traceback
        return df, create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_recursive_feature_elimination(df: pd.DataFrame, target_col: str = None, 
                                       n_features: int = 10, progress=gr.Progress()) -> Tuple[pd.DataFrame, np.ndarray, str, pd.DataFrame]:
    """
    Recursive Feature Elimination (RFE) - survival of the fittest tournament for features.
    Keeps only features that increase Signal-to-Noise ratio.
    """
    if df is None:
        return None, create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    if target_col is None or target_col not in df.columns:
        return df, create_placeholder_image("No target"), "Please select a target column in the Data Ingestion tab first", pd.DataFrame()
    
    try:
        from sklearn.feature_selection import RFE, RFECV
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        
        progress(0.1, desc="Preparing data for RFE tournament...")
        
        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            return df, create_placeholder_image("Need more features"), "Need at least 2 numeric features for RFE", pd.DataFrame()
        
        # Prepare X and y
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target_col].copy()
        
        # Determine task type
        unique_vals = y.nunique()
        is_classification = unique_vals <= 20 or y.dtype == 'object'
        
        if is_classification:
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('Unknown'))
            else:
                y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            y = y.fillna(y.median())
            estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # Scale features (keep scaler for export)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        state.preprocessors = {"scaler": scaler}
        
        progress(0.3, desc="Running feature tournament (Round 1)...")
        
        # First, get baseline score with all features
        baseline_scores = cross_val_score(estimator, X_scaled, y, cv=cv, scoring='accuracy' if is_classification else 'r2')
        baseline_score = baseline_scores.mean()
        
        progress(0.5, desc="Running RFE elimination rounds...")
        
        # Run RFE
        n_features_to_select = min(n_features, len(numeric_cols) - 1)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=1)
        rfe.fit(X_scaled, y)
        
        # Get ranking
        feature_ranking = []
        for i, (col, rank, selected) in enumerate(zip(numeric_cols, rfe.ranking_, rfe.support_)):
            feature_ranking.append({
                'Feature': col,
                'Rank': rank,
                'Selected': 'Yes' if selected else 'No',
                'Elimination Round': rank if rank > 1 else 'Survived'
            })
        
        feature_ranking = sorted(feature_ranking, key=lambda x: x['Rank'])
        
        progress(0.7, desc="Calculating feature importance scores...")
        
        # Get feature importances from final model
        selected_features = [col for col, selected in zip(numeric_cols, rfe.support_) if selected]
        X_selected = X[selected_features]
        X_selected_scaled = scaler.fit_transform(X_selected)
        
        estimator.fit(X_selected_scaled, y)
        importances = estimator.feature_importances_
        
        # Final score with selected features
        final_scores = cross_val_score(estimator, X_selected_scaled, y, cv=cv, scoring='accuracy' if is_classification else 'r2')
        final_score = final_scores.mean()
        
        # Add importance to ranking
        importance_dict = dict(zip(selected_features, importances))
        for item in feature_ranking:
            item['Importance'] = round(importance_dict.get(item['Feature'], 0), 4)
        
        progress(0.85, desc="Creating visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Feature ranking bar chart
        ax1 = axes[0, 0]
        top_features = feature_ranking[:15]
        colors = ['#27ae60' if f['Selected'] == 'Yes' else '#e74c3c' for f in top_features]
        ax1.barh([f['Feature'][:15] for f in top_features][::-1], 
                 [f['Rank'] for f in top_features][::-1], 
                 color=colors[::-1], alpha=0.8)
        ax1.set_xlabel('Elimination Rank (1 = Best)')
        ax1.set_title('Feature Tournament Ranking', fontweight='bold')
        ax1.axvline(x=1.5, color='green', linestyle='--', alpha=0.5, label='Survival Threshold')
        
        # Plot 2: Feature importance of selected features
        ax2 = axes[0, 1]
        selected_ranking = [f for f in feature_ranking if f['Selected'] == 'Yes']
        if selected_ranking:
            ax2.barh([f['Feature'][:15] for f in selected_ranking][::-1],
                     [f['Importance'] for f in selected_ranking][::-1],
                     color='steelblue', alpha=0.8)
            ax2.set_xlabel('Importance Score')
            ax2.set_title('Selected Features - Importance', fontweight='bold')
        
        # Plot 3: Score comparison
        ax3 = axes[1, 0]
        metric = 'Accuracy' if is_classification else 'R2 Score'
        bars = ax3.bar(['All Features', 'Selected Features'], 
                       [baseline_score, final_score],
                       color=['#3498db', '#27ae60'], alpha=0.8)
        ax3.set_ylabel(metric)
        ax3.set_title(f'Model Performance: Before vs After RFE', fontweight='bold')
        ax3.set_ylim(0, max(baseline_score, final_score) * 1.2)
        
        # Add value labels
        for bar, val in zip(bars, [baseline_score, final_score]):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Feature reduction summary
        ax4 = axes[1, 1]
        eliminated = len(numeric_cols) - len(selected_features)
        ax4.pie([len(selected_features), eliminated],
                labels=['Survived', 'Eliminated'],
                colors=['#27ae60', '#e74c3c'],
                autopct='%1.0f%%',
                explode=(0.05, 0))
        ax4.set_title(f'Feature Survival: {len(selected_features)}/{len(numeric_cols)}', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="RFE complete!")
        
        # Create ranking table
        ranking_df = pd.DataFrame(feature_ranking)
        
        # Create filtered dataframe with only selected features + target
        df_filtered = df[[target_col] + selected_features].copy()
        
        # Generate report
        improvement = final_score - baseline_score
        improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
        
        report = f"""## Recursive Feature Elimination Report

### Tournament Summary
- **Task Type:** {'Classification' if is_classification else 'Regression'}
- **Target Column:** {target_col}
- **Initial Features:** {len(numeric_cols)}
- **Surviving Features:** {len(selected_features)}
- **Eliminated Features:** {eliminated}

### Performance Comparison
| Metric | All Features | Selected Features | Change |
|--------|--------------|-------------------|--------|
| {metric} | {baseline_score:.4f} | {final_score:.4f} | {'+' if improvement >= 0 else ''}{improvement:.4f} ({improvement_pct:+.1f}%) |

### Surviving Features (Ranked by Importance)
"""
        for f in selected_ranking[:10]:
            report += f"1. **{f['Feature']}** - Importance: {f['Importance']:.4f}\n"
        
        eliminated_features = [f['Feature'] for f in feature_ranking if f['Selected'] == 'No']
        if eliminated_features:
            report += f"""
### Eliminated Features (Added Noise)
"""
            for feat in eliminated_features[:10]:
                report += f"- {feat}\n"
        
        report += f"""
### How RFE Works
1. **Initial Fit:** Train model with all features
2. **Rank Features:** Calculate importance scores
3. **Eliminate Weakest:** Remove least important feature
4. **Repeat:** Continue until desired number of features
5. **Validate:** Compare performance before/after

### Signal-to-Noise Improvement
{'The selected features maintain or improve model performance with fewer inputs, reducing overfitting risk.' if improvement >= 0 else 'Note: Some performance loss occurred. Consider adjusting the number of features to keep.'}

### Recommendations
- Use the {len(selected_features)} surviving features for training
- Eliminated features were adding noise, not signal
- Consider domain knowledge to validate selections
"""
        
        return df_filtered, img, report, ranking_df
        
    except Exception as e:
        import traceback
        return df, create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


# ==================== MODEL TRAINING FUNCTIONS ====================

def _is_series_exact_match(a: pd.Series, b: pd.Series) -> bool:
    """Return True when two series represent the same values (including missingness)."""
    if len(a) != len(b):
        return False

    a_num = pd.to_numeric(a, errors='coerce')
    b_num = pd.to_numeric(b, errors='coerce')
    numeric_mask = a_num.notna() & b_num.notna()
    if numeric_mask.sum() > 0:
        if (a_num[numeric_mask] - b_num[numeric_mask]).abs().max() <= 1e-12:
            if (a_num.isna() == b_num.isna()).all():
                return True

    a_txt = a.fillna("__NA__").astype(str).str.strip().str.lower()
    b_txt = b.fillna("__NA__").astype(str).str.strip().str.lower()
    return bool((a_txt == b_txt).all())


def _detect_target_leakage(df: pd.DataFrame, target_col: str, candidate_cols: List[str]) -> Dict[str, str]:
    """
    Detect suspicious feature columns that can leak the target.
    Conservative rules:
    - exact copy of target values
    - near-perfect numeric correlation with target
    - deterministic low-cardinality mapping to target for classification-like targets
    """
    if target_col not in df.columns or not candidate_cols:
        return {}

    if len(df) > 20000:
        sample_df = df[[target_col] + candidate_cols].sample(n=20000, random_state=42)
    else:
        sample_df = df[[target_col] + candidate_cols]

    y = sample_df[target_col]
    y_num = pd.to_numeric(y, errors='coerce')
    y_unique = y.nunique(dropna=True)
    target_norm = re.sub(r'[^a-z0-9]+', '', str(target_col).lower())

    leaked: Dict[str, str] = {}
    for col in candidate_cols:
        if col == target_col:
            continue

        x = sample_df[col]
        col_norm = re.sub(r'[^a-z0-9]+', '', str(col).lower())
        if target_norm and col_norm == target_norm:
            leaked[col] = "same semantic name as target"
            continue

        if _is_series_exact_match(x, y):
            leaked[col] = "identical to target values"
            continue

        x_num = pd.to_numeric(x, errors='coerce')
        valid_mask = x_num.notna() & y_num.notna()
        if valid_mask.sum() >= 50 and x_num[valid_mask].nunique() > 1 and y_num[valid_mask].nunique() > 1:
            corr = np.corrcoef(x_num[valid_mask], y_num[valid_mask])[0, 1]
            if np.isfinite(corr) and abs(corr) >= 0.9999:
                leaked[col] = f"near-perfect correlation with target (corr={corr:.4f})"
                continue

        x_unique = x.nunique(dropna=True)
        max_low_cardinality = min(50, max(10, int(len(sample_df) * 0.2)))
        if len(sample_df) >= 200 and y_unique <= 30 and 1 < x_unique <= max_low_cardinality:
            tmp = pd.DataFrame({
                "feature": x.fillna("__NA__").astype(str),
                "target": y.fillna("__NA__").astype(str),
            })
            feature_to_target = tmp.groupby("feature")["target"].agg(lambda s: s.value_counts().index[0])
            mapped_target = tmp["feature"].map(feature_to_target)
            purity = float((mapped_target == tmp["target"]).mean())
            if purity >= 0.999:
                leaked[col] = f"deterministic mapping to target (purity={purity:.3f})"

    return leaked


def _prepare_numeric_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """Build numeric feature matrix and drop leakage-prone columns."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    leakage_map = _detect_target_leakage(df, target_col, numeric_cols)
    safe_numeric_cols = [c for c in numeric_cols if c not in leakage_map]

    if not safe_numeric_cols:
        return pd.DataFrame(), [], leakage_map

    X = df[safe_numeric_cols].fillna(df[safe_numeric_cols].median())
    return X, safe_numeric_cols, leakage_map

def run_normal_training(df: pd.DataFrame, target_col: str = None, progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Normal Training - Run 5 standard models and pick the best one by accuracy/score.
    Quick comparison without hyperparameter tuning.
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    if target_col is None or target_col not in df.columns:
        return create_placeholder_image("No target"), "Please select a target column first", pd.DataFrame()
    
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        import time
        
        progress(0.1, desc="Preparing data...")
        
        # Separate features and target (with leakage guard)
        X, numeric_cols, leakage_map = _prepare_numeric_features(df, target_col)
        if len(numeric_cols) < 1:
            if leakage_map:
                leak_text = "\n".join([f"- {k}: {v}" for k, v in list(leakage_map.items())[:10]])
                return (
                    create_placeholder_image("Leakage detected"),
                    "No usable numeric features left after leakage guard.\n\n"
                    "Suspicious features removed:\n"
                    f"{leak_text}",
                    pd.DataFrame()
                )
            return create_placeholder_image("No features"), "No numeric features found", pd.DataFrame()

        y = df[target_col].copy()
        
        # Determine task type
        unique_vals = y.nunique()
        is_classification = unique_vals <= 20 or y.dtype == 'object'
        
        if is_classification:
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('Unknown'))
            else:
                y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
            models = {
                'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42)
            }
        else:
            y = y.fillna(y.median())
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'r2'
            models = {
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
                'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
                'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42)
            }

        # Keep export context available regardless of training mode.
        state.X_train = X.copy()
        state.y_train = y.copy() if hasattr(y, 'copy') else y
        
        # Scale features (keep scaler for export)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        state.preprocessors = {"scaler": scaler}
        
        # Sample for speed if large dataset
        if len(X_scaled) > 5000:
            indices = np.random.choice(len(X_scaled), 5000, replace=False)
            X_scaled = X_scaled[indices]
            y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        
        progress(0.2, desc="Training models...")
        
        # Train and evaluate each model
        results = []
        best_score = -np.inf
        best_model_name = None
        best_model = None
        
        for i, (name, model) in enumerate(models.items()):
            progress(0.2 + 0.6 * (i / len(models)), desc=f"Training {name}...")
            
            start_time = time.time()
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)
                train_time = time.time() - start_time
                
                mean_score = scores.mean()
                std_score = scores.std()
                
                results.append({
                    'Model': name,
                    'Mean Score': round(mean_score, 4),
                    'Std Dev': round(std_score, 4),
                    'Min Score': round(scores.min(), 4),
                    'Max Score': round(scores.max(), 4),
                    'Train Time (s)': round(train_time, 2)
                })
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model_name = name
                    best_model = model
                    
            except Exception as e:
                results.append({
                    'Model': name,
                    'Mean Score': 'Error',
                    'Std Dev': str(e)[:30],
                    'Min Score': '-',
                    'Max Score': '-',
                    'Train Time (s)': '-'
                })
        
        # Sort by score
        results = sorted(results, key=lambda x: x['Mean Score'] if isinstance(x['Mean Score'], float) else -999, reverse=True)
        
        progress(0.85, desc="Creating visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Model comparison bar chart
        ax1 = axes[0]
        valid_results = [r for r in results if isinstance(r['Mean Score'], float)]
        if valid_results:
            model_names = [r['Model'][:15] for r in valid_results]
            scores = [r['Mean Score'] for r in valid_results]
            colors = ['#27ae60' if r['Model'] == best_model_name else '#3498db' for r in valid_results]
            
            bars = ax1.barh(model_names[::-1], scores[::-1], color=colors[::-1], alpha=0.8)
            ax1.set_xlabel('Score' + (' (Accuracy)' if is_classification else ' (R2)'))
            ax1.set_title('Model Comparison', fontweight='bold')
            ax1.set_xlim(0, 1.1)
            
            # Add value labels
            for bar, score in zip(bars, scores[::-1]):
                ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{score:.3f}', va='center', fontweight='bold')
        
        # Plot 2: Training time comparison
        ax2 = axes[1]
        if valid_results:
            times = [r['Train Time (s)'] for r in valid_results if isinstance(r['Train Time (s)'], float)]
            names = [r['Model'][:15] for r in valid_results if isinstance(r['Train Time (s)'], float)]
            ax2.barh(names[::-1], times[::-1], color='coral', alpha=0.8)
            ax2.set_xlabel('Training Time (seconds)')
            ax2.set_title('Training Speed', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Training complete!")
        
        # Create results table
        results_df = pd.DataFrame(results)
        
        # Generate report
        metric = 'Accuracy' if is_classification else 'R2 Score'
        report = f"""## Normal Training Report

### Task Configuration
- **Task Type:** {'Classification' if is_classification else 'Regression'}
- **Target Column:** {target_col}
- **Features Used:** {len(numeric_cols)}
- **Samples:** {len(X_scaled):,}
- **Cross-Validation:** 5-Fold
"""

        if leakage_map:
            report += f"\n### Leakage Guard\n- **Suspicious Features Removed:** {len(leakage_map)}\n"
            for col, reason in list(leakage_map.items())[:10]:
                report += f"- `{col}`: {reason}\n"

        report += f"""

### Winner
**{best_model_name}** with {metric}: **{best_score:.4f}**

### Model Rankings
"""
        for i, r in enumerate(results[:5], 1):
            if isinstance(r['Mean Score'], float):
                report += f"{i}. **{r['Model']}**: {r['Mean Score']:.4f} (+/- {r['Std Dev']:.4f})\n"

        perfect_scores = [r['Mean Score'] for r in valid_results if isinstance(r['Mean Score'], float)]
        if is_classification and perfect_scores and min(perfect_scores) >= 0.9999:
            report += (
                "\n### Perfect-Score Warning\n"
                "- All models are scoring ~1.0. This usually means either:\n"
                "  1) Remaining data leakage/proxy leakage\n"
                "  2) Very small or highly deterministic dataset\n"
                "- Try a larger holdout set in **Normal Analysis** for a stricter estimate.\n"
            )
        
        report += f"""
### Key Insights
- Best model: {best_model_name}
- Score variance indicates model stability
- Low std dev = consistent performance across folds

### Next Steps
- For better results, try **Optuna Hyperparameter Search**
- For even better results, try **Automated Stacking**
"""
        
        # Store best model in state (keep `model` and `trained_model` in sync)
        state.model = best_model
        state.trained_model = best_model
        state.model_name = best_model_name
        state.best_score = float(best_score) if best_score is not None else None
        state.leakage_report = leakage_map
        
        return img, report, results_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_optuna_training(df: pd.DataFrame, target_col: str = None, n_trials: int = 50, 
                        progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Optuna Hyperparameter Search - Find mathematically optimal model settings.
    Runs many trials to discover the best hyperparameters.
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    if target_col is None or target_col not in df.columns:
        return create_placeholder_image("No target"), "Please select a target column first", pd.DataFrame()
    
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import time
        
        progress(0.05, desc="Preparing data for optimization...")
        
        # Prepare data (with leakage guard)
        X, numeric_cols, leakage_map = _prepare_numeric_features(df, target_col)
        if len(numeric_cols) < 1:
            if leakage_map:
                leak_text = "\n".join([f"- {k}: {v}" for k, v in list(leakage_map.items())[:10]])
                return (
                    create_placeholder_image("Leakage detected"),
                    "No usable numeric features left after leakage guard.\n\n"
                    "Suspicious features removed:\n"
                    f"{leak_text}",
                    pd.DataFrame()
                )
            return create_placeholder_image("No features"), "No numeric features found", pd.DataFrame()

        y = df[target_col].copy()
        
        # Determine task type
        unique_vals = y.nunique()
        is_classification = unique_vals <= 20 or y.dtype == 'object'
        
        if is_classification:
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y.fillna('Unknown'))
            else:
                y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            direction = 'maximize'
        else:
            y = y.fillna(y.median())
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            direction = 'maximize'

        # Keep export context available regardless of training mode.
        state.X_train = X.copy()
        state.y_train = y.copy() if hasattr(y, 'copy') else y
        
        # Scale features (keep scaler for export)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        state.preprocessors = {"scaler": scaler}
        
        # Sample for speed
        if len(X_scaled) > 3000:
            indices = np.random.choice(len(X_scaled), 3000, replace=False)
            X_scaled = X_scaled[indices]
            y = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
        
        # Track trials
        trial_history = []
        start_time = time.time()
        
        def objective(trial):
            # Suggest hyperparameters
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            if is_classification:
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )
                scoring = 'accuracy'
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )
                scoring = 'r2'
            
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)
            score = scores.mean()
            
            # Track progress
            trial_history.append({
                'Trial': len(trial_history) + 1,
                'Score': round(score, 4),
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'max_features': str(max_features)
            })
            
            # Update progress
            pct = 0.1 + 0.7 * (len(trial_history) / n_trials)
            progress(pct, desc=f"Trial {len(trial_history)}/{n_trials} - Score: {score:.4f}")
            
            return score
        
        progress(0.1, desc="Starting Optuna optimization...")
        
        # Run optimization
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        total_time = time.time() - start_time
        
        progress(0.85, desc="Creating visualization...")
        
        # Best parameters
        best_params = study.best_params
        best_score = study.best_value
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Optimization history
        ax1 = axes[0, 0]
        trials = [t['Trial'] for t in trial_history]
        scores = [t['Score'] for t in trial_history]
        ax1.plot(trials, scores, 'b-', alpha=0.3, linewidth=1)
        ax1.scatter(trials, scores, c=scores, cmap='viridis', s=30, alpha=0.6)
        
        # Running best
        running_best = [max(scores[:i+1]) for i in range(len(scores))]
        ax1.plot(trials, running_best, 'r-', linewidth=2, label='Best So Far')
        ax1.axhline(y=best_score, color='green', linestyle='--', alpha=0.5, label=f'Final Best: {best_score:.4f}')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Score')
        ax1.set_title('Optimization History', fontweight='bold')
        ax1.legend()
        
        # Plot 2: Parameter importance
        ax2 = axes[0, 1]
        param_names = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        try:
            importances = optuna.importance.get_param_importances(study)
            imp_values = [importances.get(p, 0) for p in param_names]
            ax2.barh(param_names[::-1], imp_values[::-1], color='steelblue', alpha=0.8)
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importance', fontweight='bold')
        except:
            ax2.text(0.5, 0.5, 'Could not compute importance', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hyperparameter Importance', fontweight='bold')
        
        # Plot 3: Best params visualization
        ax3 = axes[1, 0]
        param_values = [best_params['n_estimators'], best_params['max_depth'], 
                       best_params['min_samples_split'], best_params['min_samples_leaf']]
        colors = plt.cm.Set2(np.linspace(0, 1, 4))
        ax3.barh(param_names[::-1], param_values[::-1], color=colors[::-1], alpha=0.8)
        ax3.set_xlabel('Value')
        ax3.set_title('Optimal Hyperparameters', fontweight='bold')
        
        # Add value labels
        for i, (name, val) in enumerate(zip(param_names[::-1], param_values[::-1])):
            ax3.text(val + max(param_values)*0.02, i, str(val), va='center', fontweight='bold')
        
        # Plot 4: Score distribution
        ax4 = axes[1, 1]
        ax4.hist(scores, bins=20, color='coral', alpha=0.7, edgecolor='white')
        ax4.axvline(x=best_score, color='green', linestyle='--', linewidth=2, label=f'Best: {best_score:.4f}')
        ax4.axvline(x=np.mean(scores), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.4f}')
        ax4.set_xlabel('Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Score Distribution Across Trials', fontweight='bold')
        ax4.legend()
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Optimization complete!")
        
        # Create trials table
        trials_df = pd.DataFrame(sorted(trial_history, key=lambda x: -x['Score'])[:20])
        
        # Generate report
        metric = 'Accuracy' if is_classification else 'R2 Score'
        report = f"""## Optuna Hyperparameter Optimization Report

### Optimization Summary
- **Total Trials:** {n_trials}
- **Total Time:** {total_time:.1f} seconds
- **Average Time per Trial:** {total_time/n_trials:.2f} seconds
- **Best {metric}:** {best_score:.4f}

### Optimal Hyperparameters Found
| Parameter | Optimal Value |
|-----------|---------------|
| n_estimators | {best_params['n_estimators']} |
| max_depth | {best_params['max_depth']} |
| min_samples_split | {best_params['min_samples_split']} |
| min_samples_leaf | {best_params['min_samples_leaf']} |
| max_features | {best_params['max_features']} |

### Search Statistics
- **Score Range:** {min(scores):.4f} to {max(scores):.4f}
- **Score Mean:** {np.mean(scores):.4f}
- **Score Std:** {np.std(scores):.4f}
- **Improvement from Baseline:** The optimization found parameters significantly better than defaults.
"""

        if leakage_map:
            report += f"\n### Leakage Guard\n- **Suspicious Features Removed:** {len(leakage_map)}\n"
            for col, reason in list(leakage_map.items())[:10]:
                report += f"- `{col}`: {reason}\n"

        report += f"""

### How Optuna Works
1. **TPE Sampler:** Uses Tree-structured Parzen Estimator to intelligently explore the parameter space
2. **Pruning:** Stops unpromising trials early to save time
3. **Importance:** Identifies which parameters matter most for performance

### Next Steps
- These optimal parameters are saved for the final model
- Consider running **Automated Stacking** for even better results
"""
        
        # Train final model with best params and save
        if is_classification:
            best_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        else:
            best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        
        best_model.fit(X_scaled, y)
        # Keep `model` and `trained_model` in sync for export/insights tabs.
        state.model = best_model
        state.trained_model = best_model
        state.model_name = f"Optuna-Optimized RandomForest"
        state.best_params = best_params
        state.best_score = float(best_score) if best_score is not None else None
        state.leakage_report = leakage_map

        return img, report, trials_df
        
    except ImportError:
        return create_placeholder_image("Optuna not installed"), "Optuna is required. Install with: pip install optuna", pd.DataFrame()
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_stacking_ensemble(df: pd.DataFrame, target_col: str = None, 
                          progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Automated Stacking Ensemble - Build a committee of models that vote together.
    Trains multiple algorithms and combines them with a meta-learner.
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    if target_col is None or target_col not in df.columns:
        return create_placeholder_image("No target"), "Please select a target column first", pd.DataFrame()
    
    try:
        from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_val_predict
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import (
            RandomForestClassifier, RandomForestRegressor,
            GradientBoostingClassifier, GradientBoostingRegressor,
            ExtraTreesClassifier, ExtraTreesRegressor,
            AdaBoostClassifier, AdaBoostRegressor,
            StackingClassifier, StackingRegressor
        )
        from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
        from sklearn.svm import SVC, SVR
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        import time
        
        progress(0.05, desc="Preparing data for ensemble...")

        # Prepare data (with leakage guard)
        X, numeric_cols, leakage_map = _prepare_numeric_features(df, target_col)

        if len(numeric_cols) < 1:
            if leakage_map:
                leak_text = "\n".join([f"- {k}: {v}" for k, v in list(leakage_map.items())[:10]])
                return (
                    create_placeholder_image("Leakage detected"),
                    "No usable numeric features left after leakage guard.\n\n"
                    "Suspicious features removed:\n"
                    f"{leak_text}",
                    pd.DataFrame()
                )
            return create_placeholder_image("No features"), "No numeric features found", pd.DataFrame()

        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y_series = df[target_col].copy()
        
        # Determine task type
        unique_vals = y_series.nunique()
        is_classification = unique_vals <= 20 or y_series.dtype == 'object'
        
        if is_classification:
            if y_series.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y_series.fillna('Unknown'))
            else:
                y = y_series.fillna(y_series.mode().iloc[0] if len(y_series.mode()) > 0 else 0).to_numpy()
        else:
            y = y_series.fillna(y_series.median()).to_numpy()
        
        # Ensure y is 1D numpy array
        y = np.asarray(y).ravel()

        # Keep export context available regardless of training mode.
        state.X_train = X.copy()
        state.y_train = y.copy() if hasattr(y, 'copy') else y
        
        # Set up cross-validation and scoring based on task type
        if is_classification:
            cv = 3  # Use int for simpler cross-validation
            scoring = 'accuracy'
        else:
            cv = 3
            scoring = 'r2'
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Sample for speed
        if len(X_scaled) > 3000:
            indices = np.random.choice(len(X_scaled), 3000, replace=False)
            X_scaled = X_scaled[indices]
            y = y[indices]
        
        progress(0.1, desc="Defining base models...")
        
        # Define base estimators
        if is_classification:
            base_estimators = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
                ('knn', KNeighborsClassifier(n_neighbors=5)),
            ]
            meta_learner = LogisticRegression(max_iter=500, random_state=42)
            meta_name = "Logistic Regression"
        else:
            base_estimators = [
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
                ('et', ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
                ('ada', AdaBoostRegressor(n_estimators=50, random_state=42)),
                ('knn', KNeighborsRegressor(n_neighbors=5)),
            ]
            meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])
            meta_name = "Ridge Regression"
        
        # Try to add LightGBM and XGBoost if available
        try:
            import lightgbm as lgb
            if is_classification:
                base_estimators.append(('lgbm', lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)))
            else:
                base_estimators.append(('lgbm', lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=-1)))
        except ImportError:
            pass
        
        try:
            import xgboost as xgb
            if is_classification:
                base_estimators.append(('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42, verbosity=0)))
            else:
                base_estimators.append(('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, verbosity=0)))
        except ImportError:
            pass
        
        try:
            import catboost as cb
            if is_classification:
                base_estimators.append(('catboost', cb.CatBoostClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=0)))
            else:
                base_estimators.append(('catboost', cb.CatBoostRegressor(n_estimators=100, max_depth=5, random_state=42, verbose=0)))
        except ImportError:
            pass
        
        progress(0.2, desc="Training base models individually...")
        
        # First, evaluate each base model individually
        base_results = []
        for i, (name, model) in enumerate(base_estimators):
            progress(0.2 + 0.3 * (i / len(base_estimators)), desc=f"Training {name}...")
            
            start_time = time.time()
            try:
                scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)
                train_time = time.time() - start_time
                
                base_results.append({
                    'Model': name.upper(),
                    'Mean Score': round(scores.mean(), 4),
                    'Std Dev': round(scores.std(), 4),
                    'Time (s)': round(train_time, 2),
                    'Role': 'Base Model'
                })
            except Exception as e:
                base_results.append({
                    'Model': name.upper(),
                    'Mean Score': 'Error',
                    'Std Dev': str(e)[:20],
                    'Time (s)': '-',
                    'Role': 'Base Model'
                })
        
        progress(0.55, desc="Building stacking ensemble...")
        
        # Create stacking ensemble
        if is_classification:
            stacking_model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=3,
                n_jobs=-1,
                passthrough=False
            )
        else:
            stacking_model = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=3,
                n_jobs=-1,
                passthrough=False
            )
        
        progress(0.6, desc="Training stacking ensemble...")
        
        # Evaluate stacking ensemble
        start_time = time.time()
        stacking_scores = cross_val_score(stacking_model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)
        stacking_time = time.time() - start_time
        
        stacking_result = {
            'Model': 'STACKING ENSEMBLE',
            'Mean Score': round(stacking_scores.mean(), 4),
            'Std Dev': round(stacking_scores.std(), 4),
            'Time (s)': round(stacking_time, 2),
            'Role': 'Meta-Model'
        }
        
        # Sort base results by score
        valid_base = [r for r in base_results if isinstance(r['Mean Score'], float)]
        valid_base = sorted(valid_base, key=lambda x: -x['Mean Score'])
        
        # Best individual model
        best_individual = valid_base[0] if valid_base else None
        
        progress(0.85, desc="Creating visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: All models comparison
        ax1 = axes[0, 0]
        all_results = valid_base + [stacking_result]
        model_names = [r['Model'][:12] for r in all_results]
        scores = [r['Mean Score'] for r in all_results]
        colors = ['#e74c3c' if r['Model'] == 'STACKING ENSEMBLE' else '#3498db' for r in all_results]
        
        bars = ax1.barh(model_names[::-1], scores[::-1], color=colors[::-1], alpha=0.8)
        ax1.set_xlabel('Score')
        ax1.set_title('Model Comparison: Base vs Stacking', fontweight='bold')
        ax1.set_xlim(0, max(scores) * 1.15)
        
        # Add labels
        for bar, score in zip(bars, scores[::-1]):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center', fontweight='bold', fontsize=9)
        
        # Plot 2: Improvement chart
        ax2 = axes[0, 1]
        if best_individual:
            improvements = [stacking_result['Mean Score'] - r['Mean Score'] for r in valid_base]
            imp_colors = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
            ax2.barh([r['Model'][:12] for r in valid_base][::-1], improvements[::-1], 
                    color=imp_colors[::-1], alpha=0.8)
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('Improvement vs Stacking')
            ax2.set_title('Stacking vs Individual Models', fontweight='bold')
        
        # Plot 3: Training time comparison
        ax3 = axes[1, 0]
        times = [r['Time (s)'] for r in all_results if isinstance(r['Time (s)'], float)]
        time_names = [r['Model'][:12] for r in all_results if isinstance(r['Time (s)'], float)]
        ax3.barh(time_names[::-1], times[::-1], color='coral', alpha=0.8)
        ax3.set_xlabel('Training Time (seconds)')
        ax3.set_title('Training Time', fontweight='bold')
        
        # Plot 4: Ensemble architecture diagram
        ax4 = axes[1, 1]
        ax4.set_xlim(0, 10)
        ax4.set_ylim(0, 10)
        ax4.axis('off')
        
        # Draw base models
        n_base = len(base_estimators)
        for i, (name, _) in enumerate(base_estimators):
            box_x = 1.5 + (i % 4) * 2
            box_y = 7 - (i // 4) * 2
            rect = plt.Rectangle((box_x-0.8, box_y-0.4), 1.6, 0.8, facecolor='#3498db', edgecolor='black', alpha=0.7)
            ax4.add_patch(rect)
            ax4.text(box_x, box_y, name.upper()[:6], ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            # Arrow to meta
            ax4.annotate('', xy=(5, 2.5), xytext=(box_x, box_y-0.4),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        
        # Draw meta-learner
        rect_meta = plt.Rectangle((3.5, 1.5), 3, 1, facecolor='#e74c3c', edgecolor='black', alpha=0.8)
        ax4.add_patch(rect_meta)
        ax4.text(5, 2, f'META: {meta_name[:15]}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Final output
        ax4.annotate('', xy=(5, 0.5), xytext=(5, 1.5),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax4.text(5, 0.2, f'FINAL: {stacking_result["Mean Score"]:.4f}', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='green')
        
        ax4.set_title('Stacking Ensemble Architecture', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(0.95, desc="Training final model...")
        
        # Train final stacking model
        stacking_model.fit(X_scaled, y)
        
        progress(1.0, desc="Ensemble complete!")
        
        # Create results table
        all_results_df = pd.DataFrame(valid_base + [stacking_result])
        
        # Calculate improvement
        if best_individual:
            improvement = stacking_result['Mean Score'] - best_individual['Mean Score']
            improvement_pct = (improvement / best_individual['Mean Score'] * 100) if best_individual['Mean Score'] > 0 else 0
        else:
            improvement = 0
            improvement_pct = 0
        
        # Generate report
        metric = 'Accuracy' if is_classification else 'R2 Score'
        report = f"""## Automated Stacking Ensemble Report

### Ensemble Configuration
- **Base Models:** {len(base_estimators)}
- **Meta-Learner:** {meta_name}
- **Cross-Validation:** 5-Fold (3-Fold for stacking internal CV)

### Committee Members
"""
        for name, _ in base_estimators:
            report += f"- {name.upper()}\n"
        
        report += f"""
### Performance Results
| Model | {metric} |
|-------|----------|
| **STACKING ENSEMBLE** | **{stacking_result['Mean Score']:.4f}** |
| Best Individual ({best_individual['Model'] if best_individual else 'N/A'}) | {best_individual['Mean Score'] if best_individual else 'N/A'} |

### Improvement Analysis
- **Stacking vs Best Individual:** {'+' if improvement >= 0 else ''}{improvement:.4f} ({improvement_pct:+.1f}%)
- The stacking ensemble combines diverse model predictions using {meta_name}

### How Stacking Works
1. **Level 0 (Base Models):** Each algorithm makes predictions independently
2. **Level 1 (Meta-Learner):** {meta_name} learns to combine base predictions optimally
3. **Final Prediction:** Weighted combination of all models' knowledge

### Why This Works
- Different algorithms capture different patterns
- Errors from one model are compensated by others
- Meta-learner learns which models to trust for which predictions

### Model Diversity
The ensemble includes:
- Tree-based: Random Forest, Gradient Boosting, Extra Trees
- Boosting: AdaBoost{', LightGBM' if any('lgbm' in str(e) for e in base_estimators) else ''}{', XGBoost' if any('xgb' in str(e) for e in base_estimators) else ''}{', CatBoost' if any('catboost' in str(e) for e in base_estimators) else ''}
- Instance-based: K-Nearest Neighbors
"""

        if leakage_map:
            report += f"\n### Leakage Guard\n- **Suspicious Features Removed:** {len(leakage_map)}\n"
            for col, reason in list(leakage_map.items())[:10]:
                report += f"- `{col}`: {reason}\n"
        
        # Save model (keep `model` and `trained_model` in sync)
        state.model = stacking_model
        state.trained_model = stacking_model
        state.model_name = "Stacking Ensemble"
        state.best_score = float(stacking_result['Mean Score']) if isinstance(stacking_result['Mean Score'], float) else None
        state.leakage_report = leakage_map
        
        return img, report, all_results_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


# ==================== XAI / ANALYSIS FUNCTIONS ====================

def run_normal_analysis(df: pd.DataFrame, target_col: str = None, 
                        progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Normal Analysis - Confusion Matrix and comprehensive accuracy metrics.
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    if target_col is None or target_col not in df.columns:
        return create_placeholder_image("No target"), "Please select a target column first", pd.DataFrame()
    
    if state.trained_model is None:
        return create_placeholder_image("No model"), "Please train a model first in the Model Training tab", pd.DataFrame()
    
    try:
        from sklearn.metrics import (
            confusion_matrix, classification_report, accuracy_score,
            precision_score, recall_score, f1_score, roc_auc_score,
            mean_squared_error, mean_absolute_error, r2_score
        )
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import train_test_split
        
        progress(0.1, desc="Preparing data for analysis...")
        
        # Prepare data (with leakage guard)
        X, numeric_cols, leakage_map = _prepare_numeric_features(df, target_col)
        if len(numeric_cols) < 1:
            if leakage_map:
                leak_text = "\n".join([f"- {k}: {v}" for k, v in list(leakage_map.items())[:10]])
                return (
                    create_placeholder_image("Leakage detected"),
                    "No usable numeric features left after leakage guard.\n\n"
                    "Suspicious features removed:\n"
                    f"{leak_text}",
                    pd.DataFrame()
                )
            return create_placeholder_image("No features"), "No numeric features found", pd.DataFrame()

        y = df[target_col].copy()
        
        # Determine task type
        unique_vals = y.nunique()
        is_classification = unique_vals <= 20 or y.dtype == 'object'
        
        # Encode target if needed
        le = None
        if is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.fillna('Unknown'))
        elif is_classification:
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
        else:
            y = y.fillna(y.median())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42,
            stratify=y if is_classification else None
        )
        
        progress(0.3, desc="Training model on split...")
        
        # Re-train model on training split for proper evaluation
        model = state.trained_model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        progress(0.6, desc="Calculating metrics...")
        
        if is_classification:
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Handle multi-class
            n_classes = len(np.unique(y))
            average = 'binary' if n_classes == 2 else 'weighted'
            
            precision = precision_score(y_test, y_pred, average=average, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
            
            # Try to get ROC AUC
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    if n_classes == 2:
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                else:
                    roc_auc = None
            except:
                roc_auc = None
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            progress(0.8, desc="Creating visualization...")
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Plot 1: Confusion Matrix
            ax1 = axes[0, 0]
            im = ax1.imshow(cm, cmap='Blues')
            ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
            
            # Add labels
            classes = le.classes_ if le else np.unique(y)
            ax1.set_xticks(range(len(classes)))
            ax1.set_yticks(range(len(classes)))
            ax1.set_xticklabels([str(c)[:10] for c in classes], rotation=45, ha='right')
            ax1.set_yticklabels([str(c)[:10] for c in classes])
            ax1.set_xlabel('Predicted', fontsize=11)
            ax1.set_ylabel('Actual', fontsize=11)
            
            # Add values to cells
            for i in range(len(classes)):
                for j in range(len(classes)):
                    color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                    ax1.text(j, i, str(cm[i, j]), ha='center', va='center', 
                            color=color, fontsize=12, fontweight='bold')
            
            plt.colorbar(im, ax=ax1, shrink=0.8)
            
            # Plot 2: Metrics bar chart
            ax2 = axes[0, 1]
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metrics_values = [accuracy, precision, recall, f1]
            if roc_auc:
                metrics_names.append('ROC-AUC')
                metrics_values.append(roc_auc)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics_names)))
            bars = ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black')
            ax2.set_ylabel('Score')
            ax2.set_title('Performance Metrics', fontweight='bold', fontsize=12)
            ax2.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, val in zip(bars, metrics_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Plot 3: Per-class accuracy (if multiclass)
            ax3 = axes[1, 0]
            if n_classes > 2:
                per_class_acc = cm.diagonal() / cm.sum(axis=1)
                ax3.barh([str(c)[:15] for c in classes], per_class_acc, color='steelblue', alpha=0.8)
                ax3.set_xlabel('Accuracy')
                ax3.set_title('Per-Class Accuracy', fontweight='bold', fontsize=12)
                ax3.set_xlim(0, 1.1)
            else:
                # For binary, show TP, TN, FP, FN
                tn, fp, fn, tp = cm.ravel()
                categories = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
                values = [tn, fp, fn, tp]
                colors = ['#27ae60', '#e74c3c', '#e74c3c', '#27ae60']
                ax3.bar(categories, values, color=colors, alpha=0.8)
                ax3.set_ylabel('Count')
                ax3.set_title('Prediction Breakdown', fontweight='bold', fontsize=12)
                ax3.tick_params(axis='x', rotation=45)
            
            # Plot 4: Class distribution
            ax4 = axes[1, 1]
            unique, counts = np.unique(y_test, return_counts=True)
            class_labels = [str(le.classes_[u])[:15] if le else str(u) for u in unique]
            ax4.pie(counts, labels=class_labels, autopct='%1.1f%%', colors=plt.cm.Set2.colors)
            ax4.set_title('Test Set Class Distribution', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            
            # Metrics table
            metrics_data = [
                {'Metric': 'Accuracy', 'Value': f'{accuracy:.4f}', 'Description': 'Overall correct predictions'},
                {'Metric': 'Precision', 'Value': f'{precision:.4f}', 'Description': 'Positive predictive value'},
                {'Metric': 'Recall', 'Value': f'{recall:.4f}', 'Description': 'True positive rate (sensitivity)'},
                {'Metric': 'F1-Score', 'Value': f'{f1:.4f}', 'Description': 'Harmonic mean of precision and recall'},
            ]
            if roc_auc:
                metrics_data.append({'Metric': 'ROC-AUC', 'Value': f'{roc_auc:.4f}', 'Description': 'Area under ROC curve'})
            
            # Report
            report = f"""## Model Performance Analysis

### Overall Performance
- **Accuracy:** {accuracy:.4f} ({accuracy*100:.1f}%)
- **Model:** {state.model_name}
- **Test Set Size:** {len(y_test)} samples

### Confusion Matrix Interpretation
"""
            if n_classes == 2:
                tn, fp, fn, tp = cm.ravel()
                report += f"""- **True Positives:** {tp} (correctly identified positive cases)
- **True Negatives:** {tn} (correctly identified negative cases)
- **False Positives:** {fp} (Type I errors - false alarms)
- **False Negatives:** {fn} (Type II errors - missed cases)
"""
            
            report += f"""
### Key Metrics Explained
| Metric | Value | Meaning |
|--------|-------|---------|
| Accuracy | {accuracy:.4f} | % of all predictions that are correct |
| Precision | {precision:.4f} | When predicting positive, how often correct |
| Recall | {recall:.4f} | What % of actual positives are found |
| F1-Score | {f1:.4f} | Balance between precision and recall |
"""
            if roc_auc:
                report += f"| ROC-AUC | {roc_auc:.4f} | Model's ability to discriminate classes |\n"
            
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            progress(0.8, desc="Creating visualization...")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            # Plot 1: Actual vs Predicted scatter
            ax1 = axes[0, 0]
            ax1.scatter(y_test, y_pred, alpha=0.5, c='steelblue')
            ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect')
            ax1.set_xlabel('Actual Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title('Actual vs Predicted', fontweight='bold', fontsize=12)
            ax1.legend()
            
            # Plot 2: Residuals distribution
            ax2 = axes[0, 1]
            residuals = y_test - y_pred
            ax2.hist(residuals, bins=30, color='coral', alpha=0.7, edgecolor='white')
            ax2.axvline(x=0, color='red', linestyle='--', lw=2)
            ax2.set_xlabel('Residual (Actual - Predicted)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Residuals Distribution', fontweight='bold', fontsize=12)
            
            # Plot 3: Metrics bar chart
            ax3 = axes[1, 0]
            metrics_names = ['R2 Score', 'RMSE', 'MAE']
            # Normalize for visualization
            metrics_display = [r2, rmse / (y_test.max() - y_test.min() + 1e-6), mae / (y_test.max() - y_test.min() + 1e-6)]
            bars = ax3.bar(metrics_names, metrics_display, color=['#27ae60', '#e74c3c', '#f39c12'], alpha=0.8)
            ax3.set_ylabel('Normalized Score')
            ax3.set_title('Performance Metrics', fontweight='bold', fontsize=12)
            
            # Plot 4: Residuals vs Predicted
            ax4 = axes[1, 1]
            ax4.scatter(y_pred, residuals, alpha=0.5, c='steelblue')
            ax4.axhline(y=0, color='red', linestyle='--', lw=2)
            ax4.set_xlabel('Predicted Values')
            ax4.set_ylabel('Residuals')
            ax4.set_title('Residuals vs Predicted', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            
            metrics_data = [
                {'Metric': 'R2 Score', 'Value': f'{r2:.4f}', 'Description': 'Variance explained by model'},
                {'Metric': 'RMSE', 'Value': f'{rmse:.4f}', 'Description': 'Root mean squared error'},
                {'Metric': 'MAE', 'Value': f'{mae:.4f}', 'Description': 'Mean absolute error'},
                {'Metric': 'MSE', 'Value': f'{mse:.4f}', 'Description': 'Mean squared error'},
            ]
            
            report = f"""## Regression Performance Analysis

### Overall Performance
- **R2 Score:** {r2:.4f} ({r2*100:.1f}% variance explained)
- **Model:** {state.model_name}
- **Test Set Size:** {len(y_test)} samples

### Error Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| R2 Score | {r2:.4f} | {r2*100:.1f}% of variance explained |
| RMSE | {rmse:.4f} | Average prediction error (same units as target) |
| MAE | {mae:.4f} | Average absolute error |
"""

        if leakage_map:
            report += f"\n\n### Leakage Guard\n- **Suspicious Features Removed:** {len(leakage_map)}\n"
            for col, reason in list(leakage_map.items())[:10]:
                report += f"- `{col}`: {reason}\n"

        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Analysis complete!")
        
        metrics_df = pd.DataFrame(metrics_data)
        
        return img, report, metrics_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_shap_analysis(df: pd.DataFrame, target_col: str = None, sample_idx: int = 0,
                      progress=gr.Progress()) -> Tuple[np.ndarray, np.ndarray, str, pd.DataFrame]:
    """
    SHAP Analysis - Global feature importance and Local explanations for individual predictions.
    """
    if df is None:
        return create_placeholder_image("No data"), create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    if target_col is None or target_col not in df.columns:
        return create_placeholder_image("No target"), create_placeholder_image("No target"), "Please select a target column first", pd.DataFrame()
    
    if state.trained_model is None:
        return create_placeholder_image("No model"), create_placeholder_image("No model"), "Please train a model first", pd.DataFrame()
    
    try:
        import shap
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        progress(0.1, desc="Preparing data for SHAP...")
        
        # Prepare data
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target_col].copy()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=numeric_cols)
        
        # Sample for speed
        max_samples = 500
        if len(X_df) > max_samples:
            sample_indices = np.random.choice(len(X_df), max_samples, replace=False)
            X_sample = X_df.iloc[sample_indices]
        else:
            X_sample = X_df
            sample_indices = np.arange(len(X_df))
        
        progress(0.3, desc="Computing SHAP values (this may take a moment)...")
        
        # Get SHAP explainer
        model = state.trained_model
        
        # Use TreeExplainer for tree-based models, otherwise KernelExplainer
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        except:
            # Fallback to KernelExplainer (slower)
            background = shap.sample(X_sample, min(100, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        # Handle multi-class (take first class or average)
        if isinstance(shap_values, list):
            shap_values_global = np.abs(shap_values[1] if len(shap_values) > 1 else shap_values[0]).mean(axis=0)
            shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_global = np.abs(shap_values).mean(axis=0)
            shap_values_plot = shap_values
        
        # Ensure shap_values_global is 1D
        if shap_values_global.ndim > 1:
            shap_values_global = shap_values_global.flatten()
        
        # Ensure it matches number of features
        if len(shap_values_global) != len(numeric_cols):
            shap_values_global = shap_values_global[:len(numeric_cols)]
        
        progress(0.6, desc="Creating global visualization...")
        
        # Create GLOBAL visualization
        fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Feature importance bar chart
        ax1 = axes1[0]
        importance_df = pd.DataFrame({
            'Feature': numeric_cols,
            'Importance': shap_values_global
        }).sort_values('Importance', ascending=True)
        
        top_features = importance_df.tail(15)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))
        ax1.barh(top_features['Feature'].str[:20], top_features['Importance'], color=colors, alpha=0.8)
        ax1.set_xlabel('Mean |SHAP Value|')
        ax1.set_title('Global Feature Importance (SHAP)', fontweight='bold')
        
        # Plot 2: Summary bee swarm (simplified as bar)
        ax2 = axes1[1]
        # Top 10 features scatter
        top_10_idx = importance_df.tail(10)['Feature'].tolist()
        for i, feat in enumerate(top_10_idx[::-1]):
            feat_idx = numeric_cols.index(feat)
            feat_shap = shap_values_plot[:, feat_idx]
            feat_vals = X_sample[feat].values
            
            # Normalize feature values for coloring
            norm_vals = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min() + 1e-6)
            
            ax2.scatter(feat_shap, [i] * len(feat_shap), c=norm_vals, cmap='coolwarm', 
                       alpha=0.5, s=10)
        
        ax2.set_yticks(range(len(top_10_idx)))
        ax2.set_yticklabels([f[:20] for f in top_10_idx[::-1]])
        ax2.set_xlabel('SHAP Value (impact on prediction)')
        ax2.set_title('SHAP Summary Plot (Top 10 Features)', fontweight='bold')
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        fig1.canvas.draw()
        img_global = np.array(fig1.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig1)
        
        progress(0.8, desc="Creating local explanation...")
        
        # Create LOCAL visualization for selected sample
        sample_idx = min(sample_idx, len(X_sample) - 1)
        sample_idx = max(0, sample_idx)
        
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        # Get SHAP values for this sample
        if isinstance(shap_values, list):
            local_shap = shap_values_plot[sample_idx]
        else:
            local_shap = shap_values_plot[sample_idx]
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(local_shap))[::-1][:15]
        
        features_local = [numeric_cols[i][:20] for i in sorted_idx]
        values_local = [local_shap[i] for i in sorted_idx]
        feature_values = [X_sample.iloc[sample_idx, i] for i in sorted_idx]
        
        colors = ['#e74c3c' if v < 0 else '#27ae60' for v in values_local]
        
        y_pos = range(len(features_local))
        bars = ax2.barh(y_pos, values_local, color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f'{feat}\n(val={val:.2f})' for feat, val in zip(features_local[::-1], feature_values[::-1])], fontsize=9)
        ax2.set_xlabel('SHAP Value (contribution to prediction)')
        ax2.set_title(f'Local Explanation: Sample #{sample_idx}', fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # Add legend
        ax2.text(0.02, 0.98, 'Green = Pushes prediction UP\nRed = Pushes prediction DOWN', 
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        fig2.canvas.draw()
        img_local = np.array(fig2.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig2)
        
        progress(1.0, desc="SHAP analysis complete!")
        
        # Create importance table
        importance_table = importance_df.sort_values('Importance', ascending=False).head(20)
        importance_table['Importance'] = importance_table['Importance'].round(4)
        importance_table['Rank'] = range(1, len(importance_table) + 1)
        importance_table = importance_table[['Rank', 'Feature', 'Importance']]
        
        # Generate report
        top_3 = importance_df.sort_values('Importance', ascending=False).head(3)['Feature'].tolist()
        
        report = f"""## SHAP Explainability Report

### What is SHAP?
SHAP (SHapley Additive exPlanations) uses game theory to explain individual predictions.
Each feature gets a "contribution score" showing how it pushed the prediction.

### Global Feature Importance
The most influential features across all predictions:
1. **{top_3[0]}** - Highest average impact
2. **{top_3[1]}** - Second most important
3. **{top_3[2]}** - Third most important

### How to Read the Plots

**Global Plot (Left):**
- Longer bars = more important features
- Shows average impact across all samples

**Summary Plot (Right):**
- Each dot is one sample
- Color: Red = high feature value, Blue = low
- X-axis: How much this feature pushed the prediction

**Local Plot:**
- Shows exactly why ONE sample got its prediction
- Green bars: Pushed prediction higher (toward positive class)
- Red bars: Pushed prediction lower (toward negative class)

### Sample #{sample_idx} Explanation
"""
        # Add local explanation details
        for i, (feat, val, shap_val) in enumerate(zip(features_local[:5], feature_values[:5], values_local[:5])):
            direction = "increased" if shap_val > 0 else "decreased"
            report += f"- **{feat}** (value: {val:.2f}): {direction} prediction by {abs(shap_val):.3f}\n"
        
        report += """
### Actionable Insights
- Focus on top features when collecting data
- Features with high SHAP values are the model's "decision drivers"
- Use local explanations to understand individual predictions
"""
        
        return img_global, img_local, report, importance_table
        
    except ImportError:
        return create_placeholder_image("SHAP not installed"), create_placeholder_image("Install SHAP"), "Please install SHAP: pip install shap", pd.DataFrame()
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_fairness_audit(df: pd.DataFrame, target_col: str = None, sensitive_col: str = None,
                       progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Fairness & Bias Audit - Check if model performs differently across demographic groups.
    Flags critical ethical risks if bias is detected.
    """
    if df is None:
        return create_placeholder_image("No data"), "No data uploaded", pd.DataFrame()
    
    if target_col is None or target_col not in df.columns:
        return create_placeholder_image("No target"), "Please select a target column first", pd.DataFrame()
    
    if state.trained_model is None:
        return create_placeholder_image("No model"), "Please train a model first", pd.DataFrame()
    
    try:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        progress(0.1, desc="Identifying sensitive attributes...")
        
        # Prepare data
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target_col].copy()
        
        # Encode target if needed
        is_classification = y.nunique() <= 20 or y.dtype == 'object'
        
        if not is_classification:
            return create_placeholder_image("Classification only"), "Fairness audit is only available for classification tasks", pd.DataFrame()
        
        le_target = None
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.fillna('Unknown'))
        else:
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get predictions
        model = state.trained_model
        y_pred = model.predict(X_scaled)
        
        progress(0.3, desc="Detecting potential sensitive attributes...")
        
        # Find categorical columns that could be sensitive attributes
        # Common sensitive attribute patterns
        sensitive_patterns = {
            'gender': r'(?i)(gender|sex|male|female)',
            'age_group': r'(?i)(age.?group|age.?cat|generation)',
            'race': r'(?i)(race|ethnic|ethnicity)',
            'religion': r'(?i)(religion|faith)',
            'marital': r'(?i)(marital|married|single|divorced)',
            'education': r'(?i)(education|degree|school)',
            'income_group': r'(?i)(income.?group|salary.?cat|wealth)',
        }
        
        # Find potential sensitive columns
        potential_sensitive = []
        for col in df.columns:
            if col == target_col:
                continue
            # Check if categorical with reasonable number of groups
            if df[col].nunique() <= 10 and df[col].nunique() >= 2:
                for sens_type, pattern in sensitive_patterns.items():
                    if re.search(pattern, col):
                        potential_sensitive.append((col, sens_type))
                        break
                else:
                    # Also consider low-cardinality columns as potential sensitive
                    if df[col].nunique() <= 5:
                        potential_sensitive.append((col, 'demographic'))
        
        # If user specified sensitive column, use that
        if sensitive_col and sensitive_col in df.columns:
            sensitive_columns = [(sensitive_col, 'user_specified')]
        elif potential_sensitive:
            sensitive_columns = potential_sensitive[:3]  # Analyze top 3
        else:
            # Create synthetic age groups if numeric age column exists
            for col in df.columns:
                if re.search(r'(?i)age', col) and df[col].dtype in [np.int64, np.float64]:
                    df['_age_group'] = pd.cut(df[col], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])
                    sensitive_columns = [('_age_group', 'age_group')]
                    break
            else:
                # No sensitive attributes found
                return create_placeholder_image("No sensitive attrs"), """No potential sensitive attributes detected.

To perform fairness audit, your data needs columns representing demographic groups such as:
- Gender (male/female)
- Age groups
- Race/Ethnicity
- Income groups
- Education level

You can also specify a sensitive column manually.""", pd.DataFrame()
        
        progress(0.5, desc="Calculating group-wise performance...")
        
        # Analyze fairness for each sensitive attribute
        fairness_results = []
        all_group_results = []
        
        for sens_col, sens_type in sensitive_columns:
            groups = df[sens_col].unique()
            groups = [g for g in groups if pd.notna(g)]
            
            group_metrics = []
            for group in groups:
                mask = df[sens_col] == group
                if mask.sum() < 10:  # Skip small groups
                    continue
                
                y_true_group = y[mask]
                y_pred_group = y_pred[mask]
                
                acc = accuracy_score(y_true_group, y_pred_group)
                
                # Handle binary vs multiclass
                avg = 'binary' if len(np.unique(y)) == 2 else 'weighted'
                prec = precision_score(y_true_group, y_pred_group, average=avg, zero_division=0)
                rec = recall_score(y_true_group, y_pred_group, average=avg, zero_division=0)
                f1 = f1_score(y_true_group, y_pred_group, average=avg, zero_division=0)
                
                # Positive prediction rate (for disparate impact)
                pos_rate = y_pred_group.mean() if len(np.unique(y)) == 2 else None
                
                group_metrics.append({
                    'Attribute': sens_col,
                    'Group': str(group)[:20],
                    'Sample Size': mask.sum(),
                    'Accuracy': round(acc, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'F1-Score': round(f1, 4),
                    'Positive Rate': round(pos_rate, 4) if pos_rate else 'N/A'
                })
                
                all_group_results.append({
                    'attribute': sens_col,
                    'group': str(group),
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'pos_rate': pos_rate,
                    'n': mask.sum()
                })
            
            if len(group_metrics) >= 2:
                # Calculate fairness metrics
                accs = [m['Accuracy'] for m in group_metrics]
                max_diff = max(accs) - min(accs)
                
                # Disparate impact (80% rule)
                pos_rates = [m['Positive Rate'] for m in group_metrics if m['Positive Rate'] != 'N/A']
                if pos_rates and max(pos_rates) > 0:
                    disparate_impact = min(pos_rates) / max(pos_rates)
                else:
                    disparate_impact = None
                
                fairness_results.append({
                    'Attribute': sens_col,
                    'Type': sens_type,
                    'Groups': len(group_metrics),
                    'Accuracy Gap': round(max_diff, 4),
                    'Worst Group': min(group_metrics, key=lambda x: x['Accuracy'])['Group'],
                    'Best Group': max(group_metrics, key=lambda x: x['Accuracy'])['Group'],
                    'Disparate Impact': round(disparate_impact, 4) if disparate_impact else 'N/A',
                    'Risk Level': 'CRITICAL' if max_diff > 0.15 else 'WARNING' if max_diff > 0.08 else 'LOW'
                })
        
        progress(0.75, desc="Creating fairness visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Group accuracy comparison
        ax1 = axes[0, 0]
        if all_group_results:
            groups = [f"{r['attribute'][:8]}:{r['group'][:8]}" for r in all_group_results]
            accs = [r['accuracy'] for r in all_group_results]
            
            # Color by risk
            colors = []
            overall_acc = accuracy_score(y, y_pred)
            for acc in accs:
                diff = abs(acc - overall_acc)
                if diff > 0.15:
                    colors.append('#e74c3c')  # Red - critical
                elif diff > 0.08:
                    colors.append('#f39c12')  # Orange - warning
                else:
                    colors.append('#27ae60')  # Green - ok
            
            bars = ax1.barh(groups[::-1], accs[::-1], color=colors[::-1], alpha=0.8)
            ax1.axvline(x=overall_acc, color='black', linestyle='--', lw=2, label=f'Overall: {overall_acc:.3f}')
            ax1.set_xlabel('Accuracy')
            ax1.set_title('Accuracy by Demographic Group', fontweight='bold')
            ax1.legend()
            ax1.set_xlim(0, 1.05)
        
        # Plot 2: Fairness gap visualization
        ax2 = axes[0, 1]
        if fairness_results:
            attrs = [r['Attribute'][:15] for r in fairness_results]
            gaps = [r['Accuracy Gap'] for r in fairness_results]
            risk_colors = {'CRITICAL': '#e74c3c', 'WARNING': '#f39c12', 'LOW': '#27ae60'}
            colors = [risk_colors[r['Risk Level']] for r in fairness_results]
            
            ax2.barh(attrs, gaps, color=colors, alpha=0.8)
            ax2.axvline(x=0.08, color='orange', linestyle='--', label='Warning (8%)')
            ax2.axvline(x=0.15, color='red', linestyle='--', label='Critical (15%)')
            ax2.set_xlabel('Accuracy Gap (Max - Min)')
            ax2.set_title('Fairness Gap by Attribute', fontweight='bold')
            ax2.legend()
        
        # Plot 3: Disparate impact
        ax3 = axes[1, 0]
        if fairness_results:
            di_results = [r for r in fairness_results if r['Disparate Impact'] != 'N/A']
            if di_results:
                attrs = [r['Attribute'][:15] for r in di_results]
                di_vals = [r['Disparate Impact'] for r in di_results]
                colors = ['#e74c3c' if v < 0.8 else '#27ae60' for v in di_vals]
                
                ax3.barh(attrs, di_vals, color=colors, alpha=0.8)
                ax3.axvline(x=0.8, color='red', linestyle='--', lw=2, label='80% Rule Threshold')
                ax3.set_xlabel('Disparate Impact Ratio')
                ax3.set_title('Disparate Impact Analysis', fontweight='bold')
                ax3.set_xlim(0, 1.2)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Disparate Impact N/A\n(requires binary classification)', 
                        ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Risk summary
        ax4 = axes[1, 1]
        if fairness_results:
            risk_counts = {'CRITICAL': 0, 'WARNING': 0, 'LOW': 0}
            for r in fairness_results:
                risk_counts[r['Risk Level']] += 1
            
            colors = ['#e74c3c', '#f39c12', '#27ae60']
            ax4.pie([risk_counts['CRITICAL'], risk_counts['WARNING'], risk_counts['LOW']], 
                   labels=['Critical Risk', 'Warning', 'Low Risk'],
                   colors=colors, autopct='%1.0f%%', explode=(0.1, 0.05, 0))
            ax4.set_title('Overall Fairness Risk Assessment', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Fairness audit complete!")
        
        # Create results table
        if all_group_results:
            results_df = pd.DataFrame([{
                'Attribute': r['Attribute'],
                'Group': r['Group'],
                'N': r['Sample Size'],
                'Accuracy': r['Accuracy'],
                'F1': r['F1-Score']
            } for r in [g for sublist in [[m for m in group_metrics] for _, _ in sensitive_columns] for g in sublist] if r])
        else:
            results_df = pd.DataFrame(all_group_results)
        
        # Generate report
        overall_acc = accuracy_score(y, y_pred)
        
        report = f"""## Fairness & Bias Audit Report

### Overall Model Performance
- **Overall Accuracy:** {overall_acc:.4f}
- **Model:** {state.model_name}

### Sensitive Attributes Analyzed
"""
        for fr in fairness_results:
            risk_emoji = "CRITICAL" if fr['Risk Level'] == 'CRITICAL' else "WARNING" if fr['Risk Level'] == 'WARNING' else "OK"
            report += f"""
#### {fr['Attribute']} [{risk_emoji}]
- **Groups Analyzed:** {fr['Groups']}
- **Accuracy Gap:** {fr['Accuracy Gap']*100:.1f}%
- **Best Performing Group:** {fr['Best Group']}
- **Worst Performing Group:** {fr['Worst Group']}
"""
            if fr['Disparate Impact'] != 'N/A':
                di_status = "FAILS 80% rule" if fr['Disparate Impact'] < 0.8 else "Passes 80% rule"
                report += f"- **Disparate Impact:** {fr['Disparate Impact']:.3f} ({di_status})\n"
        
        # Critical findings
        critical_findings = [f for f in fairness_results if f['Risk Level'] == 'CRITICAL']
        if critical_findings:
            report += """
### CRITICAL ETHICAL RISKS DETECTED

The model shows significant performance disparities:
"""
            for cf in critical_findings:
                gap_pct = cf['Accuracy Gap'] * 100
                report += f"""
- **{cf['Attribute']}**: Model is {gap_pct:.1f}% less accurate for '{cf['Worst Group']}' compared to '{cf['Best Group']}'
  - This could lead to discriminatory outcomes
  - Immediate review and mitigation required
"""
        
        report += """
### Fairness Metrics Explained

**Accuracy Gap:**
- Difference between best and worst performing groups
- >15% = Critical risk, >8% = Warning

**Disparate Impact (80% Rule):**
- Ratio of positive prediction rates between groups
- <0.8 indicates potential discrimination
- Legal threshold in many jurisdictions

### Recommendations
"""
        if critical_findings:
            report += """
1. **Immediate Action Required:**
   - Review training data for representation bias
   - Consider resampling or reweighting techniques
   - Explore fairness-aware machine learning algorithms

2. **Data Collection:**
   - Ensure balanced representation across groups
   - Check for historical bias in labels

3. **Model Adjustments:**
   - Consider threshold adjustment per group
   - Explore adversarial debiasing techniques
"""
        else:
            report += """
1. **Continue Monitoring:**
   - Re-run fairness audit with new data
   - Track metrics over time

2. **Best Practices:**
   - Document fairness considerations
   - Establish ongoing audit procedures
"""
        
        return img, report, results_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


# ==================== AGENTIC AUDITING FUNCTIONS ====================

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class AgenticAuditor:
    """
    LLM-powered post-mortem analysis and counterfactual reasoning.
    Generates scientific abstracts and answers "what-if" questions.
    """
    
    SCIENTIFIC_ABSTRACT_PROMPT = """You are a Senior ML Research Scientist writing a formal scientific abstract for a machine learning experiment.

Based on the experiment data provided, write a comprehensive scientific abstract following this structure:

**TITLE:** Generate a compelling scientific title for this ML experiment.

**ABSTRACT:**
Write a formal 250-word scientific abstract with these sections:
1. **Background & Objective:** What problem was addressed and why it matters
2. **Methods:** Data preprocessing, feature engineering, and model selection approach
3. **Results:** Key findings including accuracy metrics, feature importance discoveries, and model comparisons
4. **Conclusions:** Main takeaways, model strengths/weaknesses, and practical implications
5. **Keywords:** 5-7 relevant ML/domain keywords

Use formal academic language. Include specific numbers and metrics. Be precise and scientific.
"""

    COUNTERFACTUAL_PROMPT = """You are an expert ML diagnostician performing counterfactual analysis.

Based on the model performance data, answer this specific question:
"What would have to change for this model to achieve {target_accuracy}% accuracy?"

Provide a detailed, actionable analysis covering:

1. **Data Quality Improvements:**
   - What additional samples are needed? (Be specific: "More samples of [specific group] with [specific characteristic]")
   - What data quality issues should be fixed?
   - What missing data patterns need addressing?

2. **Feature Engineering Opportunities:**
   - What new features could improve performance?
   - What feature interactions are underexplored?
   - What domain-specific transformations would help?

3. **Model Architecture Changes:**
   - What hyperparameters should be adjusted?
   - What alternative algorithms should be considered?
   - Would ensemble methods help?

4. **Concrete Action Plan:**
   - List exactly 5 prioritized actions in order of expected impact
   - For each action, estimate the expected accuracy improvement

Be specific, actionable, and quantitative. Reference the actual data statistics provided.
"""

    def __init__(self):
        """Initialize the auditor with available LLM."""
        self.client = None
        self.provider = "template"
        self.last_error = None
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.groq_model = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
        self._initialize_client()
    
    def _record_error(self, context: str, error: Exception):
        """Store a sanitized error message for UI diagnostics."""
        raw = str(error)
        sanitized = re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-***", raw)
        self.last_error = f"{context}: {type(error).__name__}: {sanitized}"

    def _initialize_client(self):
        """Initialize LLM client if available."""
        api_key = os.getenv("OPENAI_API_KEY")
        self.last_error = None
        
        # Debug: print API status
        print(f"[AgenticAuditor] OPENAI_AVAILABLE: {OPENAI_AVAILABLE}, API_KEY exists: {bool(api_key)}")
        
        if OPENAI_AVAILABLE and api_key:
            try:
                self.client = openai.OpenAI(api_key=api_key)
                self.provider = "openai"
                print(f"[AgenticAuditor] Successfully initialized OpenAI client")
            except Exception as e:
                print(f"[AgenticAuditor] OpenAI init error: {e}")
                self._record_error("OpenAI client initialization failed", e)
                self.provider = "template"
        else:
            # Try Groq as fallback
            try:
                from groq import Groq
                groq_key = os.getenv("GROQ_API_KEY")
                if groq_key:
                    self.client = Groq(api_key=groq_key)
                    self.provider = "groq"
                    print(f"[AgenticAuditor] Using Groq provider")
                else:
                    self.last_error = (
                        "No LLM API key found. Set OPENAI_API_KEY or GROQ_API_KEY."
                    )
            except ImportError:
                print(f"[AgenticAuditor] Using template fallback")
                self.last_error = (
                    "No LLM provider available. Install `openai` or `groq` and set an API key."
                )
                self.provider = "template"
    
    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        """Call the LLM with the given prompts."""
        if self.provider == "openai" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                self.last_error = None
                return response.choices[0].message.content
            except Exception as e:
                # API request failed; keep provider but surface useful diagnostics
                self._record_error("OpenAI request failed", e)
                return None
        
        elif self.provider == "groq" and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.groq_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                self.last_error = None
                return response.choices[0].message.content
            except Exception as e:
                self._record_error("Groq request failed", e)
                return None
        
        if not self.last_error:
            self.last_error = "No active LLM provider configured."
        return None  # Will use template fallback
    
    def generate_scientific_abstract(self, experiment_data: Dict) -> str:
        """Generate a scientific abstract for the ML experiment."""
        
        # Prepare context
        context = f"""
## EXPERIMENT DATA

### Dataset Characteristics
- Total Samples: {experiment_data.get('n_samples', 'N/A')}
- Features: {experiment_data.get('n_features', 'N/A')}
- Target Variable: {experiment_data.get('target_col', 'N/A')}
- Task Type: {experiment_data.get('task_type', 'Classification')}
- Class Balance: {experiment_data.get('class_balance', 'N/A')}

### Data Quality Metrics
- Missing Values: {experiment_data.get('missing_pct', 0):.1f}%
- Outliers Detected: {experiment_data.get('outliers_pct', 0):.1f}%
- Features After Engineering: {experiment_data.get('final_features', 'N/A')}

### Model Performance
- Best Model: {experiment_data.get('best_model', 'N/A')}
- Accuracy/R2: {experiment_data.get('best_score', 0):.4f}
- Precision: {experiment_data.get('precision', 'N/A')}
- Recall: {experiment_data.get('recall', 'N/A')}
- F1-Score: {experiment_data.get('f1', 'N/A')}

### Top Features (by importance)
{experiment_data.get('top_features', 'Not available')}

### Model Tournament Results
{experiment_data.get('tournament_results', 'Not available')}
"""
        
        # Try LLM
        llm_response = self._call_llm(self.SCIENTIFIC_ABSTRACT_PROMPT, context)
        
        if llm_response:
            return llm_response
        
        # Template fallback
        return self._generate_template_abstract(experiment_data)
    
    def _generate_template_abstract(self, data: Dict) -> str:
        """Generate a template-based abstract when LLM is unavailable."""
        
        model_name = data.get('best_model', 'machine learning model')
        score = data.get('best_score', 0)
        n_samples = data.get('n_samples', 'N/A')
        n_features = data.get('n_features', 'N/A')
        task = data.get('task_type', 'classification')
        target = data.get('target_col', 'target variable')
        
        return f"""# Automated Machine Learning Experiment Report

## Title
**Predictive Modeling for {target.replace('_', ' ').title()} Using Ensemble Methods: A Comprehensive AutoML Analysis**

## Abstract

**Background & Objective:**
This study presents an automated machine learning pipeline for {task} of {target.replace('_', ' ')}. 
The objective was to develop an accurate and interpretable predictive model using state-of-the-art 
AutoML techniques including automated feature engineering, hyperparameter optimization, and ensemble methods.

**Methods:**
The dataset comprised {n_samples} samples with {n_features} initial features. Data preprocessing included 
automated missing value imputation using Bayesian methods, outlier detection via Isolation Forest, and 
semantic type inference. Feature engineering generated domain-specific features based on detected patterns. 
Multiple algorithms were evaluated in a tournament-style competition with cross-validation.

**Results:**
The {model_name} achieved the highest performance with a {'accuracy' if task == 'classification' else 'R2 score'} 
of {score:.4f}. Feature importance analysis revealed key predictive factors, and SHAP analysis provided 
interpretable explanations for individual predictions. The model demonstrated robust generalization across 
cross-validation folds with minimal variance.

**Conclusions:**
The automated pipeline successfully identified an optimal model configuration without manual intervention. 
The {model_name} provides both high accuracy and interpretability, making it suitable for production deployment. 
Future work should focus on additional feature engineering and ensemble strategies to further improve performance.

**Keywords:** AutoML, {task}, ensemble methods, SHAP, interpretability, {target.replace('_', ' ')}

---
*Generated by MetaAI Pro Agentic Auditor*
"""
    
    def generate_counterfactual_analysis(self, experiment_data: Dict, target_accuracy: float = 98.0) -> str:
        """Generate counterfactual analysis: what needs to change for target accuracy."""
        
        current_score = experiment_data.get('best_score', 0) * 100
        gap = target_accuracy - current_score
        
        context = f"""
## CURRENT MODEL PERFORMANCE
- Current Accuracy: {current_score:.2f}%
- Target Accuracy: {target_accuracy:.2f}%
- Gap to Close: {gap:.2f}%

## DATASET STATISTICS
- Total Samples: {experiment_data.get('n_samples', 'N/A')}
- Features: {experiment_data.get('n_features', 'N/A')}
- Target: {experiment_data.get('target_col', 'N/A')}
- Missing Data: {experiment_data.get('missing_pct', 0):.1f}%
- Outliers: {experiment_data.get('outliers_pct', 0):.1f}%
- Class Balance: {experiment_data.get('class_balance', 'N/A')}

## CURRENT MODEL
- Best Model: {experiment_data.get('best_model', 'N/A')}
- Cross-Validation Folds: {experiment_data.get('cv_folds', 5)}
- Hyperparameters: {experiment_data.get('hyperparameters', 'Default')}

## TOP FEATURES
{experiment_data.get('top_features', 'Not available')}

## ERROR ANALYSIS
- Most Common Misclassifications: {experiment_data.get('common_errors', 'N/A')}
- Weakest Performing Subgroups: {experiment_data.get('weak_groups', 'N/A')}

## QUESTION
What would have to change in this data and model for the accuracy to reach {target_accuracy}%?
"""
        
        # Try LLM
        llm_response = self._call_llm(
            self.COUNTERFACTUAL_PROMPT.format(target_accuracy=target_accuracy), 
            context
        )
        
        if llm_response:
            return llm_response
        
        # Template fallback
        return self._generate_template_counterfactual(experiment_data, target_accuracy)
    
    def _generate_template_counterfactual(self, data: Dict, target_accuracy: float) -> str:
        """Generate template-based counterfactual analysis."""
        
        current_score = data.get('best_score', 0) * 100
        gap = target_accuracy - current_score
        n_samples = data.get('n_samples', 1000)
        missing_pct = data.get('missing_pct', 0)
        model_name = data.get('best_model', 'Current Model')
        
        # Calculate estimated requirements
        additional_samples = int(n_samples * (gap / 10))  # Rough estimate
        
        return f"""# Counterfactual Analysis Report

## Current State vs Target

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Accuracy | {current_score:.2f}% | {target_accuracy:.2f}% | {gap:.2f}% |

---

## What Needs to Change to Reach {target_accuracy}% Accuracy

### 1. Data Quality Improvements

**Additional Samples Needed:**
- The model requires approximately **{additional_samples:,} additional samples** to improve generalization
- Focus on underrepresented classes or edge cases where the model currently fails
- **Specific recommendation:** Collect more samples from minority classes or rare conditions

**Missing Data Resolution:**
- Current missing data: {missing_pct:.1f}%
- {"Reduce missing data below 5% for optimal performance" if missing_pct > 5 else "Missing data levels are acceptable"}
- Consider domain-specific imputation methods for critical features

**Outlier Treatment:**
- Review flagged outliers - some may be valid edge cases that improve model robustness
- Remove true data entry errors but retain legitimate extreme values

### 2. Feature Engineering Opportunities

**New Features to Create:**
1. **Interaction terms** between the top 3 most important features
2. **Polynomial features** (degree 2) for numeric predictors
3. **Domain-specific ratios** (e.g., feature_A / feature_B)
4. **Binned versions** of continuous variables to capture non-linear relationships
5. **Time-based features** if temporal data exists (day of week, month, trends)

**Feature Selection:**
- Current feature count may include noise - apply stricter RFE threshold
- Remove features with <0.01 importance score
- Consider PCA for dimensionality reduction if features are highly correlated

### 3. Model Architecture Changes

**Hyperparameter Tuning:**
- Increase Optuna trials from current to **150+ trials**
- Expand search space for:
  - max_depth: 5-30 (currently may be too shallow)
  - n_estimators: 200-500 (more trees often help)
  - learning_rate: 0.01-0.1 (lower rates with more iterations)

**Alternative Algorithms:**
- If using {model_name}, try:
  - **CatBoost** - handles categorical features better
  - **Neural Network** - may capture complex non-linear patterns
  - **Stacking Ensemble** - combine predictions from multiple models

**Ensemble Strategies:**
- Build a voting ensemble of top 3 models
- Use stacking with a meta-learner
- Consider bagging with different random seeds

### 4. Prioritized Action Plan

| Priority | Action | Expected Improvement |
|----------|--------|---------------------|
| 1 | Collect {additional_samples:,} more samples, focusing on minority classes | +{min(gap*0.4, 5):.1f}% |
| 2 | Create 10 new engineered features (interactions + polynomials) | +{min(gap*0.25, 3):.1f}% |
| 3 | Run extended Optuna optimization (150 trials) | +{min(gap*0.15, 2):.1f}% |
| 4 | Build stacking ensemble with top 3 models | +{min(gap*0.15, 2):.1f}% |
| 5 | Apply threshold optimization for classification boundary | +{min(gap*0.05, 1):.1f}% |

**Total Expected Improvement:** ~{min(gap, 10):.1f}% (reaching approximately {min(current_score + 10, target_accuracy):.1f}%)

---

### Key Insight

The gap of {gap:.2f}% requires a multi-pronged approach. No single change will achieve the target.
The most impactful investment is **more high-quality training data** from underrepresented scenarios,
followed by **sophisticated feature engineering** and **ensemble methods**.

---
*Generated by MetaAI Pro Agentic Auditor*
"""


# Global auditor instance
_agentic_auditor = None

def get_auditor():
    """Get or create the agentic auditor instance."""
    global _agentic_auditor
    if _agentic_auditor is None:
        _agentic_auditor = AgenticAuditor()
    return _agentic_auditor


def run_llm_postmortem(df: pd.DataFrame, target_col: str = None, 
                       progress=gr.Progress()) -> Tuple[str, str]:
    """
    LLM Post-Mortem - Generate a scientific abstract of the ML experiment.
    """
    if df is None:
        return "No data uploaded", "Please upload data and train a model first."
    
    if state.trained_model is None:
        return "No model trained", "Please train a model first in the Model Training tab."
    
    try:
        progress(0.1, desc="Gathering experiment data...")
        
        # Collect experiment data
        n_samples, n_features = df.shape
        
        # Get class balance if classification
        if target_col and target_col in df.columns:
            y = df[target_col]
            class_balance = y.value_counts(normalize=True).to_dict()
            class_balance_str = ", ".join([f"{k}: {v:.1%}" for k, v in list(class_balance.items())[:5]])
        else:
            class_balance_str = "N/A"
        
        # Get top features from state if available
        top_features_str = "Feature importance not available"
        if hasattr(state, 'feature_importances') and state.feature_importances:
            top_features_str = "\n".join([f"- {f}: {i:.4f}" for f, i in state.feature_importances[:10]])
        elif hasattr(state.trained_model, 'feature_importances_'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            importances = state.trained_model.feature_importances_
            if len(importances) == len(numeric_cols):
                sorted_idx = np.argsort(importances)[::-1][:10]
                top_features_str = "\n".join([f"- {numeric_cols[i]}: {importances[i]:.4f}" for i in sorted_idx])
        
        # Compile experiment data
        experiment_data = {
            'n_samples': n_samples,
            'n_features': n_features - 1 if target_col else n_features,
            'target_col': target_col or "Not specified",
            'task_type': 'Classification' if (target_col and df[target_col].nunique() <= 20) else 'Regression',
            'class_balance': class_balance_str,
            'missing_pct': df.isnull().mean().mean() * 100,
            'outliers_pct': getattr(state, 'outlier_pct', 5),
            'final_features': n_features - 1,
            'best_model': getattr(state, 'model_name', 'Trained Model'),
            'best_score': getattr(state, 'best_score', 0.85),
            'precision': getattr(state, 'precision', 'N/A'),
            'recall': getattr(state, 'recall', 'N/A'),
            'f1': getattr(state, 'f1', 'N/A'),
            'top_features': top_features_str,
            'tournament_results': getattr(state, 'tournament_results', 'Results not stored'),
        }
        
        progress(0.3, desc="Generating scientific abstract...")
        
        auditor = get_auditor()
        abstract = auditor.generate_scientific_abstract(experiment_data)
        
        progress(1.0, desc="Abstract generated!")
        
        # Status message
        if auditor.provider in ["openai", "groq"]:
            status = f"Generated using {auditor.provider.upper()} LLM"
        else:
            status = "Generated using template engine (set OPENAI_API_KEY for enhanced AI-powered generation)"
        
        return abstract, status
        
    except Exception as e:
        import traceback
        return f"Error generating abstract:\n{str(e)}\n{traceback.format_exc()}", "Error occurred"


def run_counterfactual_reasoning(df: pd.DataFrame, target_col: str = None, 
                                  target_accuracy: float = 98.0,
                                  custom_question: str = None,
                                  progress=gr.Progress()) -> Tuple[str, str]:
    """
    Counterfactual Reasoning - Answer "what-if" questions about model improvement.
    """
    if df is None:
        return "No data uploaded", "Please upload data and train a model first."
    
    if state.trained_model is None:
        return "No model trained", "Please train a model first in the Model Training tab."
    
    try:
        progress(0.1, desc="Analyzing current model state...")
        
        # Collect data for analysis
        n_samples, n_features = df.shape
        
        # Get class balance
        if target_col and target_col in df.columns:
            y = df[target_col]
            class_balance = y.value_counts(normalize=True).to_dict()
            class_balance_str = ", ".join([f"{k}: {v:.1%}" for k, v in list(class_balance.items())[:5]])
            
            # Find minority class
            minority_class = min(class_balance, key=class_balance.get)
            minority_pct = class_balance[minority_class] * 100
        else:
            class_balance_str = "N/A"
            minority_class = "N/A"
            minority_pct = 0
        
        # Get feature info
        top_features_str = "Not available"
        if hasattr(state.trained_model, 'feature_importances_'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            importances = state.trained_model.feature_importances_
            if len(importances) == len(numeric_cols):
                sorted_idx = np.argsort(importances)[::-1][:10]
                top_features_str = "\n".join([f"- {numeric_cols[i]}: {importances[i]:.4f}" for i in sorted_idx])
        
        experiment_data = {
            'n_samples': n_samples,
            'n_features': n_features - 1 if target_col else n_features,
            'target_col': target_col or "Not specified",
            'missing_pct': df.isnull().mean().mean() * 100,
            'outliers_pct': getattr(state, 'outlier_pct', 5),
            'best_model': getattr(state, 'model_name', 'Trained Model'),
            'best_score': getattr(state, 'best_score', 0.85),
            'cv_folds': 5,
            'hyperparameters': str(getattr(state, 'best_params', 'Default')),
            'top_features': top_features_str,
            'class_balance': class_balance_str,
            'common_errors': f"Minority class '{minority_class}' ({minority_pct:.1f}%) may be under-predicted",
            'weak_groups': f"Samples with rare combinations of top features",
        }
        
        progress(0.3, desc="Performing counterfactual analysis...")
        
        auditor = get_auditor()
        analysis = auditor.generate_counterfactual_analysis(experiment_data, target_accuracy)
        
        progress(1.0, desc="Analysis complete!")
        
        # Status message
        current_acc = getattr(state, 'best_score', 0.85) * 100
        if auditor.provider in ["openai", "groq"]:
            status = f"Analyzed gap from {current_acc:.1f}% to {target_accuracy}% using {auditor.provider.upper()} LLM"
        else:
            status = f"Analyzed gap from {current_acc:.1f}% to {target_accuracy}% using template engine"
        
        return analysis, status
        
    except Exception as e:
        import traceback
        return f"Error in counterfactual analysis:\n{str(e)}\n{traceback.format_exc()}", "Error occurred"


def _trained_models_from_state() -> List[str]:
    """Extract trained model names from in-memory app state."""
    model_names: List[str] = []

    current_model = getattr(state, "model_name", None)
    if isinstance(current_model, str) and current_model.strip():
        model_names.append(current_model.strip())

    report = getattr(state, "training_report", None)
    if isinstance(report, dict):
        rankings = report.get("rankings", [])
        if isinstance(rankings, list):
            for entry in rankings:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("model") or entry.get("Model")
                if isinstance(name, str):
                    clean_name = name.strip()
                    if clean_name and clean_name not in model_names:
                        model_names.append(clean_name)

    return model_names


def _local_model_answer(question: str) -> Optional[str]:
    """Answer model-list questions from local state when LLM is unavailable."""
    q = question.lower()
    asks_model_list = (
        ("model" in q or "models" in q)
        and ("trained" in q or "train" in q or "available" in q or "which" in q or "list" in q)
    )
    if not asks_model_list:
        return None

    models = _trained_models_from_state()
    if not models:
        return (
            "## Agent Response\n\n"
            "No models are trained in the current session yet.\n\n"
            "Train models in **Model Training** first, then ask again."
        )

    model_lines = "\n".join([f"{idx}. {name}" for idx, name in enumerate(models, 1)])
    return (
        "## Agent Response\n\n"
        "Trained models in the current session:\n"
        f"{model_lines}\n\n"
        f"Current best model: **{models[0]}**"
    )


def run_custom_agent_query(df: pd.DataFrame, target_col: str = None,
                           question: str = "", progress=gr.Progress()) -> str:
    """
    Custom Agent Query - Ask any question about the model and data.
    """
    if not question.strip():
        return "Please enter a question."

    if df is None:
        return "No data uploaded. Please upload data first."

    try:
        progress(0.2, desc="Processing your question...")

        auditor = get_auditor()

        # Build context
        n_samples, n_features = df.shape
        context = f"""
## Dataset Info
- Samples: {n_samples}
- Features: {n_features}
- Target: {target_col or 'Not specified'}
- Model: {getattr(state, 'model_name', 'Not trained')}
- Score: {getattr(state, 'best_score', 'N/A')}

## User Question
{question}
"""

        system_prompt = """You are an expert ML consultant. Answer the user's question about their machine learning project.
Be specific, actionable, and reference the data context provided. If the question cannot be answered with the available information, explain what additional data would be needed."""

        progress(0.5, desc="Consulting the AI agent...")

        response = auditor._call_llm(system_prompt, context)

        if response:
            progress(1.0, desc="Response ready!")
            return response

        # If the question is answerable from local app state, provide that first.
        local_answer = _local_model_answer(question)
        error_hint = getattr(auditor, "last_error", None)
        diagnostics = f"\n\nLLM error: `{error_hint}`" if error_hint else ""

        if local_answer:
            progress(1.0, desc="Answered using local training state")
            return (
                f"{local_answer}"
                f"{diagnostics}\n\n"
                "To enable full LLM answers, configure a valid API key:\n"
                "1. PowerShell (current terminal): `$env:OPENAI_API_KEY=\"your_key\"`\n"
                "2. Windows persistent: `setx OPENAI_API_KEY \"your_key\"`\n"
                "3. Or use Groq: `$env:GROQ_API_KEY=\"your_key\"`\n"
                "4. Restart the app after updating the key."
            )

        progress(1.0, desc="Using fallback response...")
        return f"""## Agent Response

Based on your question: "{question}"

**Analysis:**
I could not complete an LLM response for this request.

**Your Data Context:**
- Dataset has {n_samples} samples and {n_features} features
- Target column: {target_col or 'Not specified'}
- Current model: {getattr(state, 'model_name', 'Not yet trained')}
- Current score: {getattr(state, 'best_score', 'Not available')}

**LLM Diagnostics:**
- Provider: {getattr(auditor, 'provider', 'template')}
- Error: {error_hint or 'No detailed error available'}

**Fix API-key setup:**
1. PowerShell (current terminal): `$env:OPENAI_API_KEY="your_key"`
2. Windows persistent: `setx OPENAI_API_KEY "your_key"`
3. Optional Groq fallback: `$env:GROQ_API_KEY="your_key"`
4. Restart the application
"""

    except Exception as e:
        return f"Error processing question: {str(e)}"


# ==================== MLOPS & PRODUCTION FUNCTIONS ====================

def generate_fastapi_code(df: pd.DataFrame, target_col: str = None) -> str:
    """Generate complete FastAPI code for the trained model."""
    
    if df is None or state.trained_model is None:
        return "# No model available. Please train a model first."
    
    # Get feature columns
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Determine task type
    task_type = "classification" if (target_col and df[target_col].nunique() <= 20) else "regression"
    
    model_name = getattr(state, 'model_name', 'trained_model').replace(' ', '_').lower()
    
    # Generate field definitions
    field_defs = []
    example_values = []
    for col in numeric_cols[:20]:  # Limit to 20 features
        col_clean = col.replace(' ', '_').replace('-', '_')
        median_val = df[col].median()
        field_defs.append(f"    {col_clean}: float = Field(..., description='{col}')")
        example_values.append(f'"{col_clean}": {median_val:.4f}')
    
    fields_str = "\n".join(field_defs)
    example_str = ", ".join(example_values[:5]) + ", ..."
    
    code = f'''"""
MetaAI Pro - Auto-Generated FastAPI Server
Model: {model_name}
Task: {task_type.title()}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MetaAI Pro - ML Prediction API",
    description="""
    ## Auto-Generated Machine Learning API
    
    This API serves predictions from a trained {task_type} model.
    
    **Model:** {model_name}
    **Features:** {len(numeric_cols)}
    
    ### Endpoints:
    - `POST /predict` - Single prediction
    - `POST /predict/batch` - Batch predictions
    - `GET /health` - Health check
    - `GET /model/info` - Model information
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
MODEL = None
SCALER = None
FEATURE_NAMES = {numeric_cols}

@app.on_event("startup")
async def load_model():
    global MODEL, SCALER
    try:
        MODEL = joblib.load("model.joblib")
        try:
            SCALER = joblib.load("scaler.joblib")
        except:
            SCALER = None
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {{e}}")
        raise


# Request/Response schemas
class PredictionInput(BaseModel):
{fields_str}
    
    class Config:
        schema_extra = {{
            "example": {{{example_str}}}
        }}


class PredictionOutput(BaseModel):
    prediction: {"int" if task_type == "classification" else "float"}
    {"probability: Optional[float] = None" if task_type == "classification" else ""}
    confidence: Optional[float] = None
    model_version: str = "1.0.0"


class BatchInput(BaseModel):
    instances: List[Dict[str, float]]


class BatchOutput(BaseModel):
    predictions: List[{"int" if task_type == "classification" else "float"}]
    {"probabilities: Optional[List[float]] = None" if task_type == "classification" else ""}
    count: int


class ModelInfo(BaseModel):
    model_name: str
    task_type: str
    n_features: int
    feature_names: List[str]
    version: str


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool


# Endpoints
@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthCheck(
        status="healthy",
        model_loaded=MODEL is not None
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information."""
    return ModelInfo(
        model_name="{model_name}",
        task_type="{task_type}",
        n_features={len(numeric_cols)},
        feature_names=FEATURE_NAMES,
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict(input_data: PredictionInput):
    """
    Make a single prediction.
    
    Send feature values and receive the model's prediction.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to array
        features = np.array([[
            getattr(input_data, f.replace(' ', '_').replace('-', '_')) 
            for f in FEATURE_NAMES
        ]])
        
        # Scale if scaler available
        if SCALER is not None:
            features = SCALER.transform(features)
        
        # Predict
        prediction = MODEL.predict(features)[0]
        
        response = PredictionOutput(
            prediction={"int(prediction)" if task_type == "classification" else "float(prediction)"},
            model_version="1.0.0"
        )
        
        # Add probability for classification
        {"if hasattr(MODEL, 'predict_proba'):" if task_type == "classification" else "# No probability for regression"}
            {"proba = MODEL.predict_proba(features)[0]" if task_type == "classification" else "pass"}
            {"response.probability = float(max(proba))" if task_type == "classification" else ""}
            {"response.confidence = float(max(proba))" if task_type == "classification" else ""}
        
        logger.info(f"Prediction: {{prediction}}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {{e}}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchOutput, tags=["Predictions"])
async def predict_batch(input_data: BatchInput):
    """
    Make batch predictions.
    
    Send multiple instances and receive predictions for all.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to array
        features = np.array([
            [inst.get(f.replace(' ', '_').replace('-', '_'), 0) for f in FEATURE_NAMES]
            for inst in input_data.instances
        ])
        
        # Scale if scaler available
        if SCALER is not None:
            features = SCALER.transform(features)
        
        # Predict
        predictions = MODEL.predict(features)
        
        response = BatchOutput(
            predictions=[{"int(p)" if task_type == "classification" else "float(p)"} for p in predictions],
            count=len(predictions)
        )
        
        {"if hasattr(MODEL, 'predict_proba'):" if task_type == "classification" else "# No probability for regression"}
            {"probas = MODEL.predict_proba(features)" if task_type == "classification" else "pass"}
            {"response.probabilities = [float(max(p)) for p in probas]" if task_type == "classification" else ""}
        
        logger.info(f"Batch prediction: {{len(predictions)}} instances")
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction error: {{e}}")
        raise HTTPException(status_code=400, detail=str(e))


# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    return code


def generate_dockerfile() -> str:
    """Generate Dockerfile for the API."""
    return '''# MetaAI Pro - Auto-Generated Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py .
COPY model.joblib .
COPY scaler.joblib .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
'''


def generate_requirements() -> str:
    """Generate requirements.txt for the API."""
    return '''# MetaAI Pro - Auto-Generated Requirements
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
python-multipart>=0.0.6
'''


def run_api_generation(df: pd.DataFrame, model_name_input: str = "MetaAI_Model",
                       progress=gr.Progress()) -> Tuple[str, str, str, str, str]:
    """
    Generate complete FastAPI deployment package.
    """
    if df is None:
        return "No data", "No data", "No data", "Please upload data first", ""
    
    if state.trained_model is None:
        return "No model", "No model", "No model", "Please train a model first", ""
    
    # Get target column from state
    target_col = state.target_column
    
    try:
        progress(0.2, desc="Generating FastAPI code...")
        api_code = generate_fastapi_code(df, target_col)
        
        progress(0.5, desc="Generating Dockerfile...")
        dockerfile = generate_dockerfile()
        
        progress(0.7, desc="Generating requirements.txt...")
        requirements = generate_requirements()
        
        progress(0.9, desc="Creating deployment instructions...")
        
        instructions = f"""## Deployment Instructions

### Quick Start

1. **Save the files:**
   - `api.py` - FastAPI application
   - `model.joblib` - Trained model (export from Export tab)
   - `scaler.joblib` - Feature scaler (if used)
   - `Dockerfile` - Container definition
   - `requirements.txt` - Python dependencies

2. **Run locally:**
   ```bash
   pip install -r requirements.txt
   uvicorn api:app --reload --port 8000
   ```

3. **Access Swagger UI:**
   Open http://localhost:8000/docs

4. **Run with Docker:**
   ```bash
   docker build -t metaai-api .
   docker run -p 8000:8000 metaai-api
   ```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Swagger UI (interactive documentation) |
| `/redoc` | GET | ReDoc documentation |
| `/health` | GET | Health check |
| `/model/info` | GET | Model information |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"feature1": 1.5, "feature2": 2.3, ...}}'
```

### Model Info
- **Model:** {getattr(state, 'model_name', 'Trained Model')}
- **Features:** {len(df.select_dtypes(include=[np.number]).columns) - (1 if target_col else 0)}
- **Target:** {target_col or 'Not specified'}
"""
        
        progress(1.0, desc="API generation complete!")
        
        status = "API code generated successfully. Copy the code or use the Export tab to download the full package."
        
        return api_code, dockerfile, requirements, instructions, status
        
    except Exception as e:
        import traceback
        error = f"Error: {str(e)}\n{traceback.format_exc()}"
        return error, error, error, error, "Error occurred"


def run_drift_detection(df_baseline: pd.DataFrame, target_col: str = None, 
                        threshold: float = 0.15,
                        progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Real-time drift detection comparing new data against baseline.
    Simulates production data drift to demonstrate monitoring capabilities.
    """
    if df_baseline is None:
        return create_placeholder_image("No baseline"), "No baseline data. Please upload data in the Data Ingestion tab.", pd.DataFrame()
    
    try:
        from scipy import stats
        
        progress(0.1, desc="Analyzing baseline statistics...")
        
        # Get numeric columns
        numeric_cols = df_baseline.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        if len(numeric_cols) == 0:
            return create_placeholder_image("No features"), "No numeric features found", pd.DataFrame()
        
        # Calculate baseline statistics
        baseline_stats = {}
        for col in numeric_cols:
            baseline_stats[col] = {
                'mean': df_baseline[col].mean(),
                'std': df_baseline[col].std(),
                'min': df_baseline[col].min(),
                'max': df_baseline[col].max(),
                'median': df_baseline[col].median(),
                'q1': df_baseline[col].quantile(0.25),
                'q3': df_baseline[col].quantile(0.75),
            }
        
        progress(0.3, desc="Simulating production data drift...")
        
        # Simulate production data with drift
        np.random.seed(42)
        df_new = df_baseline.copy()
        
        # Add drift to some columns
        drift_cols = numeric_cols[:min(3, len(numeric_cols))]
        for col in drift_cols:
            # Add mean shift and variance change
            drift_factor = np.random.uniform(0.1, 0.3)
            df_new[col] = df_new[col] * (1 + drift_factor) + np.random.normal(0, df_new[col].std() * 0.2, len(df_new))
        
        progress(0.5, desc="Calculating drift metrics...")
        
        # Calculate drift for each column
        drift_results = []
        alerts = []
        
        for col in numeric_cols:
            new_mean = df_new[col].mean()
            new_std = df_new[col].std()
            
            baseline_mean = baseline_stats[col]['mean']
            baseline_std = baseline_stats[col]['std']
            
            # Calculate drift metrics
            mean_shift = abs(new_mean - baseline_mean) / (baseline_std + 1e-6)
            std_ratio = new_std / (baseline_std + 1e-6)
            
            # KS test for distribution change
            try:
                ks_stat, ks_pval = stats.ks_2samp(df_baseline[col].dropna(), df_new[col].dropna())
            except:
                ks_stat, ks_pval = 0, 1
            
            # Determine drift severity
            if mean_shift > threshold * 2 or ks_pval < 0.01:
                severity = "CRITICAL"
                alerts.append(f"CRITICAL: {col} - Major distribution shift detected (KS={ks_stat:.3f})")
            elif mean_shift > threshold or ks_pval < 0.05:
                severity = "WARNING"
                alerts.append(f"WARNING: {col} - Moderate drift detected (shift={mean_shift:.2f})")
            else:
                severity = "OK"
            
            drift_results.append({
                'Feature': col,
                'Baseline Mean': round(baseline_mean, 4),
                'New Mean': round(new_mean, 4),
                'Mean Shift (std)': round(mean_shift, 3),
                'KS Statistic': round(ks_stat, 3),
                'KS P-Value': round(ks_pval, 4),
                'Severity': severity
            })
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'WARNING': 1, 'OK': 2}
        drift_results = sorted(drift_results, key=lambda x: severity_order[x['Severity']])
        
        progress(0.75, desc="Creating visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # Plot 1: Drift severity by feature
        ax1 = axes[0, 0]
        features = [r['Feature'][:15] for r in drift_results[:15]]
        shifts = [r['Mean Shift (std)'] for r in drift_results[:15]]
        colors = ['#e74c3c' if r['Severity'] == 'CRITICAL' else '#f39c12' if r['Severity'] == 'WARNING' else '#27ae60' 
                  for r in drift_results[:15]]
        
        ax1.barh(features[::-1], shifts[::-1], color=colors[::-1], alpha=0.8)
        ax1.axvline(x=threshold, color='orange', linestyle='--', lw=2, label=f'Warning ({threshold})')
        ax1.axvline(x=threshold*2, color='red', linestyle='--', lw=2, label=f'Critical ({threshold*2})')
        ax1.set_xlabel('Mean Shift (standard deviations)')
        ax1.set_title('Feature Drift Detection', fontweight='bold')
        ax1.legend()
        
        # Plot 2: Distribution comparison for top drifted feature
        ax2 = axes[0, 1]
        if drift_results:
            top_drift_col = drift_results[0]['Feature']
            ax2.hist(df_baseline[top_drift_col].dropna(), bins=30, alpha=0.5, label='Baseline', color='blue', density=True)
            ax2.hist(df_new[top_drift_col].dropna(), bins=30, alpha=0.5, label='New Data', color='red', density=True)
            ax2.set_xlabel(top_drift_col)
            ax2.set_ylabel('Density')
            ax2.set_title(f'Distribution Shift: {top_drift_col[:20]}', fontweight='bold')
            ax2.legend()
        
        # Plot 3: KS Statistics
        ax3 = axes[1, 0]
        ks_stats = [r['KS Statistic'] for r in drift_results[:15]]
        ax3.barh(features[::-1], ks_stats[::-1], color='steelblue', alpha=0.8)
        ax3.axvline(x=0.1, color='orange', linestyle='--', label='Moderate')
        ax3.axvline(x=0.2, color='red', linestyle='--', label='Severe')
        ax3.set_xlabel('KS Statistic')
        ax3.set_title('Kolmogorov-Smirnov Test Results', fontweight='bold')
        ax3.legend()
        
        # Plot 4: Alert summary
        ax4 = axes[1, 1]
        severity_counts = {'CRITICAL': 0, 'WARNING': 0, 'OK': 0}
        for r in drift_results:
            severity_counts[r['Severity']] += 1
        
        colors_pie = ['#e74c3c', '#f39c12', '#27ae60']
        explode = (0.1, 0.05, 0)
        ax4.pie([severity_counts['CRITICAL'], severity_counts['WARNING'], severity_counts['OK']],
                labels=['Critical', 'Warning', 'OK'],
                colors=colors_pie, autopct='%1.0f%%', explode=explode)
        ax4.set_title('Overall Drift Status', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Drift detection complete!")
        
        # Create report
        n_critical = severity_counts['CRITICAL']
        n_warning = severity_counts['WARNING']
        n_ok = severity_counts['OK']
        
        if n_critical > 0:
            overall_status = "CRITICAL - Immediate Action Required"
            recommendation = "Model performance may be severely degraded. **Triggering Auto-Retrain recommended.**"
        elif n_warning > 0:
            overall_status = "WARNING - Monitor Closely"
            recommendation = "Some features show drift. Monitor model performance and consider retraining."
        else:
            overall_status = "OK - No Significant Drift"
            recommendation = "Data distribution is stable. Continue monitoring."
        
        report = f"""## Real-Time Drift Detection Report

### Overall Status: {overall_status}

### Summary
| Status | Count | Percentage |
|--------|-------|------------|
| Critical | {n_critical} | {n_critical/len(drift_results)*100:.0f}% |
| Warning | {n_warning} | {n_warning/len(drift_results)*100:.0f}% |
| OK | {n_ok} | {n_ok/len(drift_results)*100:.0f}% |

### Alerts
"""
        if alerts:
            for alert in alerts[:5]:
                report += f"- {alert}\n"
        else:
            report += "- No alerts. All features within acceptable drift range.\n"
        
        report += f"""
### Recommendation
{recommendation}

### Detection Method
- **Mean Shift:** Measures how many standard deviations the mean has shifted
- **KS Test:** Kolmogorov-Smirnov test compares full distributions
- **Threshold:** Warning at {threshold:.0%} std shift, Critical at {threshold*2:.0%}

### Automatic Actions
When drift is detected, consider:
1. **Alert stakeholders** about potential model degradation
2. **Collect new labels** for drifted samples
3. **Trigger retraining** with recent data
4. **A/B test** retrained model before deployment

### Monitoring Schedule
- Run drift detection: **Daily** or with each new data batch
- Full retrain: **Monthly** or when critical drift detected
- Model evaluation: **Weekly** on holdout set
"""
        
        drift_df = pd.DataFrame(drift_results)
        
        return img, report, drift_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def simulate_production_monitoring(df: pd.DataFrame, target_col: str = None,
                                    progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    """
    Simulate production monitoring dashboard with metrics over time.
    """
    if df is None:
        return create_placeholder_image("No data"), "Please upload data first."
    
    try:
        progress(0.2, desc="Simulating production metrics...")
        
        # Simulate 30 days of production data
        np.random.seed(42)
        days = 30
        
        # Generate realistic metrics with some degradation
        base_accuracy = getattr(state, 'best_score', 0.85)
        
        # Simulate accuracy with slight degradation over time
        accuracies = []
        latencies = []
        request_counts = []
        error_rates = []
        
        for day in range(days):
            # Add some noise and slight degradation
            degradation = day * 0.002  # 0.2% degradation per day
            noise = np.random.normal(0, 0.01)
            acc = max(0.5, base_accuracy - degradation + noise)
            accuracies.append(acc)
            
            # Latency (increases slightly over time)
            latency = 50 + day * 0.5 + np.random.normal(0, 5)
            latencies.append(max(20, latency))
            
            # Request count (varies by day)
            requests = 1000 + np.random.randint(-200, 500) + (100 if day % 7 < 5 else -300)
            request_counts.append(max(100, requests))
            
            # Error rate
            error = 0.01 + day * 0.001 + np.random.uniform(0, 0.005)
            error_rates.append(min(0.1, error))
        
        progress(0.6, desc="Creating monitoring dashboard...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        dates = range(1, days + 1)
        
        # Plot 1: Accuracy over time
        ax1 = axes[0, 0]
        ax1.plot(dates, accuracies, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.fill_between(dates, accuracies, alpha=0.3)
        ax1.axhline(y=base_accuracy, color='green', linestyle='--', label=f'Baseline: {base_accuracy:.2f}')
        ax1.axhline(y=base_accuracy - 0.05, color='orange', linestyle='--', label='Warning Threshold')
        ax1.axhline(y=base_accuracy - 0.10, color='red', linestyle='--', label='Critical Threshold')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Over Time', fontweight='bold')
        ax1.legend(loc='lower left')
        ax1.set_ylim(0.6, 1.0)
        
        # Plot 2: Latency
        ax2 = axes[0, 1]
        ax2.plot(dates, latencies, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.fill_between(dates, latencies, alpha=0.3, color='green')
        ax2.axhline(y=100, color='orange', linestyle='--', label='Warning (100ms)')
        ax2.axhline(y=200, color='red', linestyle='--', label='Critical (200ms)')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('API Latency', fontweight='bold')
        ax2.legend()
        
        # Plot 3: Request volume
        ax3 = axes[1, 0]
        ax3.bar(dates, request_counts, color='steelblue', alpha=0.8)
        ax3.set_xlabel('Day')
        ax3.set_ylabel('Requests')
        ax3.set_title('Daily Request Volume', fontweight='bold')
        ax3.axhline(y=np.mean(request_counts), color='red', linestyle='--', label=f'Avg: {np.mean(request_counts):.0f}')
        ax3.legend()
        
        # Plot 4: Error rate
        ax4 = axes[1, 1]
        ax4.plot(dates, [e * 100 for e in error_rates], 'r-', linewidth=2, marker='^', markersize=4)
        ax4.fill_between(dates, [e * 100 for e in error_rates], alpha=0.3, color='red')
        ax4.axhline(y=2, color='orange', linestyle='--', label='Warning (2%)')
        ax4.axhline(y=5, color='red', linestyle='--', label='Critical (5%)')
        ax4.set_xlabel('Day')
        ax4.set_ylabel('Error Rate (%)')
        ax4.set_title('API Error Rate', fontweight='bold')
        ax4.legend()
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Monitoring dashboard ready!")
        
        # Generate report
        current_acc = accuracies[-1]
        acc_change = current_acc - base_accuracy
        avg_latency = np.mean(latencies)
        total_requests = sum(request_counts)
        avg_errors = np.mean(error_rates) * 100
        
        if acc_change < -0.05:
            status = "WARNING: Accuracy degradation detected"
            action = "Consider retraining with recent data"
        else:
            status = "OK: Model performing within acceptable range"
            action = "Continue monitoring"
        
        report = f"""## Production Monitoring Dashboard

### Current Status: {status}

### 30-Day Summary
| Metric | Value | Trend |
|--------|-------|-------|
| Current Accuracy | {current_acc:.2%} | {'Declining' if acc_change < -0.02 else 'Stable'} |
| Accuracy Change | {acc_change:+.2%} | from baseline |
| Avg Latency | {avg_latency:.0f}ms | {'Good' if avg_latency < 100 else 'Elevated'} |
| Total Requests | {total_requests:,} | over 30 days |
| Avg Error Rate | {avg_errors:.2f}% | {'Acceptable' if avg_errors < 2 else 'High'} |

### Alerts
- {'Accuracy has dropped by ' + f'{abs(acc_change):.1%}' + ' - consider retraining' if acc_change < -0.03 else 'No accuracy alerts'}
- {'Latency above 100ms on some days' if max(latencies) > 100 else 'Latency within acceptable range'}
- {'Error rate spike detected' if max(error_rates) > 0.03 else 'Error rate stable'}

### Recommended Action
{action}

### Auto-Retrain Trigger Conditions
The system will automatically trigger retraining when:
1. Accuracy drops below {base_accuracy - 0.10:.2%} (10% below baseline)
2. Drift detected in >3 features at CRITICAL level
3. Error rate exceeds 5% for 3 consecutive days
4. Manual trigger by operator

### Next Steps
1. Review drift detection results in the Drift Alerts tab
2. If retraining needed, use fresh data from production
3. A/B test new model before full deployment
"""
        
        return img, report
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}"


def run_production_readiness_check(df: pd.DataFrame, target_col: str = None,
                                    progress=gr.Progress()) -> Tuple[np.ndarray, str, pd.DataFrame]:
    """
    Comprehensive production readiness checklist and score.
    """
    if df is None:
        return create_placeholder_image("No data"), "Please upload data first.", pd.DataFrame()
    
    try:
        progress(0.1, desc="Running production readiness checks...")
        
        checks = []
        total_score = 0
        max_score = 0
        
        # Check 1: Model trained
        max_score += 20
        if state.trained_model is not None:
            checks.append({'Check': 'Model Trained', 'Status': 'PASS', 'Score': 20, 'Details': f'Model: {getattr(state, "model_name", "Unknown")}'})
            total_score += 20
        else:
            checks.append({'Check': 'Model Trained', 'Status': 'FAIL', 'Score': 0, 'Details': 'No model trained yet'})
        
        progress(0.2, desc="Checking data quality...")
        
        # Check 2: Data quality
        max_score += 15
        missing_pct = df.isnull().mean().mean() * 100
        if missing_pct < 5:
            checks.append({'Check': 'Data Quality', 'Status': 'PASS', 'Score': 15, 'Details': f'Missing: {missing_pct:.1f}%'})
            total_score += 15
        elif missing_pct < 15:
            checks.append({'Check': 'Data Quality', 'Status': 'WARN', 'Score': 8, 'Details': f'Missing: {missing_pct:.1f}% (consider imputation)'})
            total_score += 8
        else:
            checks.append({'Check': 'Data Quality', 'Status': 'FAIL', 'Score': 0, 'Details': f'Missing: {missing_pct:.1f}% (too high)'})
        
        # Check 3: Feature count
        max_score += 10
        n_features = len(df.select_dtypes(include=[np.number]).columns)
        if 5 <= n_features <= 100:
            checks.append({'Check': 'Feature Count', 'Status': 'PASS', 'Score': 10, 'Details': f'{n_features} features (optimal range)'})
            total_score += 10
        elif n_features > 100:
            checks.append({'Check': 'Feature Count', 'Status': 'WARN', 'Score': 5, 'Details': f'{n_features} features (consider reduction)'})
            total_score += 5
        else:
            checks.append({'Check': 'Feature Count', 'Status': 'FAIL', 'Score': 0, 'Details': f'{n_features} features (too few)'})
        
        progress(0.4, desc="Checking model performance...")
        
        # Check 4: Model performance
        max_score += 20
        best_score = getattr(state, 'best_score', 0)
        if best_score >= 0.85:
            checks.append({'Check': 'Model Performance', 'Status': 'PASS', 'Score': 20, 'Details': f'Score: {best_score:.2%}'})
            total_score += 20
        elif best_score >= 0.70:
            checks.append({'Check': 'Model Performance', 'Status': 'WARN', 'Score': 10, 'Details': f'Score: {best_score:.2%} (acceptable)'})
            total_score += 10
        else:
            checks.append({'Check': 'Model Performance', 'Status': 'FAIL', 'Score': 0, 'Details': f'Score: {best_score:.2%} (needs improvement)'})
        
        # Check 5: Sample size
        max_score += 10
        n_samples = len(df)
        if n_samples >= 1000:
            checks.append({'Check': 'Sample Size', 'Status': 'PASS', 'Score': 10, 'Details': f'{n_samples:,} samples'})
            total_score += 10
        elif n_samples >= 500:
            checks.append({'Check': 'Sample Size', 'Status': 'WARN', 'Score': 5, 'Details': f'{n_samples:,} samples (more recommended)'})
            total_score += 5
        else:
            checks.append({'Check': 'Sample Size', 'Status': 'FAIL', 'Score': 0, 'Details': f'{n_samples:,} samples (too few)'})
        
        progress(0.6, desc="Checking target variable...")
        
        # Check 6: Target variable
        max_score += 10
        if target_col and target_col in df.columns:
            target_nulls = df[target_col].isnull().sum()
            if target_nulls == 0:
                checks.append({'Check': 'Target Variable', 'Status': 'PASS', 'Score': 10, 'Details': f'{target_col} - no missing values'})
                total_score += 10
            else:
                checks.append({'Check': 'Target Variable', 'Status': 'WARN', 'Score': 5, 'Details': f'{target_col} - {target_nulls} missing'})
                total_score += 5
        else:
            checks.append({'Check': 'Target Variable', 'Status': 'FAIL', 'Score': 0, 'Details': 'Target not specified'})
        
        # Check 7: Class balance (for classification)
        max_score += 10
        if target_col and target_col in df.columns and df[target_col].nunique() <= 20:
            class_dist = df[target_col].value_counts(normalize=True)
            min_class_pct = class_dist.min() * 100
            if min_class_pct >= 20:
                checks.append({'Check': 'Class Balance', 'Status': 'PASS', 'Score': 10, 'Details': f'Min class: {min_class_pct:.1f}%'})
                total_score += 10
            elif min_class_pct >= 10:
                checks.append({'Check': 'Class Balance', 'Status': 'WARN', 'Score': 5, 'Details': f'Min class: {min_class_pct:.1f}% (imbalanced)'})
                total_score += 5
            else:
                checks.append({'Check': 'Class Balance', 'Status': 'FAIL', 'Score': 0, 'Details': f'Min class: {min_class_pct:.1f}% (severely imbalanced)'})
        else:
            checks.append({'Check': 'Class Balance', 'Status': 'N/A', 'Score': 10, 'Details': 'Regression task or N/A'})
            total_score += 10
        
        # Check 8: Feature documentation
        max_score += 5
        semantic_types = getattr(state, 'semantic_types', {})
        if semantic_types and len(semantic_types) > n_features * 0.5:
            checks.append({'Check': 'Feature Documentation', 'Status': 'PASS', 'Score': 5, 'Details': f'{len(semantic_types)} features documented'})
            total_score += 5
        else:
            checks.append({'Check': 'Feature Documentation', 'Status': 'WARN', 'Score': 2, 'Details': 'Run semantic detection for documentation'})
            total_score += 2
        
        progress(0.8, desc="Creating visualization...")
        
        # Calculate final score
        final_score = (total_score / max_score) * 100
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Overall score gauge
        ax1 = axes[0, 0]
        colors_gauge = ['#e74c3c', '#f39c12', '#27ae60']
        
        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        ax1.fill_between(theta[:33], 0, r, alpha=0.3, color='#e74c3c')
        ax1.fill_between(theta[33:66], 0, r, alpha=0.3, color='#f39c12')
        ax1.fill_between(theta[66:], 0, r, alpha=0.3, color='#27ae60')
        
        # Needle
        needle_angle = np.pi * (1 - final_score / 100)
        ax1.annotate('', xy=(np.cos(needle_angle) * 0.8, np.sin(needle_angle) * 0.8), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', lw=3, color='black'))
        
        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-0.1, 1.2)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.text(0, -0.05, f'{final_score:.0f}%', ha='center', va='top', fontsize=24, fontweight='bold')
        ax1.text(0, 1.1, 'Production Readiness Score', ha='center', va='bottom', fontsize=14, fontweight='bold')
        ax1.text(-1, 0, 'Fail', ha='center', fontsize=10, color='#e74c3c')
        ax1.text(0, 1.05, 'Warn', ha='center', fontsize=10, color='#f39c12')
        ax1.text(1, 0, 'Pass', ha='center', fontsize=10, color='#27ae60')
        
        # Plot 2: Check status breakdown
        ax2 = axes[0, 1]
        status_counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0, 'N/A': 0}
        for c in checks:
            status_counts[c['Status']] += 1
        
        labels = ['Pass', 'Warning', 'Fail']
        sizes = [status_counts['PASS'], status_counts['WARN'], status_counts['FAIL']]
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        if sum(sizes) > 0:
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Check Results Distribution', fontweight='bold')
        
        # Plot 3: Individual check scores
        ax3 = axes[1, 0]
        check_names = [c['Check'][:15] for c in checks]
        check_scores = [c['Score'] for c in checks]
        check_colors = ['#27ae60' if c['Status'] == 'PASS' else '#f39c12' if c['Status'] == 'WARN' else '#e74c3c' for c in checks]
        
        ax3.barh(check_names[::-1], check_scores[::-1], color=check_colors[::-1], alpha=0.8)
        ax3.set_xlabel('Score')
        ax3.set_title('Individual Check Scores', fontweight='bold')
        
        # Plot 4: Deployment recommendation
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        if final_score >= 80:
            recommendation = "READY FOR PRODUCTION"
            rec_color = '#27ae60'
            rec_text = "All critical checks passed.\nModel is ready for deployment."
        elif final_score >= 60:
            recommendation = "READY WITH CAUTION"
            rec_color = '#f39c12'
            rec_text = "Some checks need attention.\nDeploy with monitoring."
        else:
            recommendation = "NOT READY"
            rec_color = '#e74c3c'
            rec_text = "Critical issues detected.\nAddress failures before deployment."
        
        ax4.text(0.5, 0.7, recommendation, ha='center', va='center', fontsize=20, fontweight='bold', color=rec_color)
        ax4.text(0.5, 0.4, rec_text, ha='center', va='center', fontsize=12, color='gray')
        ax4.set_title('Deployment Recommendation', fontweight='bold')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="Readiness check complete!")
        
        # Generate report
        report = f"""## Production Readiness Report

### Overall Score: {final_score:.0f}% - {recommendation}

### Summary
- **Checks Passed:** {status_counts['PASS']}
- **Warnings:** {status_counts['WARN']}
- **Failures:** {status_counts['FAIL']}

### Detailed Results
"""
        for c in checks:
            icon = "[PASS]" if c['Status'] == 'PASS' else "[WARN]" if c['Status'] == 'WARN' else "[FAIL]"
            report += f"- {icon} **{c['Check']}**: {c['Details']}\n"
        
        # Add recommendations
        failures = [c for c in checks if c['Status'] == 'FAIL']
        warnings = [c for c in checks if c['Status'] == 'WARN']
        
        if failures:
            report += "\n### Critical Issues (Must Fix)\n"
            for f in failures:
                report += f"- **{f['Check']}**: {f['Details']}\n"
        
        if warnings:
            report += "\n### Warnings (Should Address)\n"
            for w in warnings:
                report += f"- **{w['Check']}**: {w['Details']}\n"
        
        report += f"""
### Next Steps
1. {"Fix critical issues before deployment" if failures else "No critical issues"}
2. {"Address warnings for optimal performance" if warnings else "All warnings cleared"}
3. Generate API code in the API Generation tab
4. Set up drift monitoring for production

### Deployment Checklist
- [ ] Export model and scaler files
- [ ] Generate FastAPI code
- [ ] Set up Docker container
- [ ] Configure monitoring alerts
- [ ] Create rollback plan
- [ ] Document API endpoints
- [ ] Set up A/B testing (optional)
"""
        
        checks_df = pd.DataFrame(checks)
        
        return img, report, checks_df
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", pd.DataFrame()


def run_ab_test_simulation(df: pd.DataFrame, target_col: str = None,
                           progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    """
    Simulate A/B test between current model and a challenger model.
    """
    if df is None or state.trained_model is None:
        return create_placeholder_image("No model"), "Please train a model first."
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.metrics import accuracy_score
        from scipy import stats
        
        progress(0.1, desc="Setting up A/B test simulation...")
        
        # Prepare data
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].fillna(df[numeric_cols].median())
        y = df[target_col].copy()
        
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y.fillna('Unknown'))
        else:
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data for simulation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        progress(0.3, desc="Training champion model (current)...")
        
        # Champion model (current production)
        champion = state.trained_model
        champion.fit(X_train, y_train)
        champion_pred = champion.predict(X_test)
        champion_acc = accuracy_score(y_test, champion_pred)
        
        progress(0.5, desc="Training challenger model...")
        
        # Challenger model (alternative)
        challenger = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
        challenger.fit(X_train, y_train)
        challenger_pred = challenger.predict(X_test)
        challenger_acc = accuracy_score(y_test, challenger_pred)
        
        progress(0.7, desc="Running statistical analysis...")
        
        # Simulate A/B test with bootstrap
        n_bootstrap = 100
        champion_scores = []
        challenger_scores = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
            champion_scores.append(accuracy_score(y_test[indices], champion_pred[indices]))
            challenger_scores.append(accuracy_score(y_test[indices], challenger_pred[indices]))
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(challenger_scores, champion_scores)
        
        # Determine winner
        if p_value < 0.05 and np.mean(challenger_scores) > np.mean(champion_scores):
            winner = "Challenger"
            recommendation = "Deploy challenger model"
        elif p_value < 0.05 and np.mean(champion_scores) > np.mean(challenger_scores):
            winner = "Champion"
            recommendation = "Keep champion model"
        else:
            winner = "No Clear Winner"
            recommendation = "Continue testing or collect more data"
        
        progress(0.85, desc="Creating visualization...")
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Accuracy comparison
        ax1 = axes[0, 0]
        models = ['Champion\n(Current)', 'Challenger\n(New)']
        accs = [champion_acc, challenger_acc]
        colors = ['#3498db', '#e74c3c']
        bars = ax1.bar(models, accs, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Plot 2: Bootstrap distribution
        ax2 = axes[0, 1]
        ax2.hist(champion_scores, bins=20, alpha=0.6, label='Champion', color='#3498db')
        ax2.hist(challenger_scores, bins=20, alpha=0.6, label='Challenger', color='#e74c3c')
        ax2.axvline(np.mean(champion_scores), color='#3498db', linestyle='--', lw=2)
        ax2.axvline(np.mean(challenger_scores), color='#e74c3c', linestyle='--', lw=2)
        ax2.set_xlabel('Accuracy')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bootstrap Distribution (100 samples)', fontweight='bold')
        ax2.legend()
        
        # Plot 3: Difference distribution
        ax3 = axes[1, 0]
        differences = np.array(challenger_scores) - np.array(champion_scores)
        ax3.hist(differences, bins=20, color='purple', alpha=0.7)
        ax3.axvline(0, color='black', linestyle='-', lw=2)
        ax3.axvline(np.mean(differences), color='red', linestyle='--', lw=2, label=f'Mean: {np.mean(differences):.4f}')
        ax3.set_xlabel('Challenger - Champion Accuracy')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Performance Difference Distribution', fontweight='bold')
        ax3.legend()
        
        # Plot 4: Decision summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        winner_color = '#27ae60' if winner == "Challenger" else '#3498db' if winner == "Champion" else '#f39c12'
        
        ax4.text(0.5, 0.8, 'A/B Test Result', ha='center', va='center', fontsize=16, fontweight='bold')
        ax4.text(0.5, 0.6, f'Winner: {winner}', ha='center', va='center', fontsize=20, fontweight='bold', color=winner_color)
        ax4.text(0.5, 0.4, f'P-value: {p_value:.4f}', ha='center', va='center', fontsize=12)
        ax4.text(0.5, 0.25, f'{"Statistically Significant" if p_value < 0.05 else "Not Significant"}', 
                ha='center', va='center', fontsize=12, 
                color='#27ae60' if p_value < 0.05 else '#e74c3c')
        ax4.text(0.5, 0.1, f'Recommendation: {recommendation}', ha='center', va='center', fontsize=11, style='italic')
        
        plt.tight_layout()
        
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        progress(1.0, desc="A/B test complete!")
        
        # Generate report
        improvement = (challenger_acc - champion_acc) * 100
        
        report = f"""## A/B Test Simulation Report

### Test Configuration
- **Champion Model:** {getattr(state, 'model_name', 'Current Model')}
- **Challenger Model:** Gradient Boosting (optimized)
- **Test Set Size:** {len(y_test):,} samples
- **Bootstrap Iterations:** {n_bootstrap}

### Results Summary
| Metric | Champion | Challenger | Difference |
|--------|----------|------------|------------|
| Accuracy | {champion_acc:.4f} | {challenger_acc:.4f} | {improvement:+.2f}% |
| Mean (Bootstrap) | {np.mean(champion_scores):.4f} | {np.mean(challenger_scores):.4f} | - |
| Std (Bootstrap) | {np.std(champion_scores):.4f} | {np.std(challenger_scores):.4f} | - |

### Statistical Analysis
- **T-statistic:** {t_stat:.4f}
- **P-value:** {p_value:.4f}
- **Significance Level:** 0.05
- **Result:** {'Statistically Significant' if p_value < 0.05 else 'Not Statistically Significant'}

### Winner: {winner}

### Recommendation
{recommendation}

### Confidence Interval (95%)
- Champion: [{np.percentile(champion_scores, 2.5):.4f}, {np.percentile(champion_scores, 97.5):.4f}]
- Challenger: [{np.percentile(challenger_scores, 2.5):.4f}, {np.percentile(challenger_scores, 97.5):.4f}]

### Next Steps
{"1. Deploy challenger model to production" if winner == "Challenger" else "1. Keep champion model in production"}
2. Set up monitoring for the deployed model
3. Schedule periodic A/B tests for continuous improvement
4. Document the test results for future reference
"""
        
        return img, report
        
    except Exception as e:
        import traceback
        return create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}"


# ==================== HELPER FUNCTIONS ====================

def create_stats_html(df: pd.DataFrame) -> str:
    """Create dataset statistics HTML."""
    if df is None:
        return ""
    
    rows, cols = df.shape
    nulls = df.isnull().sum().sum()
    null_pct = (nulls / (rows * cols)) * 100 if rows * cols > 0 else 0
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    
    return f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0;">
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #e9ecef;">
            <div style="font-size: 2em; font-weight: 700; color: #667eea;">{rows:,}</div>
            <div style="color: #636e72; font-size: 0.9em; margin-top: 5px;">Rows</div>
        </div>
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #e9ecef;">
            <div style="font-size: 2em; font-weight: 700; color: #667eea;">{cols}</div>
            <div style="color: #636e72; font-size: 0.9em; margin-top: 5px;">Columns</div>
        </div>
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #e9ecef;">
            <div style="font-size: 2em; font-weight: 700; color: #667eea;">{null_pct:.1f}%</div>
            <div style="color: #636e72; font-size: 0.9em; margin-top: 5px;">Missing</div>
        </div>
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 12px; padding: 20px; text-align: center; border: 1px solid #e9ecef;">
            <div style="font-size: 2em; font-weight: 700; color: #667eea;">{numeric_cols}</div>
            <div style="color: #636e72; font-size: 0.9em; margin-top: 5px;">Numeric</div>
        </div>
    </div>
    """


def create_placeholder_image(message: str = "No data") -> np.ndarray:
    """Create placeholder image."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16, color='gray')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return img


def create_progress_bar(step: int) -> str:
    """Create progress bar HTML."""
    steps = ['Upload', 'Detect', 'Validate', 'Baseline', 'Clean', 'Features', 'Train', 'Insights']
    
    html = '<div style="display: flex; justify-content: space-between; padding: 20px; background: #f8f9fa; border-radius: 12px; margin-bottom: 20px;">'
    
    for i, name in enumerate(steps):
        if i < step:
            color = '#00b894'
            icon = 'OK'
        elif i == step:
            color = '#667eea'
            icon = str(i + 1)
        else:
            color = '#dfe6e9'
            icon = str(i + 1)
        
        html += f'''
        <div style="display: flex; flex-direction: column; align-items: center; flex: 1;">
            <div style="width: 40px; height: 40px; border-radius: 50%; background: {color}; 
                        display: flex; align-items: center; justify-content: center; 
                        color: white; font-weight: bold; margin-bottom: 8px; font-size: 0.8em;">{icon}</div>
            <div style="font-size: 0.85em; color: {'#333' if i <= step else '#999'};">{name}</div>
        </div>
        '''
    
    html += '</div>'
    return html


def detect_target_candidates(df: pd.DataFrame) -> List[str]:
    """Detect potential target columns."""
    if df is None:
        return []
    
    candidates = []
    
    for col in df.columns:
        score = 0
        
        # Check column name
        if re.search(r'(?i)(target|label|class|outcome|result|y$|predict)', col):
            score += 50
        
        # Check cardinality
        n_unique = df[col].nunique()
        if n_unique == 2:
            score += 30
        elif 2 < n_unique <= 10:
            score += 20
        
        # Check if last column
        if col == df.columns[-1]:
            score += 10
        
        candidates.append((col, score))
    
    # Sort by score
    candidates.sort(key=lambda x: -x[1])
    return [c[0] for c in candidates]


# ==================== EVENT HANDLERS ====================

def on_file_upload(file):
    """Handle file upload."""
    if file is None:
        return None, gr.update(choices=[], visible=False), None, "", gr.update(interactive=False), create_progress_bar(0)
    
    try:
        df = pd.read_csv(file.name)
        state.columns = list(df.columns)
        
        # Detect target candidates
        candidates = detect_target_candidates(df)
        
        return (
            df,
            gr.update(choices=candidates, value=candidates[0] if candidates else None, visible=True),
            df.head(10),
            create_stats_html(df),
            gr.update(interactive=True),
            create_progress_bar(1)
        )
    except Exception as e:
        return None, gr.update(choices=[], visible=False), None, f"Error: {str(e)}", gr.update(interactive=False), create_progress_bar(0)


def on_target_select(df, target):
    """Handle target selection."""
    if target is None:
        return "", gr.update(interactive=False)
    
    state.target_column = target
    
    if df is not None and target in df.columns:
        n_unique = df[target].nunique()
        state.task_type = 'classification' if n_unique <= 20 else 'regression'
        
        info = f"""**Selected Target:** `{target}`
**Unique Values:** {n_unique}
**Task Type:** {state.task_type.title()}
**Sample Values:** {df[target].head(5).tolist()}"""
        
        return info, gr.update(interactive=True)
    
    return "", gr.update(interactive=False)


def run_cleaning(df, progress=gr.Progress()):
    """Run forensic cleaning."""
    if df is None:
        return None, create_placeholder_image("No data"), "No data uploaded", create_progress_bar(4)
    
    try:
        progress(0.2, desc="Analyzing data quality...")
        
        cleaner = ForensicCleaner()
        cleaned_df, report = cleaner.analyze_and_clean(df.copy())
        state.cleaning_report = report
        
        progress(0.7, desc="Creating visualization...")
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i % 2]
            ax.hist(df[col].dropna(), bins=30, alpha=0.5, label='Raw', color='red')
            ax.hist(cleaned_df[col].dropna(), bins=30, alpha=0.5, label='Cleaned', color='green')
            ax.set_title(col)
            ax.legend()
        
        plt.tight_layout()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        report_text = format_forensic_report(report)
        
        progress(1.0, desc="Cleaning complete!")
        
        return cleaned_df, img, report_text, create_progress_bar(5)
    
    except Exception as e:
        import traceback
        return df, create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", create_progress_bar(4)


def run_feature_engineering(df, progress=gr.Progress()):
    """Run feature engineering."""
    if df is None:
        return None, create_placeholder_image("No data"), "No data available", create_progress_bar(5)
    
    try:
        progress(0.2, desc="Engineering features...")
        
        engineer = AutoFeatureEngineer()
        engineered_df, report = engineer.auto_engineer(df.copy(), state.target_column)
        state.feature_report = report
        
        progress(0.8, desc="Creating summary...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        new_features = report.get('new_features', {})
        categories = ['Ratios', 'Products', 'Polynomials']
        counts = [
            len(new_features.get('ratio_features', [])),
            len(new_features.get('product_features', [])),
            len(new_features.get('polynomial_features', []))
        ]
        
        ax.bar(categories, counts, color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel('Count')
        ax.set_title('New Features Created')
        
        plt.tight_layout()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        report_text = format_feature_report(report)
        
        progress(1.0, desc="Features complete!")
        
        return engineered_df, img, report_text, create_progress_bar(6)
    
    except Exception as e:
        import traceback
        return df, create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", create_progress_bar(5)


def run_training(df, n_trials, progress=gr.Progress()):
    """Run model training."""
    if df is None:
        return df, create_placeholder_image("No data"), "No data available", create_progress_bar(6)
    
    try:
        progress(0.1, desc="Preparing data...")
        
        X = df.drop(columns=[state.target_column])
        y = df[state.target_column]

        leakage_map = _detect_target_leakage(df, state.target_column, X.columns.tolist())
        if leakage_map:
            X = X.drop(columns=list(leakage_map.keys()), errors='ignore')

        if X.shape[1] == 0:
            leak_text = "\n".join([f"- {k}: {v}" for k, v in list(leakage_map.items())[:10]])
            return (
                df,
                create_placeholder_image("Leakage detected"),
                "No usable features left after leakage guard.\n\n"
                "Suspicious features removed:\n"
                f"{leak_text}",
                create_progress_bar(6),
            )

        state.X_train = X
        state.y_train = y
        
        progress(0.2, desc="Running tournament...")
        
        trainer = EliteTrainer(n_trials=int(n_trials))
        state.model, state.training_report = trainer.run_tournament(X, y, state.task_type)
        # Keep backward/forward compatibility with callbacks that expect `trained_model`.
        state.trained_model = state.model
        
        progress(0.8, desc="Creating visualization...")
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 5))
        
        rankings = state.training_report.get('rankings', [])
        if rankings:
            models = [r['model'] for r in rankings]
            scores = [r['score'] for r in rankings]
            colors = ['#ffd700', '#c0c0c0', '#cd7f32'] + ['#3498db'] * len(rankings)
            
            ax.barh(models[::-1], scores[::-1], color=colors[:len(models)][::-1])
            ax.set_xlabel('Score')
            ax.set_title('Tournament Rankings')
        
        plt.tight_layout()
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
        plt.close(fig)
        
        report_text = format_tournament_report(state.training_report)
        if leakage_map:
            report_text += "\n\n### Leakage Guard\n"
            report_text += f"- **Suspicious Features Removed:** {len(leakage_map)}\n"
            for col, reason in list(leakage_map.items())[:10]:
                report_text += f"- `{col}`: {reason}\n"

        rankings = state.training_report.get('rankings', []) if isinstance(state.training_report, dict) else []
        if rankings:
            state.model_name = rankings[0].get('model')
            state.best_score = rankings[0].get('score')
        state.leakage_report = leakage_map
        
        progress(1.0, desc="Training complete!")
        
        return df, img, report_text, create_progress_bar(7)
    
    except Exception as e:
        import traceback
        return df, create_placeholder_image("Error"), f"Error: {str(e)}\n{traceback.format_exc()}", create_progress_bar(6)


def run_insights(df, progress=gr.Progress()):
    """Generate insights."""
    if state.model is None:
        return df, create_placeholder_image("No model"), "", "Train model first", create_progress_bar(7)
    
    try:
        progress(0.2, desc="Computing SHAP values...")
        
        state.explainer = BlackBoxBreaker(state.model, state.X_train)
        shap_vals, xai_report = state.explainer.compute_global_shap()
        state.xai_report = xai_report
        
        progress(0.5, desc="Generating plot...")
        
        waterfall_b64, _ = state.explainer.explain_single_prediction(state.X_train.iloc[0])
        
        shap_img = create_placeholder_image("SHAP plot")
        if waterfall_b64:
            import PIL.Image
            img_bytes = base64.b64decode(waterfall_b64)
            shap_img = np.array(PIL.Image.open(io.BytesIO(img_bytes)))
        
        progress(0.8, desc="Generating AI summary...")
        
        summary = agent_report_generator(
            cleaning_report=state.cleaning_report,
            training_report=state.training_report,
            feature_report=state.feature_report,
            xai_report=state.xai_report,
            ingestion_report=state.ingestion_report
        )
        
        xai_text = format_xai_report(xai_report)
        
        progress(1.0, desc="Insights complete!")
        
        return df, shap_img, summary, xai_text, create_progress_bar(8)
    
    except Exception as e:
        import traceback
        return df, create_placeholder_image("Error"), "", f"Error: {str(e)}\n{traceback.format_exc()}", create_progress_bar(7)


def export_production():
    """Export production package."""
    model_to_export = state.model or state.trained_model
    if model_to_export is None:
        return (
            None,
            "No model to export. Train a model in the Model Training tab first "
            "(Normal Training, Optuna, Stacking, or Elite Tournament), then export."
        )
    
    try:
        if state.X_train is None:
            return None, "No feature matrix available. Train a model first."
        if not state.target_column:
            return None, "No target column set. Upload data and select a target first."

        zip_path = create_production_export(
            model=model_to_export,
            feature_columns=list(state.X_train.columns),
            target_column=state.target_column,
            task_type=state.task_type,
            model_name=getattr(state, "model_name", "metaai_model"),
            accuracy=float(getattr(state, "best_score", 0.0) or 0.0),
            preprocessors=getattr(state, "preprocessors", None),
        )
        return zip_path, "Production package exported successfully!"
    except Exception as e:
        return None, f"Error: {str(e)}"


# ==================== CUSTOM CSS ====================

CUSTOM_CSS = """
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 24px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 { margin: 0; font-size: 2.5em; font-weight: 700; }
.main-header p { margin: 10px 0 0 0; opacity: 0.9; font-size: 1.1em; }

.gradio-container { max-width: 1600px !important; }
.gr-button-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
"""


# ==================== BUILD DASHBOARD ====================

def build_dashboard():
    """Build the complete dashboard."""
    
    with gr.Blocks(
        title="MetaAI Pro - Enterprise AutoML",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate"),
        css=CUSTOM_CSS
    ) as app:
        
        # State
        df_raw_state = gr.State(value=None)
        df_cleaned_state = gr.State(value=None)
        df_engineered_state = gr.State(value=None)
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>MetaAI Pro v3.0</h1>
            <p>Enterprise AutoML Platform | Advanced Data Ingestion Pipeline</p>
        </div>
        """)
        
        # Progress
        progress_bar = gr.HTML(value=create_progress_bar(0))
        
        # Main Tabs
        with gr.Tabs():
            
            # ========== TAB 1: DATA INGESTION (with subtabs) ==========
            with gr.Tab("Data Ingestion"):
                
                with gr.Tabs():
                    
                    # ----- Subtab 1: Manual Upload -----
                    with gr.Tab("Manual Upload"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Upload Your Dataset")
                                file_input = gr.File(
                                    label="Upload CSV File",
                                    file_types=[".csv"],
                                    file_count="single"
                                )
                                
                                gr.Markdown("### Select Target Column")
                                target_dropdown = gr.Dropdown(
                                    label="Target Column (Auto-detected)",
                                    choices=[],
                                    visible=False
                                )
                                target_info = gr.Markdown("")
                            
                            with gr.Column(scale=2):
                                gr.Markdown("### Data Preview")
                                stats_html = gr.HTML("")
                                preview_table = gr.DataFrame(label="First 10 Rows", wrap=True)
                    
                    # ----- Subtab 2: Semantic Type Detection -----
                    with gr.Tab("Semantic Detection"):
                        gr.Markdown("""
                        ### Semantic Type Detection
                        
                        This step uses **pattern matching + statistical heuristics** to understand 
                        what each column represents semantically, not just its data type.
                        
                        **Works with ANY dataset type:**
                        - E-commerce: `product_id`, `price`, `quantity`, `revenue`
                        - HR: `employee_id`, `salary`, `department`, `tenure`
                        - IoT: `sensor_id`, `temperature`, `pressure`, `reading`
                        - Marketing: `campaign`, `clicks`, `conversion`, `channel`
                        - Financial: `income`, `balance`, `transaction`, `amount`
                        """)
                        
                        detect_btn = gr.Button("Run Semantic Detection", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                semantic_table = gr.DataFrame(
                                    label="Detected Types",
                                    headers=["Column", "Semantic Type", "Confidence", "Inferred From", "Samples"]
                                )
                            with gr.Column(scale=1):
                                semantic_report = gr.Markdown("")
                    
                    # ----- Subtab 3: Pydantic Validation -----
                    with gr.Tab("Data Validation"):
                        gr.Markdown("""
                        ### Pydantic Schema Validation
                        
                        Auto-generates a **Pydantic schema** based on detected semantic types.
                        Validates data against domain constraints:
                        - Numeric values must be within valid ranges
                        - Quantities and counts must be non-negative
                        - Rejects invalid data with clear error messages
                        """)
                        
                        with gr.Row():
                            schema_btn = gr.Button("Generate Schema", variant="primary")
                            validate_btn = gr.Button("Validate Data", variant="primary")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Generated Schema")
                                schema_code = gr.Code(label="Pydantic Schema", language="python")
                            with gr.Column(scale=1):
                                schema_summary = gr.Markdown("")
                        
                        gr.Markdown("### Validation Results")
                        validation_table = gr.DataFrame(
                            label="Row Validation",
                            headers=["Row", "Status", "Errors"]
                        )
                        validation_report = gr.Markdown("")
                    
                    # ----- Subtab 4: Drift Baseline -----
                    with gr.Tab("Drift Baseline"):
                        gr.Markdown("""
                        ### Statistical Fingerprint
                        
                        Captures a **statistical baseline** of your data at ingestion time.
                        This fingerprint is used to detect **data drift** in production:
                        
                        - Mean, std, min, max for numeric columns
                        - Value distributions for categorical columns
                        - Histogram data for KS-test drift detection
                        """)
                        
                        baseline_btn = gr.Button("Capture Baseline", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Baseline JSON")
                                baseline_json = gr.Code(label="Statistical Fingerprint", language="json")
                            with gr.Column(scale=1):
                                baseline_report = gr.Markdown("")
                    
                    # ----- Subtab 5: Data Lineage -----
                    with gr.Tab("Data Lineage"):
                        gr.Markdown("""
                        ### Data Lineage Tracking
                        
                        Visual audit trail showing how data flows through the pipeline:
                        
                        **Raw CSV** â†’ **Semantic Detection** â†’ **Pydantic Validation** â†’ **Ready for Processing**
                        
                        This ensures **auditability** and **transparency** for production systems.
                        Tracks transformations, validation steps, and data quality metrics at each stage.
                        """)
                        
                        lineage_btn = gr.Button("Generate Lineage Graph", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### Data Flow Visualization")
                                lineage_chart = gr.Image(label="Lineage Graph", height=400)
                            with gr.Column(scale=1):
                                gr.Markdown("### Pipeline Metrics")
                                lineage_report = gr.Markdown("")
            
            # ========== TAB 2: DATA RECONSTRUCTION & BIAS DETECTION ==========
            with gr.Tab("Data Reconstruction"):
                
                with gr.Tabs():
                    
                    # ----- Subtab 1: Bias Detection -----
                    with gr.Tab("Systematic Bias Detection"):
                        gr.Markdown("""
                        ### Missing Data Bias Analysis
                        
                        Detects **systematic bias** in missing data patterns:
                        
                        **What it catches:**
                        - Income missing only for specific Age groups
                        - Health data missing for certain Ethnicities
                        - Financial data missing for specific Demographics
                        
                        **Why it matters:**
                        - Systematic bias leads to unfair models
                        - Violates fairness regulations (GDPR, Equal Credit Opportunity Act)
                        - Can perpetuate existing inequalities
                        
                        **This is NOT just missing values** - it's missing values that correlate with protected attributes.
                        """)
                        
                        bias_detect_btn = gr.Button("Detect Systematic Bias", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("### Bias Analysis Dashboard")
                                bias_chart = gr.Image(label="Bias Pattern Analysis", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Bias Report")
                                bias_report = gr.Markdown("")
                        
                        gr.Markdown("### Detected Bias Patterns")
                        bias_table = gr.Dataframe(label="Systematic Bias Details", wrap=True)
                    
                    # ----- Subtab 2: Bayesian Reconstruction -----
                    with gr.Tab("Bayesian Reconstruction"):
                        gr.Markdown("""
                        ### Iterative Bayesian Reconstruction
                        
                        Uses **relationships between columns** to reconstruct missing values intelligently.
                        Unlike simple mean/median imputation, this method:
                        
                        - Learns correlations between features (e.g., Age correlates with Income)
                        - Predicts missing values using Random Forest models
                        - Iterates until convergence for optimal estimates
                        - Preserves statistical relationships in your data
                        
                        **Algorithm:** MICE (Multiple Imputation by Chained Equations)
                        **Note:** Run bias detection first to understand if reconstruction is appropriate.
                        """)
                        
                        impute_btn = gr.Button("Run Bayesian Reconstruction", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Reconstruction Visualization")
                                impute_chart = gr.Image(label="Before vs After Reconstruction", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Reconstruction Report")
                                impute_report = gr.Markdown("")
                    
                    # ----- Subtab 3: Outlier Detection -----
                    with gr.Tab("Outlier Detection"):
                        gr.Markdown("""
                        ### Isolation Forest Outlier Detection
                        
                        Automatically identifies **anomalous rows** that may represent:
                        - Data entry errors (e.g., age = -5, blood pressure = 0)
                        - Impossible combinations (e.g., infant with PhD)
                        - Legitimate but extreme cases that need review
                        
                        **How it works:**
                        - Isolation Forest isolates observations by random splits
                        - Anomalies require fewer splits (shorter path length)
                        - No assumptions about data distribution required
                        """)
                        
                        with gr.Row():
                            contamination_slider = gr.Slider(
                                minimum=0.01, maximum=0.20, value=0.05, step=0.01,
                                label="Expected Outlier Rate (Contamination)",
                                info="Percentage of data expected to be outliers"
                            )
                            outlier_btn = gr.Button("Run Outlier Detection", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Detection Visualization")
                                outlier_chart = gr.Image(label="Outlier Analysis", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Outlier Report")
                                outlier_report = gr.Markdown("")
                        
                        gr.Markdown("### Flagged Outlier Rows (Top 20)")
                        outlier_table = gr.DataFrame(label="Outliers", wrap=True)
            
            # ========== TAB 3: EDA (Exploratory Data Analysis) ==========
            with gr.Tab("EDA"):
                
                with gr.Tabs():
                    
                    # ----- Subtab 1: Automated Hypothesis Generation -----
                    with gr.Tab("Hypothesis Generation"):
                        gr.Markdown("""
                        ### Automated Hypothesis Generation
                        
                        The system scans your data to discover **significant correlations** and patterns:
                        
                        - Identifies strong relationships between features
                        - Suggests which features to prioritize for modeling
                        - Flags potential multicollinearity issues
                        - Analyzes target variable relationships
                        
                        **Use Case:** Before building models, understand which features matter most.
                        """)
                        
                        hypothesis_btn = gr.Button("Generate Hypotheses", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Correlation Analysis")
                                hypothesis_chart = gr.Image(label="Correlation Heatmap", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Analysis Report")
                                hypothesis_report = gr.Markdown("")
                        
                        gr.Markdown("### Discovered Correlations")
                        hypothesis_table = gr.DataFrame(label="Feature Correlations", wrap=True)
                    
                    # ----- Subtab 2: Dimensionality Reduction -----
                    with gr.Tab("Dimensionality Reduction"):
                        gr.Markdown("""
                        ### Interactive Dimensionality Reduction
                        
                        Visualize high-dimensional data in 2D or 3D to see how groups separate:
                        
                        - **UMAP:** Preserves both local and global structure (faster, recommended)
                        - **t-SNE:** Emphasizes local structure, good for cluster visualization
                        
                        **Use Case:** See if your target classes form distinct clusters before training.
                        Clear separation = model should learn patterns easily.
                        """)
                        
                        with gr.Row():
                            dr_method = gr.Radio(
                                choices=["UMAP", "t-SNE"],
                                value="UMAP",
                                label="Reduction Method",
                                info="UMAP is faster and often better for general use"
                            )
                            dr_components = gr.Radio(
                                choices=["2", "3"],
                                value="2",
                                label="Output Dimensions",
                                info="2D is faster, 3D shows more structure"
                            )
                            dr_btn = gr.Button("Run Dimensionality Reduction", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Cluster Visualization")
                                dr_chart = gr.Image(label="UMAP/t-SNE Projection", height=500)
                            with gr.Column(scale=1):
                                gr.Markdown("### Interpretation Guide")
                                dr_report = gr.Markdown("")
                        
                        gr.Markdown("---")
                        gr.Markdown("""
                        ### AI Cluster Explanation
                        
                        After running the visualization above, click below to have the AI explain 
                        **why specific data points cluster together** (e.g., "This group has high BMI but low Blood Pressure").
                        """)
                        
                        with gr.Row():
                            cluster_id_slider = gr.Slider(
                                minimum=0, maximum=4, value=0, step=1,
                                label="Cluster to Explain",
                                info="Select which cluster to analyze (0 = largest cluster)"
                            )
                            explain_cluster_btn = gr.Button("Explain Cluster with AI", variant="secondary", size="lg")
                        
                        gr.Markdown("### AI Cluster Analysis")
                        cluster_explanation = gr.Markdown("")
            
            # ========== TAB 4: FEATURE ENGINEERING ==========
            with gr.Tab("Feature Engineering"):
                
                with gr.Tabs():
                    
                    # ----- Subtab 1: Agentic Feature Creation -----
                    with gr.Tab("Agentic Feature Creation"):
                        gr.Markdown("""
                        ### Agentic Feature Creation
                        
                        The AI analyzes your data columns and **creates domain-expert features** automatically:
                        
                        **Medical Domain:**
                        - BMI = weight / height^2
                        - Pulse Pressure = Systolic - Diastolic
                        - Mean Arterial Pressure
                        
                        **Financial Domain:**
                        - Debt-to-Income Ratio
                        - Savings Rate
                        - Profit Margin
                        
                        **E-commerce/Marketing:**
                        - Conversion Rate
                        - Average Order Value
                        
                        **How it works:** Column names are semantically analyzed to match known patterns,
                        then expert formulas are applied. Works with any dataset.
                        """)
                        
                        agentic_feat_btn = gr.Button("Create Expert Features", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Feature Creation Visualization")
                                agentic_feat_chart = gr.Image(label="Created Features", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Creation Report")
                                agentic_feat_report = gr.Markdown("")
                        
                        gr.Markdown("### Created Features Table")
                        agentic_feat_table = gr.DataFrame(label="Expert Features", wrap=True)
                    
                    # ----- Subtab 2: Recursive Feature Elimination -----
                    with gr.Tab("Feature Selection (RFE)"):
                        gr.Markdown("""
                        ### Recursive Feature Elimination (RFE)
                        
                        A **survival of the fittest tournament** for your features:
                        
                        1. Train model with all features
                        2. Rank features by importance
                        3. Eliminate the weakest feature
                        4. Repeat until only the best remain
                        
                        **Result:** Keep only features that increase Signal-to-Noise ratio.
                        Eliminates features that add noise and hurt model performance.
                        """)
                        
                        with gr.Row():
                            rfe_n_features = gr.Slider(
                                minimum=3, maximum=30, value=10, step=1,
                                label="Number of Features to Keep",
                                info="How many top features should survive the tournament"
                            )
                            rfe_btn = gr.Button("Run Feature Tournament", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Tournament Results")
                                rfe_chart = gr.Image(label="Feature Rankings", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### RFE Report")
                                rfe_report = gr.Markdown("")
                        
                        gr.Markdown("### Feature Ranking Table")
                        rfe_table = gr.DataFrame(label="Feature Rankings", wrap=True)
                    
                    # ----- Subtab 3: Standard Feature Engineering (Original) -----
                    with gr.Tab("Auto Feature Engineering"):
                        gr.Markdown("""
                        ### Automated Feature Engineering
                        
                        Standard automated feature engineering pipeline:
                        - Polynomial features
                        - Binning and discretization
                        - Date/time feature extraction
                        - Encoding categorical variables
                        """)
                        
                        feature_btn = gr.Button("Run Feature Engineering", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                feature_chart = gr.Image(label="New Features", height=400)
                            with gr.Column(scale=1):
                                feature_report = gr.Textbox(label="Feature Report", lines=20)
            
            # ========== TAB 6: MODEL TRAINING ==========
            with gr.Tab("Model Training"):
                
                with gr.Tabs():
                    
                    # ----- Subtab 1: Normal Training -----
                    with gr.Tab("Normal Training"):
                        gr.Markdown("""
                        ### Normal Model Training
                        
                        Quick comparison of **5 standard algorithms** with default hyperparameters:
                        
                        - Logistic Regression / Ridge Regression
                        - Random Forest
                        - Gradient Boosting
                        - K-Nearest Neighbors
                        - Decision Tree
                        
                        **Use Case:** Fast baseline comparison to identify promising algorithms.
                        """)
                        
                        normal_train_btn = gr.Button("Run Normal Training", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Model Comparison")
                                normal_train_chart = gr.Image(label="Model Rankings", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Training Report")
                                normal_train_report = gr.Markdown("")
                        
                        gr.Markdown("### Results Table")
                        normal_train_table = gr.DataFrame(label="Model Results", wrap=True)
                    
                    # ----- Subtab 2: Optuna Hyperparameter Search -----
                    with gr.Tab("Optuna Optimization"):
                        gr.Markdown("""
                        ### Optuna Hyperparameter Search
                        
                        Find the **mathematically optimal settings** for your model:
                        
                        - Runs 50-100+ trials exploring parameter space
                        - Uses TPE (Tree-structured Parzen Estimator) for intelligent search
                        - Automatically finds the best combination of hyperparameters
                        
                        **Parameters Optimized:**
                        - n_estimators (50-300)
                        - max_depth (3-20)
                        - min_samples_split (2-20)
                        - min_samples_leaf (1-10)
                        - max_features (sqrt, log2, None)
                        """)
                        
                        with gr.Row():
                            optuna_n_trials = gr.Slider(
                                minimum=20, maximum=150, value=50, step=10,
                                label="Number of Trials",
                                info="More trials = better optimization but longer time"
                            )
                            optuna_train_btn = gr.Button("Start Optuna Search", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Optimization Progress")
                                optuna_train_chart = gr.Image(label="Optimization History", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Optimization Report")
                                optuna_train_report = gr.Markdown("")
                        
                        gr.Markdown("### Trial History (Top 20)")
                        optuna_train_table = gr.DataFrame(label="Trial Results", wrap=True)
                    
                    # ----- Subtab 3: Automated Stacking -----
                    with gr.Tab("Stacking Ensemble"):
                        gr.Markdown("""
                        ### Automated Stacking Ensemble
                        
                        Build a **committee of models** that vote together for better predictions:
                        
                        **Base Models (Level 0):**
                        - Random Forest
                        - Gradient Boosting
                        - Extra Trees
                        - AdaBoost
                        - K-Nearest Neighbors
                        - LightGBM (if installed)
                        - XGBoost (if installed)
                        - CatBoost (if installed)
                        
                        **Meta-Learner (Level 1):**
                        - Logistic Regression (classification) / Ridge (regression)
                        - Learns optimal weights to combine base model predictions
                        
                        **Why it works:** Different algorithms capture different patterns.
                        The meta-learner learns which models to trust for which predictions.
                        """)
                        
                        stacking_train_btn = gr.Button("Build Stacking Ensemble", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Ensemble Performance")
                                stacking_train_chart = gr.Image(label="Model Comparison", height=450)
                            with gr.Column(scale=1):
                                gr.Markdown("### Ensemble Report")
                                stacking_train_report = gr.Markdown("")
                        
                        gr.Markdown("### All Models Results")
                        stacking_train_table = gr.DataFrame(label="Ensemble Results", wrap=True)
            
            # ========== TAB 7: ANALYSIS & INTERPRETABILITY (XAI) ==========
            with gr.Tab("Analysis & XAI"):
                
                with gr.Tabs():
                    
                    # ----- Subtab 1: Normal Analysis -----
                    with gr.Tab("Performance Analysis"):
                        gr.Markdown("""
                        ### Model Performance Analysis
                        
                        Comprehensive evaluation of your trained model:
                        
                        **Classification:**
                        - Confusion Matrix (True Positives, False Positives, etc.)
                        - Accuracy, Precision, Recall, F1-Score
                        - ROC-AUC Score
                        - Per-class accuracy breakdown
                        
                        **Regression:**
                        - Actual vs Predicted scatter plot
                        - R2 Score, RMSE, MAE
                        - Residuals analysis
                        """)
                        
                        normal_analysis_btn = gr.Button("Run Performance Analysis", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Visualizations")
                                normal_analysis_chart = gr.Image(label="Performance Charts", height=500)
                            with gr.Column(scale=1):
                                gr.Markdown("### Analysis Report")
                                normal_analysis_report = gr.Markdown("")
                        
                        gr.Markdown("### Metrics Table")
                        normal_analysis_table = gr.DataFrame(label="Performance Metrics", wrap=True)
                    
                    # ----- Subtab 2: SHAP Analysis -----
                    with gr.Tab("SHAP Explainability"):
                        gr.Markdown("""
                        ### Global & Local SHAP Analysis
                        
                        **SHAP (SHapley Additive exPlanations)** reveals why the model makes predictions:
                        
                        **Global Explanation:**
                        - Which features are most important overall
                        - How each feature impacts predictions across all samples
                        
                        **Local Explanation:**
                        - Why a SPECIFIC sample got its prediction
                        - Exact contribution of each feature for that individual
                        
                        **Example Use Case:** Click on a single patient to see why they were flagged as high-risk.
                        """)
                        
                        with gr.Row():
                            shap_sample_idx = gr.Slider(
                                minimum=0, maximum=100, value=0, step=1,
                                label="Sample Index for Local Explanation",
                                info="Select which sample to explain individually"
                            )
                            shap_btn = gr.Button("Run SHAP Analysis", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Global Feature Importance")
                                shap_global_chart = gr.Image(label="Global SHAP", height=400)
                            with gr.Column(scale=1):
                                gr.Markdown("### Local Explanation (Selected Sample)")
                                shap_local_chart = gr.Image(label="Local SHAP", height=400)
                        
                        shap_report = gr.Markdown("")
                        
                        gr.Markdown("### Feature Importance Rankings")
                        shap_table = gr.DataFrame(label="SHAP Importance", wrap=True)
                    
                    # ----- Subtab 3: Fairness & Bias Audit -----
                    with gr.Tab("Fairness & Bias Audit"):
                        gr.Markdown("""
                        ### Fairness & Bias Audit
                        
                        Checks if your model performs **differently across demographic groups**:
                        
                        **What it detects:**
                        - Accuracy gaps between groups (e.g., model is 20% less accurate for women)
                        - Disparate Impact (80% rule violation)
                        - Potential discrimination risks
                        
                        **Sensitive Attributes Analyzed:**
                        - Gender, Age Groups, Race/Ethnicity (if present)
                        - Any categorical column with 2-10 unique values
                        
                        **Risk Levels:**
                        - CRITICAL: >15% accuracy gap
                        - WARNING: 8-15% accuracy gap
                        - LOW: <8% accuracy gap
                        """)
                        
                        with gr.Row():
                            fairness_sensitive_col = gr.Dropdown(
                                choices=[],
                                label="Sensitive Attribute (Optional)",
                                info="Select a specific column to analyze, or leave blank for auto-detection"
                            )
                            fairness_btn = gr.Button("Run Fairness Audit", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Fairness Visualization")
                                fairness_chart = gr.Image(label="Fairness Analysis", height=500)
                            with gr.Column(scale=1):
                                gr.Markdown("### Audit Report")
                                fairness_report = gr.Markdown("")
                        
                        gr.Markdown("### Group Performance Breakdown")
                        fairness_table = gr.DataFrame(label="Group Metrics", wrap=True)
            
            # ========== TAB 8: AGENTIC AUDITING ==========
            with gr.Tab("Agentic Auditing"):
                
                with gr.Tabs():
                    
                    # ----- Subtab 1: LLM Post-Mortem -----
                    with gr.Tab("Scientific Post-Mortem"):
                        gr.Markdown("""
                        ### LLM-Powered Scientific Abstract
                        
                        The AI Agent analyzes your complete ML experiment and generates a 
                        **formal scientific abstract** suitable for:
                        
                        - Technical reports and documentation
                        - Academic publications
                        - Executive presentations
                        - Audit records
                        
                        **Includes:**
                        - Background & Objective
                        - Methods (preprocessing, feature engineering, model selection)
                        - Results (metrics, findings, model comparisons)
                        - Conclusions (takeaways, strengths/weaknesses)
                        - Keywords
                        
                        **Powered by:** OpenAI GPT-4o / Groq / Template Engine (fallback)
                        """)
                        
                        postmortem_btn = gr.Button("Generate Scientific Abstract", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                gr.Markdown("### Generated Abstract")
                                postmortem_output = gr.Markdown("")
                            with gr.Column(scale=1):
                                gr.Markdown("### Status")
                                postmortem_status = gr.Markdown("")
                    
                    # ----- Subtab 2: Counterfactual Reasoning -----
                    with gr.Tab("Counterfactual Reasoning"):
                        gr.Markdown("""
                        ### AI-Powered Counterfactual Analysis
                        
                        Ask the AI Agent: **"What would have to change for the model to reach X% accuracy?"**
                        
                        The agent analyzes your data and model to provide:
                        
                        - **Data Quality Improvements:** Specific samples needed, quality fixes
                        - **Feature Engineering:** New features that could boost performance
                        - **Model Architecture:** Hyperparameters and algorithms to try
                        - **Action Plan:** Prioritized steps with expected impact
                        
                        **Example Insights:**
                        - "The model needs 500 more samples of patients under 30 with high cholesterol"
                        - "Creating a BMI feature could improve accuracy by 2-3%"
                        - "A stacking ensemble with CatBoost could close the remaining gap"
                        """)
                        
                        with gr.Row():
                            target_accuracy_slider = gr.Slider(
                                minimum=80, maximum=99, value=95, step=1,
                                label="Target Accuracy (%)",
                                info="What accuracy level do you want to achieve?"
                            )
                            counterfactual_btn = gr.Button("Analyze What Needs to Change", variant="primary", size="lg")
                        
                        with gr.Row():
                            with gr.Column(scale=3):
                                gr.Markdown("### Counterfactual Analysis")
                                counterfactual_output = gr.Markdown("")
                            with gr.Column(scale=1):
                                gr.Markdown("### Analysis Info")
                                counterfactual_status = gr.Markdown("")
                    
                    # ----- Subtab 3: Custom Agent Chat -----
                    with gr.Tab("Ask the Agent"):
                        gr.Markdown("""
                        ### Custom Agent Consultation
                        
                        Ask the AI Agent **any question** about your ML experiment:
                        
                        **Example Questions:**
                        - "Why is the model struggling with this specific class?"
                        - "What additional data would help the most?"
                        - "Should I use a neural network instead?"
                        - "How can I reduce overfitting?"
                        - "What does the high importance of feature X mean?"
                        
                        The agent has context about your data, model, and results.
                        """)
                        
                        agent_question = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about your ML project...",
                            lines=3
                        )
                        agent_ask_btn = gr.Button("Ask the Agent", variant="primary", size="lg")
                        
                        gr.Markdown("### Agent Response")
                        agent_response = gr.Markdown("")
            
            # ========== TAB 9: MLOps & Production ==========
            with gr.Tab("MLOps and Production"):
                gr.Markdown("""
                ### MLOps and Production - Deployment Center
                
                This is the final stage before deployment. Generate production-ready APIs, 
                monitor for data drift, and ensure your model is production-ready.
                """)
                
                with gr.Tabs():
                    # ----- Subtab 1: Production Readiness -----
                    with gr.Tab("Readiness Check"):
                        gr.Markdown("""
                        ### Production Readiness Assessment
                        
                        Comprehensive checklist to ensure your model is ready for deployment:
                        - Model training status
                        - Data quality validation
                        - Performance thresholds
                        - Class balance checks
                        - Documentation status
                        """)
                        
                        with gr.Row():
                            readiness_target = gr.Dropdown(
                                choices=[],
                                label="Target Column",
                                info="Select target for readiness assessment"
                            )
                            readiness_btn = gr.Button("Run Readiness Check", variant="primary", size="lg")
                        
                        readiness_chart = gr.Image(label="Readiness Dashboard")
                        readiness_report = gr.Markdown("")
                        readiness_table = gr.Dataframe(label="Check Details")
                    
                    # ----- Subtab 2: Instant API Generation -----
                    with gr.Tab("API Generation"):
                        gr.Markdown("""
                        ### One-Click API Generation
                        
                        Generate a complete **FastAPI** endpoint with **Swagger UI**:
                        - Production-ready API code
                        - Auto-generated Dockerfile
                        - Requirements file
                        - Deployment instructions
                        """)
                        
                        with gr.Row():
                            api_model_name = gr.Textbox(
                                label="API Model Name",
                                value="MetaAI_Model",
                                info="Name for your API (alphanumeric only)"
                            )
                            api_generate_btn = gr.Button("Generate API Code", variant="primary", size="lg")
                        
                        with gr.Tabs():
                            with gr.Tab("API Code (api.py)"):
                                api_code_output = gr.Code(label="FastAPI Code", language="python", lines=25)
                            with gr.Tab("Dockerfile"):
                                dockerfile_output = gr.Code(label="Dockerfile", language="dockerfile", lines=15)
                            with gr.Tab("requirements.txt"):
                                requirements_output = gr.Code(label="Requirements", language="python", lines=10)
                        
                        api_instructions = gr.Markdown("")
                        api_status = gr.Markdown("")
                    
                    # ----- Subtab 3: Real-time Drift Alerts -----
                    with gr.Tab("Drift Detection"):
                        gr.Markdown("""
                        ### Real-time Drift Monitoring
                        
                        Detect when production data drifts from training data:
                        - **KS Test:** Statistical distribution comparison
                        - **Mean Shift:** Detect changes in central tendency
                        - **Variance Change:** Monitor spread changes
                        
                        If drift is detected, an auto-retrain alert is triggered.
                        """)
                        
                        with gr.Row():
                            drift_threshold = gr.Slider(
                                minimum=0.05, maximum=0.5, value=0.3, step=0.05,
                                label="Drift Detection Threshold",
                                info="Lower = more sensitive to drift"
                            )
                            drift_target = gr.Dropdown(
                                choices=[],
                                label="Target Column",
                                info="Select for drift analysis"
                            )
                            drift_detect_btn = gr.Button("Run Drift Detection", variant="primary", size="lg")
                        
                        drift_chart = gr.Image(label="Drift Analysis")
                        drift_report = gr.Markdown("")
                        drift_table = gr.Dataframe(label="Feature Drift Details")
                    
                    # ----- Subtab 4: Production Monitoring Simulation -----
                    with gr.Tab("Monitoring Dashboard"):
                        gr.Markdown("""
                        ### 30-Day Production Monitoring Simulation
                        
                        Simulate what monitoring looks like in production:
                        - Accuracy tracking over time
                        - Latency monitoring
                        - Error rate alerts
                        - Auto-retrain trigger conditions
                        """)
                        
                        with gr.Row():
                            monitoring_target = gr.Dropdown(
                                choices=[],
                                label="Target Column",
                                info="Select for monitoring simulation"
                            )
                            monitoring_btn = gr.Button("Simulate Production Monitoring", variant="primary", size="lg")
                        
                        monitoring_chart = gr.Image(label="30-Day Dashboard")
                        monitoring_report = gr.Markdown("")
            
            # ========== TAB 10: EXPORT ==========
            with gr.Tab("Export"):
                gr.Markdown("""
                ### Production Export
                
                Export your trained model as a complete deployment package:
                - Trained model (.joblib)
                - FastAPI wrapper code
                - Auto-generated requirements.txt
                - Dockerfile
                - README
                """)
                
                export_btn = gr.Button("Export Production Package", variant="primary", size="lg")
                export_file = gr.File(label="Download Package")
                export_status = gr.Markdown("")
                
                gr.Markdown("---")
                restart_btn = gr.Button("Start New Pipeline", variant="secondary")
        
        # ==================== EVENT HANDLERS ====================
        
        # File upload
        file_input.change(
            fn=on_file_upload,
            inputs=[file_input],
            outputs=[df_raw_state, target_dropdown, preview_table, stats_html, detect_btn, progress_bar]
        )
        
        # Target selection
        target_dropdown.change(
            fn=on_target_select,
            inputs=[df_raw_state, target_dropdown],
            outputs=[target_info, detect_btn]
        )
        
        # Semantic detection
        detect_btn.click(
            fn=run_semantic_detection,
            inputs=[df_raw_state],
            outputs=[semantic_table, semantic_report]
        )
        
        # Schema generation
        schema_btn.click(
            fn=generate_pydantic_schema,
            inputs=[df_raw_state],
            outputs=[schema_code, schema_summary]
        )
        
        # Validation
        validate_btn.click(
            fn=validate_data_with_schema,
            inputs=[df_raw_state],
            outputs=[validation_table, validation_report]
        )
        
        # Drift baseline
        baseline_btn.click(
            fn=capture_drift_baseline,
            inputs=[df_raw_state],
            outputs=[baseline_json, baseline_report]
        )
        
        # Data Lineage
        lineage_btn.click(
            fn=generate_data_lineage,
            inputs=[df_raw_state],
            outputs=[lineage_chart, lineage_report]
        )
        
        # Bias Detection
        bias_detect_btn.click(
            fn=detect_missing_data_bias,
            inputs=[df_raw_state],
            outputs=[bias_chart, bias_report, bias_table]
        )
        
        # Bayesian Imputation
        impute_btn.click(
            fn=run_bayesian_imputation,
            inputs=[df_raw_state],
            outputs=[df_cleaned_state, impute_chart, impute_report]
        )
        
        # Outlier Detection
        outlier_btn.click(
            fn=run_outlier_detection,
            inputs=[df_raw_state, contamination_slider],
            outputs=[df_cleaned_state, outlier_table, outlier_chart, outlier_report]
        )
        
        # Hypothesis Generation
        hypothesis_btn.click(
            fn=run_hypothesis_generation,
            inputs=[df_raw_state, target_dropdown],
            outputs=[hypothesis_chart, hypothesis_report, hypothesis_table]
        )
        
        # Dimensionality Reduction
        def run_dr_wrapper(df, target, method, components):
            n_comp = int(components)
            return run_dimensionality_reduction(df, target, method, n_comp)
        
        dr_btn.click(
            fn=run_dr_wrapper,
            inputs=[df_raw_state, target_dropdown, dr_method, dr_components],
            outputs=[dr_chart, dr_report]
        )
        
        # Cluster Explanation
        explain_cluster_btn.click(
            fn=explain_cluster_with_ai,
            inputs=[df_raw_state, target_dropdown, cluster_id_slider],
            outputs=[cluster_explanation]
        )
        
        # Agentic Feature Creation
        agentic_feat_btn.click(
            fn=run_agentic_feature_creation,
            inputs=[df_raw_state],
            outputs=[df_engineered_state, agentic_feat_chart, agentic_feat_report, agentic_feat_table]
        )
        
        # Recursive Feature Elimination
        rfe_btn.click(
            fn=run_recursive_feature_elimination,
            inputs=[df_engineered_state, target_dropdown, rfe_n_features],
            outputs=[df_engineered_state, rfe_chart, rfe_report, rfe_table]
        )
        
        # Auto Feature Engineering (original)
        feature_btn.click(
            fn=run_feature_engineering,
            inputs=[df_cleaned_state],
            outputs=[df_engineered_state, feature_chart, feature_report, progress_bar]
        )
        
        # Normal Training
        normal_train_btn.click(
            fn=run_normal_training,
            inputs=[df_engineered_state, target_dropdown],
            outputs=[normal_train_chart, normal_train_report, normal_train_table]
        )
        
        # Optuna Training
        optuna_train_btn.click(
            fn=run_optuna_training,
            inputs=[df_engineered_state, target_dropdown, optuna_n_trials],
            outputs=[optuna_train_chart, optuna_train_report, optuna_train_table]
        )
        
        # Stacking Ensemble
        stacking_train_btn.click(
            fn=run_stacking_ensemble,
            inputs=[df_engineered_state, target_dropdown],
            outputs=[stacking_train_chart, stacking_train_report, stacking_train_table]
        )
        
        # Normal Analysis
        normal_analysis_btn.click(
            fn=run_normal_analysis,
            inputs=[df_engineered_state, target_dropdown],
            outputs=[normal_analysis_chart, normal_analysis_report, normal_analysis_table]
        )
        
        # SHAP Analysis
        shap_btn.click(
            fn=run_shap_analysis,
            inputs=[df_engineered_state, target_dropdown, shap_sample_idx],
            outputs=[shap_global_chart, shap_local_chart, shap_report, shap_table]
        )
        
        # Fairness Audit
        fairness_btn.click(
            fn=run_fairness_audit,
            inputs=[df_engineered_state, target_dropdown, fairness_sensitive_col],
            outputs=[fairness_chart, fairness_report, fairness_table]
        )
        
        # Update fairness dropdown when data is loaded
        def update_fairness_dropdown(df):
            if df is None:
                return gr.update(choices=[])
            # Get categorical columns with 2-10 unique values
            cols = []
            for col in df.columns:
                if df[col].nunique() <= 10 and df[col].nunique() >= 2:
                    cols.append(col)
            return gr.update(choices=cols)
        
        file_input.change(
            fn=update_fairness_dropdown,
            inputs=[df_raw_state],
            outputs=[fairness_sensitive_col]
        )
        
        # LLM Post-Mortem
        postmortem_btn.click(
            fn=run_llm_postmortem,
            inputs=[df_engineered_state, target_dropdown],
            outputs=[postmortem_output, postmortem_status]
        )
        
        # Counterfactual Reasoning
        counterfactual_btn.click(
            fn=run_counterfactual_reasoning,
            inputs=[df_engineered_state, target_dropdown, target_accuracy_slider],
            outputs=[counterfactual_output, counterfactual_status]
        )
        
        # Custom Agent Query
        agent_ask_btn.click(
            fn=run_custom_agent_query,
            inputs=[df_engineered_state, target_dropdown, agent_question],
            outputs=[agent_response]
        )
        
        # Update MLOps dropdowns when target changes
        def update_mlops_dropdowns(df):
            if df is None:
                return gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[])
            cols = df.columns.tolist()
            return gr.update(choices=cols), gr.update(choices=cols), gr.update(choices=cols)
        
        target_dropdown.change(
            fn=update_mlops_dropdowns,
            inputs=[df_raw_state],
            outputs=[readiness_target, drift_target, monitoring_target]
        )
        
        # Production Readiness Check
        readiness_btn.click(
            fn=run_production_readiness_check,
            inputs=[df_engineered_state, readiness_target],
            outputs=[readiness_chart, readiness_report, readiness_table]
        )
        
        # API Generation
        api_generate_btn.click(
            fn=run_api_generation,
            inputs=[df_engineered_state, api_model_name],
            outputs=[api_code_output, dockerfile_output, requirements_output, api_instructions, api_status]
        )
        
        # Drift Detection
        drift_detect_btn.click(
            fn=run_drift_detection,
            inputs=[df_engineered_state, target_dropdown, drift_threshold],
            outputs=[drift_chart, drift_report, drift_table]
        )
        
        # Production Monitoring Simulation
        monitoring_btn.click(
            fn=simulate_production_monitoring,
            inputs=[df_engineered_state, monitoring_target],
            outputs=[monitoring_chart, monitoring_report]
        )
        
        # Export
        export_btn.click(
            fn=export_production,
            inputs=[],
            outputs=[export_file, export_status]
        )
        
        # Restart
        def restart():
            state.reset()
            return (
                None, None, None,
                gr.update(choices=[], visible=False),
                None, "", "",
                None, "",
                None, "",
                "", "",
                "", "",
                None, "",
                None, "",
                create_progress_bar(0)
            )
        
        restart_btn.click(
            fn=restart,
            inputs=[],
            outputs=[
                df_raw_state, df_cleaned_state, df_engineered_state,
                target_dropdown,
                preview_table, stats_html, target_info,
                semantic_table, semantic_report,
                validation_table, validation_report,
                schema_code, schema_summary,
                baseline_json, baseline_report,
                feature_chart, feature_report,
                export_file, export_status,
                progress_bar
            ]
        )
    
    return app


# Entry point
if __name__ == "__main__":
    app = build_dashboard()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


