"""
AdvancedDataInformer - Semantic Type Detection & Bayesian Imputation
Enterprise-grade data ingestion with medical domain awareness
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Enable experimental features
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class SemanticType(Enum):
    """Semantic types for medical/healthcare data"""
    # Vital Signs
    BLOOD_PRESSURE_SYSTOLIC = "blood_pressure_systolic"
    BLOOD_PRESSURE_DIASTOLIC = "blood_pressure_diastolic"
    HEART_RATE = "heart_rate"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    OXYGEN_SATURATION = "oxygen_saturation"
    
    # Demographics
    AGE = "age"
    HEIGHT = "height"
    WEIGHT = "weight"
    BMI = "bmi"
    
    # Lab Values
    CHOLESTEROL_TOTAL = "cholesterol_total"
    CHOLESTEROL_HDL = "cholesterol_hdl"
    CHOLESTEROL_LDL = "cholesterol_ldl"
    TRIGLYCERIDES = "triglycerides"
    BLOOD_GLUCOSE = "blood_glucose"
    HBA1C = "hba1c"
    CREATININE = "creatinine"
    
    # Binary/Categorical
    BINARY_FLAG = "binary_flag"
    CATEGORICAL = "categorical"
    
    # Generic
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Profile for a single column with semantic information"""
    name: str
    dtype: str
    semantic_type: SemanticType
    missing_count: int
    missing_pct: float
    unique_count: int
    min_val: Optional[float]
    max_val: Optional[float]
    mean_val: Optional[float]
    std_val: Optional[float]
    medical_interpretation: str
    risk_level: str  # "normal", "warning", "critical"


class AdvancedDataInformer:
    """
    Advanced Data Informer with Semantic Type Detection and Bayesian Imputation
    
    Features:
    - Semantic type detection for medical/healthcare data
    - Bayesian Iterative Imputation using sklearn's IterativeImputer
    - Automatic data quality assessment
    - Medical domain-aware validation
    """
    
    # Medical type detection rules (column_name_patterns, value_ranges)
    SEMANTIC_RULES = {
        SemanticType.AGE: {
            'patterns': ['age', 'patient_age', 'age_years', 'años'],
            'range': (0, 120),
            'typical_range': (18, 85)
        },
        SemanticType.BLOOD_PRESSURE_SYSTOLIC: {
            'patterns': ['systolic', 'sbp', 'sys_bp', 'blood_pressure_sys'],
            'range': (70, 250),
            'typical_range': (90, 140)
        },
        SemanticType.BLOOD_PRESSURE_DIASTOLIC: {
            'patterns': ['diastolic', 'dbp', 'dia_bp', 'blood_pressure_dia'],
            'range': (40, 150),
            'typical_range': (60, 90)
        },
        SemanticType.HEART_RATE: {
            'patterns': ['heart_rate', 'pulse', 'hr', 'bpm', 'heartrate'],
            'range': (30, 220),
            'typical_range': (60, 100)
        },
        SemanticType.BMI: {
            'patterns': ['bmi', 'body_mass_index', 'bodymassindex'],
            'range': (10, 60),
            'typical_range': (18.5, 30)
        },
        SemanticType.WEIGHT: {
            'patterns': ['weight', 'body_weight', 'mass', 'kg', 'weight_kg'],
            'range': (20, 300),
            'typical_range': (50, 100)
        },
        SemanticType.HEIGHT: {
            'patterns': ['height', 'stature', 'height_cm'],
            'range': (100, 250),
            'typical_range': (150, 190)
        },
        SemanticType.CHOLESTEROL_TOTAL: {
            'patterns': ['cholesterol', 'total_chol', 'chol', 'totalcholesterol'],
            'range': (100, 400),
            'typical_range': (150, 200)
        },
        SemanticType.CHOLESTEROL_HDL: {
            'patterns': ['hdl', 'hdl_chol', 'good_cholesterol', 'highchol'],
            'range': (20, 100),
            'typical_range': (40, 60)
        },
        SemanticType.CHOLESTEROL_LDL: {
            'patterns': ['ldl', 'ldl_chol', 'bad_cholesterol'],
            'range': (50, 250),
            'typical_range': (70, 130)
        },
        SemanticType.TRIGLYCERIDES: {
            'patterns': ['triglycerides', 'trig', 'trigs'],
            'range': (30, 500),
            'typical_range': (50, 150)
        },
        SemanticType.BLOOD_GLUCOSE: {
            'patterns': ['glucose', 'blood_sugar', 'fasting_glucose', 'sugar', 'bloodsugar'],
            'range': (40, 500),
            'typical_range': (70, 100)
        },
        SemanticType.HBA1C: {
            'patterns': ['hba1c', 'a1c', 'glycated', 'hemoglobin_a1c'],
            'range': (4, 15),
            'typical_range': (4, 5.7)
        },
        SemanticType.TEMPERATURE: {
            'patterns': ['temperature', 'temp', 'body_temp', 'fever'],
            'range': (35, 42),
            'typical_range': (36.1, 37.2)
        },
        SemanticType.OXYGEN_SATURATION: {
            'patterns': ['spo2', 'oxygen', 'saturation', 'o2_sat', 'oxygensaturation'],
            'range': (70, 100),
            'typical_range': (95, 100)
        },
        SemanticType.RESPIRATORY_RATE: {
            'patterns': ['respiratory', 'resp_rate', 'breathing', 'rr'],
            'range': (8, 40),
            'typical_range': (12, 20)
        },
        SemanticType.CREATININE: {
            'patterns': ['creatinine', 'creat', 'serum_creatinine'],
            'range': (0.3, 15),
            'typical_range': (0.6, 1.2)
        }
    }
    
    def __init__(self, df: pd.DataFrame = None):
        self.df = df
        self.profiles: Dict[str, ColumnProfile] = {}
        self.imputer = None
        self._imputed_df = None
        
    def load_data(self, df: pd.DataFrame) -> 'AdvancedDataInformer':
        """Load DataFrame for analysis"""
        self.df = df.copy()
        return self
    
    def detect_semantic_type(self, column: str) -> Tuple[SemanticType, str]:
        """
        Detect semantic type of a column using name patterns and value ranges
        
        Returns:
            Tuple of (SemanticType, medical_interpretation)
        """
        if self.df is None or column not in self.df.columns:
            return SemanticType.UNKNOWN, "Column not found"
        
        col_data = self.df[column]
        col_name_lower = column.lower().replace('_', '').replace(' ', '')
        
        # Check if binary
        unique_vals = col_data.dropna().unique()
        if len(unique_vals) == 2:
            if set(unique_vals).issubset({0, 1, True, False, 'yes', 'no', 'Yes', 'No', 'YES', 'NO'}):
                return SemanticType.BINARY_FLAG, "Binary indicator variable"
        
        # Check if categorical (few unique values)
        if len(unique_vals) <= 10 and col_data.dtype == 'object':
            return SemanticType.CATEGORICAL, f"Categorical with {len(unique_vals)} categories"
        
        # Check if identifier
        if len(unique_vals) == len(col_data.dropna()) and col_data.dtype in ['object', 'int64']:
            if any(p in col_name_lower for p in ['id', 'key', 'index', 'code']):
                return SemanticType.IDENTIFIER, "Unique identifier"
        
        # Numeric type detection
        if pd.api.types.is_numeric_dtype(col_data):
            col_min = col_data.min()
            col_max = col_data.max()
            col_mean = col_data.mean()
            
            # Check against medical semantic rules
            for sem_type, rules in self.SEMANTIC_RULES.items():
                # Check name patterns
                name_match = any(p in col_name_lower for p in rules['patterns'])
                
                # Check value range
                range_min, range_max = rules['range']
                typical_min, typical_max = rules['typical_range']
                
                range_match = (col_min >= range_min * 0.8 and col_max <= range_max * 1.2)
                
                if name_match and range_match:
                    # Determine risk level
                    if typical_min <= col_mean <= typical_max:
                        interpretation = f"Values within normal {sem_type.value} range"
                    else:
                        interpretation = f"Some values outside typical {sem_type.value} range ({typical_min}-{typical_max})"
                    return sem_type, interpretation
                
                # Strong name match without range validation
                if name_match:
                    return sem_type, f"Detected as {sem_type.value} by column name"
            
            # Generic numeric types
            if col_data.dtype == 'float64' or (col_max - col_min) > 10:
                return SemanticType.CONTINUOUS, "Continuous numeric variable"
            else:
                return SemanticType.DISCRETE, "Discrete numeric variable"
        
        return SemanticType.UNKNOWN, "Unable to determine semantic type"
    
    def get_risk_level(self, column: str, semantic_type: SemanticType) -> str:
        """Determine risk level based on value distribution"""
        if semantic_type not in self.SEMANTIC_RULES:
            return "normal"
        
        rules = self.SEMANTIC_RULES[semantic_type]
        typical_min, typical_max = rules['typical_range']
        
        col_data = self.df[column].dropna()
        if len(col_data) == 0:
            return "unknown"
        
        # Calculate percentage outside typical range
        outside_range = ((col_data < typical_min) | (col_data > typical_max)).sum()
        pct_outside = outside_range / len(col_data) * 100
        
        if pct_outside > 30:
            return "critical"
        elif pct_outside > 15:
            return "warning"
        return "normal"
    
    def profile_column(self, column: str) -> ColumnProfile:
        """Generate comprehensive profile for a column"""
        col_data = self.df[column]
        
        semantic_type, interpretation = self.detect_semantic_type(column)
        risk_level = self.get_risk_level(column, semantic_type)
        
        profile = ColumnProfile(
            name=column,
            dtype=str(col_data.dtype),
            semantic_type=semantic_type,
            missing_count=col_data.isnull().sum(),
            missing_pct=col_data.isnull().sum() / len(col_data) * 100,
            unique_count=col_data.nunique(),
            min_val=float(col_data.min()) if pd.api.types.is_numeric_dtype(col_data) else None,
            max_val=float(col_data.max()) if pd.api.types.is_numeric_dtype(col_data) else None,
            mean_val=float(col_data.mean()) if pd.api.types.is_numeric_dtype(col_data) else None,
            std_val=float(col_data.std()) if pd.api.types.is_numeric_dtype(col_data) else None,
            medical_interpretation=interpretation,
            risk_level=risk_level
        )
        
        self.profiles[column] = profile
        return profile
    
    def profile_all_columns(self) -> Dict[str, ColumnProfile]:
        """Profile all columns in the DataFrame"""
        for col in self.df.columns:
            self.profile_column(col)
        return self.profiles
    
    def bayesian_iterative_imputation(self, 
                                       max_iter: int = 10,
                                       random_state: int = 42,
                                       estimator: str = 'bayesian') -> pd.DataFrame:
        """
        Perform Bayesian Iterative Imputation using sklearn's IterativeImputer
        
        This method fills missing values by modeling each feature with missing values
        as a function of other features, using round-robin imputation.
        
        Args:
            max_iter: Maximum number of imputation rounds
            random_state: Random seed for reproducibility
            estimator: 'bayesian' uses BayesianRidge, 'rf' uses RandomForest
            
        Returns:
            DataFrame with imputed values
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Separate numeric and non-numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return self.df.copy()
        
        # Create imputer
        if estimator == 'rf':
            base_estimator = RandomForestRegressor(n_estimators=10, random_state=random_state)
        else:
            # Default: BayesianRidge (Bayesian approach)
            from sklearn.linear_model import BayesianRidge
            base_estimator = BayesianRidge()
        
        self.imputer = IterativeImputer(
            estimator=base_estimator,
            max_iter=max_iter,
            random_state=random_state,
            initial_strategy='mean',
            imputation_order='ascending',
            skip_complete=True,
            verbose=0
        )
        
        # Fit and transform numeric columns
        numeric_data = self.df[numeric_cols].values
        imputed_numeric = self.imputer.fit_transform(numeric_data)
        
        # Create result DataFrame
        result_df = self.df.copy()
        result_df[numeric_cols] = imputed_numeric
        
        # For non-numeric columns, use mode imputation
        for col in non_numeric_cols:
            if result_df[col].isnull().any():
                mode_val = result_df[col].mode()
                if len(mode_val) > 0:
                    result_df[col].fillna(mode_val[0], inplace=True)
        
        self._imputed_df = result_df
        return result_df
    
    def get_imputation_report(self) -> str:
        """Generate report on imputation performed"""
        if self._imputed_df is None:
            return "No imputation performed yet."
        
        original_missing = self.df.isnull().sum().sum()
        final_missing = self._imputed_df.isnull().sum().sum()
        
        report_lines = [
            "## Bayesian Iterative Imputation Report",
            "",
            f"**Original Missing Values:** {original_missing}",
            f"**After Imputation:** {final_missing}",
            f"**Values Imputed:** {original_missing - final_missing}",
            "",
            "### Imputation Details by Column",
            "",
            "| Column | Original Missing | Imputed |",
            "|--------|-----------------|---------|"
        ]
        
        for col in self.df.columns:
            orig_miss = self.df[col].isnull().sum()
            if orig_miss > 0:
                final_miss = self._imputed_df[col].isnull().sum()
                report_lines.append(f"| {col} | {orig_miss} | {orig_miss - final_miss} |")
        
        report_lines.extend([
            "",
            "**Method:** Bayesian Ridge Regression with iterative imputation",
            "",
            "Each missing value was predicted using all other features as predictors,",
            "iteratively refining predictions until convergence."
        ])
        
        return "\n".join(report_lines)
    
    def get_semantic_summary(self) -> str:
        """Generate markdown summary of semantic type detection"""
        if not self.profiles:
            self.profile_all_columns()
        
        lines = [
            "## Semantic Data Profiling Report",
            "",
            "### Detected Medical/Healthcare Types",
            "",
            "| Column | Semantic Type | Risk Level | Interpretation |",
            "|--------|--------------|------------|----------------|"
        ]
        
        medical_types = []
        other_types = []
        
        for name, profile in self.profiles.items():
            if profile.semantic_type in self.SEMANTIC_RULES:
                medical_types.append(profile)
            else:
                other_types.append(profile)
        
        # Medical types first
        for p in medical_types:
            risk_badge = {"normal": "✓", "warning": "⚠", "critical": "⚠⚠"}[p.risk_level]
            lines.append(f"| {p.name} | {p.semantic_type.value} | {risk_badge} {p.risk_level} | {p.medical_interpretation} |")
        
        # Other types
        for p in other_types:
            lines.append(f"| {p.name} | {p.semantic_type.value} | - | {p.medical_interpretation} |")
        
        # Summary statistics
        lines.extend([
            "",
            "### Data Quality Summary",
            "",
            f"- **Total Columns:** {len(self.profiles)}",
            f"- **Medical Types Detected:** {len(medical_types)}",
            f"- **Columns with Missing Values:** {sum(1 for p in self.profiles.values() if p.missing_count > 0)}",
            f"- **Total Missing Values:** {sum(p.missing_count for p in self.profiles.values())}",
            f"- **Risk Warnings:** {sum(1 for p in self.profiles.values() if p.risk_level in ['warning', 'critical'])}"
        ])
        
        return "\n".join(lines)
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """Validate overall data quality and return scores"""
        if not self.profiles:
            self.profile_all_columns()
        
        total_cells = len(self.df) * len(self.df.columns)
        total_missing = sum(p.missing_count for p in self.profiles.values())
        
        completeness_score = (1 - total_missing / total_cells) * 100
        
        # Uniqueness score (for potential ID columns)
        _uniqueness_issues = sum(
            1
            for p in self.profiles.values()
            if p.semantic_type == SemanticType.IDENTIFIER and p.unique_count < len(self.df)
        )
        
        # Medical validity score
        medical_cols = [p for p in self.profiles.values() if p.semantic_type in self.SEMANTIC_RULES]
        if medical_cols:
            valid_medical = sum(1 for p in medical_cols if p.risk_level == "normal")
            medical_validity_score = valid_medical / len(medical_cols) * 100
        else:
            medical_validity_score = 100
        
        overall_score = (completeness_score * 0.4 + medical_validity_score * 0.6)
        
        return {
            "overall_score": round(overall_score, 1),
            "completeness_score": round(completeness_score, 1),
            "medical_validity_score": round(medical_validity_score, 1),
            "total_missing": total_missing,
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "medical_columns_detected": len(medical_cols),
            "risk_warnings": sum(1 for p in self.profiles.values() if p.risk_level != "normal"),
            "identifier_uniqueness_flags": _uniqueness_issues,
        }


# Convenience function for quick profiling
def quick_profile(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    """
    Quick semantic profiling of a DataFrame
    
    Returns:
        Tuple of (markdown_report, imputed_dataframe)
    """
    informer = AdvancedDataInformer(df)
    informer.profile_all_columns()
    
    # Perform imputation if there are missing values
    if df.isnull().sum().sum() > 0:
        imputed_df = informer.bayesian_iterative_imputation()
        imputation_report = informer.get_imputation_report()
    else:
        imputed_df = df.copy()
        imputation_report = "No missing values detected - no imputation needed."
    
    semantic_report = informer.get_semantic_summary()
    quality = informer.validate_data_quality()
    
    full_report = f"""
{semantic_report}

---

{imputation_report}

---

## Overall Data Quality Score: {quality['overall_score']}%

- Completeness: {quality['completeness_score']}%
- Medical Validity: {quality['medical_validity_score']}%
"""
    
    return full_report, imputed_df
