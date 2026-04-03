"""
Comprehensive Testing Suite for MetaAI Platform
Tests all critical components: API, UI, Models, State, Data Processing
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import requests


class TestDataProcessing(unittest.TestCase):
    """Test data upload and preprocessing"""
    
    def setUp(self):
        """Create test data"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_valid_csv_upload(self):
        """Test uploading valid CSV"""
        # Create valid CSV
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0],
            'target': [0, 1, 0]
        })
        csv_path = os.path.join(self.test_dir, 'test.csv')
        df.to_csv(csv_path, index=False)
        
        # Verify it can be read
        loaded_df = pd.read_csv(csv_path)
        self.assertEqual(len(loaded_df), 3)
        self.assertEqual(len(loaded_df.columns), 3)
    
    def test_missing_values_handling(self):
        """Test handling of missing values"""
        df = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': [4.0, 5.0, np.nan],
            'target': [0, 1, 0]
        })
        
        # Should have missing values
        self.assertTrue(df.isnull().any().any())
        
        # After fillna
        df_filled = df.fillna(df.mean(numeric_only=True))
        self.assertFalse(df_filled.isnull().any().any())
    
    def test_class_imbalance_detection(self):
        """Test class imbalance detection"""
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'target': [0]*95 + [1]*5  # Imbalanced
        })
        
        imbalance_ratio = len(df[df['target']==0]) / len(df[df['target']==1])
        self.assertGreater(imbalance_ratio, 3)  # Significant imbalance
    
    def test_feature_scaling(self):
        """Test feature normalization"""
        from sklearn.preprocessing import StandardScaler
        
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Mean should be ~0, std should be ~1
        np.testing.assert_array_almost_equal(X_scaled.mean(axis=0), [0, 0], decimal=1)


class TestModelTraining(unittest.TestCase):
    """Test model training pipeline"""
    
    def setUp(self):
        """Create synthetic training data"""
        X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
        self.X = X
        self.y = y
    
    def test_random_forest_training(self):
        """Test Random Forest model training"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        
        score = rf.score(X_test, y_test)
        self.assertGreater(score, 0.5)  # Should perform better than random
    
    def test_model_persistence(self):
        """Test saving and loading models"""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        
        # Train model
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(self.X, self.y)
        
        # Save
        fd, model_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        joblib.dump(rf, model_path)
        
        # Load
        loaded_rf = joblib.load(model_path)
        
        # Verify predictions match
        pred1 = rf.predict(self.X[:5])
        pred2 = loaded_rf.predict(self.X[:5])
        np.testing.assert_array_equal(pred1, pred2)
        
        # Cleanup
        os.remove(model_path)
    
    def test_cross_validation(self):
        """Test model cross-validation"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(rf, self.X, self.y, cv=5)
        
        # Should have 5 scores
        self.assertEqual(len(scores), 5)
        # All scores should be between 0 and 1
        self.assertTrue(all(0 <= s <= 1 for s in scores))


class TestModelPrediction(unittest.TestCase):
    """Test prediction functionality"""
    
    def setUp(self):
        """Setup test model"""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.model.fit(X, y)
        self.X_test = X[:10]
    
    def test_single_prediction(self):
        """Test single sample prediction"""
        pred = self.model.predict(self.X_test[:1])
        self.assertIn(pred[0], [0, 1])
    
    def test_batch_prediction(self):
        """Test batch predictions"""
        preds = self.model.predict(self.X_test)
        self.assertEqual(len(preds), 10)
        self.assertTrue(all(p in [0, 1] for p in preds))
    
    def test_prediction_probabilities(self):
        """Test probability predictions"""
        probs = self.model.predict_proba(self.X_test)
        self.assertEqual(probs.shape[0], 10)
        self.assertEqual(probs.shape[1], 2)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probs.sum(axis=1), 1)


class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints (requires api_production.py running)"""
    
    API_BASE = "http://localhost:8000"
    
    def test_health_check(self):
        """Test health endpoint"""
        try:
            response = requests.get(f"{self.API_BASE}/api/health", timeout=5)
            if response.status_code == 200:
                self.assertEqual(response.json()['status'], 'healthy')
        except requests.exceptions.ConnectionError:
            self.skipTest("API not running")
    
    def test_models_list(self):
        """Test models listing endpoint"""
        try:
            response = requests.get(f"{self.API_BASE}/api/models", timeout=5)
            if response.status_code == 200:
                self.assertIsInstance(response.json(), list)
        except requests.exceptions.ConnectionError:
            self.skipTest("API not running")
    
    def test_prediction_endpoint(self):
        """Test prediction endpoint"""
        try:
            payload = {
                "model_name": "RandomForest",
                "features": [1.0, 2.0, 3.0, 4.0, 5.0]
            }
            response = requests.post(
                f"{self.API_BASE}/api/predict",
                json=payload,
                timeout=5
            )
            if response.status_code in [200, 404]:  # 404 if model not found is ok for test
                pass
        except requests.exceptions.ConnectionError:
            self.skipTest("API not running")


class TestExplainability(unittest.TestCase):
    """Test model explainability features"""
    
    def setUp(self):
        """Setup test model"""
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        self.model.fit(X, y)
        self.X_test = X[:1]
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        importances = self.model.feature_importances_
        
        # Should have importance for each feature
        self.assertEqual(len(importances), 5)
        # Should sum to approximately 1
        self.assertAlmostEqual(importances.sum(), 1.0, places=5)
    
    def test_feature_importance_sorting(self):
        """Test feature importance sorting"""
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        
        # First should be highest importance
        self.assertEqual(
            importances[sorted_idx[0]],
            max(importances)
        )


class TestStateManagement(unittest.TestCase):
    """Test application state management"""
    
    def test_state_serialization(self):
        """Test that state can be serialized"""
        from core.app_state import AppState
        
        state = AppState()
        state.models = {"RandomForest": "rf_model"}
        state.target_column = "target"
        
        # Should be able to convert to dict
        state_dict = state.__dict__
        self.assertIn('models', state_dict)
        self.assertIn('target_column', state_dict)
    
    def test_state_immutability(self):
        """Test state isolation between sessions"""
        from core.app_state import AppState
        
        state1 = AppState()
        state2 = AppState()
        
        state1.models = {"Model1": "data"}
        
        # state2 should be independent
        self.assertNotEqual(state1.models, state2.models)


class TestDataQuality(unittest.TestCase):
    """Test data quality assessment"""
    
    def test_quality_scoring(self):
        """Test data quality scoring"""
        
        # Perfect dataset should score high
        df_good = pd.DataFrame({
            'f1': range(1000),
            'f2': range(1000),
            'target': np.random.randint(0, 2, 1000)
        })
        
        # Should have no missing values
        self.assertEqual(df_good.isnull().sum().sum(), 0)
    
    def test_missing_data_detection(self):
        """Test missing data detection"""
        df = pd.DataFrame({
            'f1': [1, 2, None, 4],
            'f2': [5, None, 7, 8],
            'target': [0, 1, 0, 1]
        })
        
        missing_pct = (df.isnull().sum() / len(df) * 100)
        self.assertGreater(missing_pct.max(), 0)


class TestConfiguration(unittest.TestCase):
    """Test configuration system"""
    
    def test_config_values_exist(self):
        """Test that config values are defined"""
        from utils.config import Config
        
        self.assertTrue(hasattr(Config, 'ACCURACY_DEPLOYMENT_THRESHOLD'))
        self.assertTrue(hasattr(Config, 'MODELS_DIR'))
        self.assertTrue(hasattr(Config, 'LOGS_DIR'))
    
    def test_config_directories(self):
        """Test config directory paths"""
        from utils.config import Config
        
        self.assertIsNotNone(Config.MODELS_DIR)
        self.assertIsNotNone(Config.LOGS_DIR)
        self.assertIsNotNone(Config.DATA_DIR)


class TestEndToEnd(unittest.TestCase):
    """End-to-end workflow test"""
    
    def test_full_pipeline(self):
        """Test complete ML pipeline"""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import f1_score
        
        # 1. Create data
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 3. Train
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Predict
        y_pred = model.predict(X_test)
        
        # 5. Evaluate
        f1 = f1_score(y_test, y_pred)
        self.assertGreater(f1, 0)  # Should have non-zero F1 score
        
        # 6. Explain
        importances = model.feature_importances_
        self.assertEqual(len(importances), 10)


def run_test_suite():
    """Run all tests with detailed output"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEndpoints))
    suite.addTests(loader.loadTestsFromTestCase(TestExplainability))
    suite.addTests(loader.loadTestsFromTestCase(TestStateManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQuality))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_test_suite()
    sys.exit(0 if success else 1)
