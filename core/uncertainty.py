import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class ConformalPredictor(BaseEstimator, ClassifierMixin):
    """
    Implements Split Conformal Prediction for classification.
    
    This wrapper takes a trained base classifier and a calibration dataset to 
    compute a threshold 'q' such that the prediction sets generated at test time 
    contain the true label with probability 1 - alpha.
    
    Paper Reference: "A Tutorial on Conformal Prediction" (Shafer & Vovk, 2008)
    """
    def __init__(self, base_estimator, alpha=0.05):
        """
        Args:
            base_estimator: A fitted sklearn-compatible classifier with predict_proba.
            alpha (float): Error rate target (e.g., 0.05 for 95% coverage).
        """
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.q_hat = None
        self.calibration_scores = None

    def fit(self, X_cal, y_cal):
        """
        Calibrate the predictor using a hold-out calibration set.
        
        Args:
            X_cal: Calibration features.
            y_cal: Calibration true labels.
        """
        # Get probability of the true class
        probs = self.base_estimator.predict_proba(X_cal)
        n = len(y_cal)
        
        # Conformity score: 1 - softmax score of the true class
        # (Standard heuristic for classification)
        true_class_probs = probs[np.arange(n), y_cal]
        scores = 1 - true_class_probs
        
        # Calculate the (1-alpha) quantile of the scores
        # We use a finite sample correction: (n+1)(1-alpha)/n roughly
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        
        self.q_hat = np.quantile(scores, q_level, method='higher')
        self.calibration_scores = scores # Store for analysis
        
        return self

    def predict(self, X):
        """
        Generate prediction sets for new data.
        
        Returns:
            prediction_sets: List of lists, where each inner list contains 
                             the class indices included in the prediction set.
        """
        if self.q_hat is None:
            raise ValueError("ConformalPredictor not calibrated. Call fit(X_cal, y_cal) first.")
            
        probs = self.base_estimator.predict_proba(X)
        
        # A class 'k' is included if: 1 - prob[k] <= q_hat
        # Which simplifies to: prob[k] >= 1 - q_hat
        threshold = 1 - self.q_hat
        
        prediction_sets = []
        for i in range(len(probs)):
            # classes where probability is high enough
            included_classes = np.where(probs[i] >= threshold)[0]
            
            # If empty set (rare but possible with this score function), 
            # include the class with max probability
            if len(included_classes) == 0:
                included_classes = [np.argmax(probs[i])]
                
            prediction_sets.append(included_classes.tolist())
            
        return prediction_sets

    def evaluate_coverage(self, X_test, y_test):
        """
        Calculate the empirical coverage and average set size.
        """
        sets = self.predict(X_test)
        coverage = np.mean([y in s for y, s in zip(y_test, sets)])
        avg_size = np.mean([len(s) for s in sets])
        
        return {
            "target_coverage": 1 - self.alpha,
            "empirical_coverage": coverage,
            "average_set_size": avg_size
        }
