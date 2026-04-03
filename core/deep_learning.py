
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: Torch/TabNet not available or failed to initialize: {e}")
    TORCH_AVAILABLE = False

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class TabularDeepLearningSpecs(BaseEstimator, ClassifierMixin):
    """
    Wrapper for PyTorch TabNet to make it compatible with Scikit-Learn pipelines
    and our existing model_training.py structure.
    """
    def __init__(self, n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2, n_shared=2,
                 lambda_sparse=1e-3, optimizer_fn=None,
                 optimizer_params=dict(lr=2e-2),
                 scheduler_params=dict(step_size=10, gamma=0.9),
                 scheduler_fn=None,
                 max_epochs=20, patience=5, batch_size=1024,
                 virtual_batch_size=128, device_name='auto'):
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lambda_sparse = lambda_sparse
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.device_name = device_name
        self.model = None

        if TORCH_AVAILABLE:
            self.optimizer_fn = optimizer_fn or torch.optim.Adam
            self.scheduler_fn = scheduler_fn or torch.optim.lr_scheduler.StepLR
        else:
            self.optimizer_fn = optimizer_fn
            self.scheduler_fn = scheduler_fn

    def fit(self, X, y, eval_set=None):
        if not TORCH_AVAILABLE:
            raise ValueError("TabNet requires PyTorch, which failed to initialize.")
            
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        self.classes_ = np.unique(y)
        
        self.model = TabNetClassifier(
            n_d=self.n_d, 
            n_a=self.n_a, 
            n_steps=self.n_steps, 
            gamma=self.gamma,
            n_independent=self.n_independent, 
            n_shared=self.n_shared,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            device_name=self.device_name,
            verbose=1
        )
        
        # Split for internal validation if eval_set is None
        if eval_set is None:
            # We use a small chunk of X for early stopping validation to prevent overfitting
            # In a real pipeline, we might want to handle this outside, but this wrapper adapts to fit() API
            from sklearn.model_selection import train_test_split
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=42)
            eval_set = [(X_train, y_train), (X_valid, y_valid)]
        else:
            X_train, y_train = X, y

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size, 
            virtual_batch_size=self.virtual_batch_size,
            num_workers=0,
            drop_last=False
        )
        return self

    def predict(self, X):
        check_is_fitted(self, ['model'])
        X = check_array(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, ['model'])
        X = check_array(X)
        return self.model.predict_proba(X)
