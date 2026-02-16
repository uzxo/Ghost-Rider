"""
Penalized Linear Model (Elastic Net)
======================================
Serves as the irreducible linear baseline.
If the ensemble cannot beat this, non-linear models are overfitting.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class LinearAlphaModel:
    """
    Elastic Net / Ridge for cross-sectional return prediction.

    This is the baseline model. If XGBoost or deep learning
    cannot beat this, the non-linear complexity is unjustified.
    """

    def __init__(self, config: dict):
        self.config = config.get('elastic_net', {})
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def build_model(self):
        """Initialize the linear model."""
        self.model = ElasticNet(
            alpha=self.config.get('alpha', 0.1),
            l1_ratio=self.config.get('l1_ratio', 0.5),
            max_iter=self.config.get('max_iter', 5000),
            tol=self.config.get('tol', 1e-4),
            random_state=42,
        )
        logger.info("ElasticNet model initialized")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None):
        """
        Train the linear model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training target.
        """
        if self.model is None:
            self.build_model()

        self.feature_names = list(X_train.columns)

        # Fill NaN and scale
        X_filled = X_train.fillna(X_train.median())
        X_scaled = self.scaler.fit_transform(X_filled)

        self.model.fit(X_scaled, y_train)

        # Log non-zero coefficients
        coefs = pd.Series(self.model.coef_, index=self.feature_names)
        nonzero = coefs[coefs.abs() > 1e-6].sort_values(ascending=False)
        logger.info(f"ElasticNet: {len(nonzero)}/{len(coefs)} "
                     f"non-zero coefficients")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions."""
        X_filled = X[self.feature_names].fillna(
            X[self.feature_names].median()
        )
        X_scaled = self.scaler.transform(X_filled)
        preds = self.model.predict(X_scaled)
        return pd.Series(preds, index=X.index)

    def get_coefficients(self) -> pd.Series:
        """Get model coefficients."""
        if self.model is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.coef_,
            index=self.feature_names
        ).sort_values(ascending=False)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5) -> Dict:
        """Time-series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {'ic': [], 'rmse': []}

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            self.build_model()
            self.train(X_tr, y_tr)
            preds = self.predict(X_te)

            ic, _ = spearmanr(y_te, preds)
            metrics['ic'].append(ic)

            rmse = np.sqrt(np.mean((y_te - preds) ** 2))
            metrics['rmse'].append(rmse)

        metrics['ic_mean'] = np.mean(metrics['ic'])
        metrics['ic_std'] = np.std(metrics['ic'])
        metrics['icir'] = (
            metrics['ic_mean'] / metrics['ic_std']
            if metrics['ic_std'] > 0 else 0
        )
        return metrics
