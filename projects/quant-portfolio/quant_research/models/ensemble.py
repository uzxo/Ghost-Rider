"""
Ensemble Meta-Model
====================
Combines predictions from multiple base models using
a second-stage model trained on out-of-fold predictions.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from scipy.stats import spearmanr
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Stacked ensemble combining multiple base model predictions.

    The meta-model learns optimal combination weights from
    out-of-fold predictions, optionally conditioned on regime.
    """

    def __init__(self, config: dict):
        ensemble_config = config.get('ensemble', {})
        self.method = ensemble_config.get('method', 'ridge')
        self.alpha = ensemble_config.get('alpha', 1.0)
        self.use_regime = ensemble_config.get('use_regime_conditioning', True)
        self.meta_model = None
        self.model_names = []

    def train(self, predictions: Dict[str, pd.Series],
              actuals: pd.Series,
              regime: Optional[pd.Series] = None):
        """
        Train the meta-model on base model predictions.

        Parameters
        ----------
        predictions : dict
            {model_name: pd.Series of predictions} for each base model.
        actuals : pd.Series
            Actual target values.
        regime : pd.Series, optional
            Regime labels for regime conditioning.
        """
        self.model_names = list(predictions.keys())

        # Build feature matrix from base model predictions
        X = pd.DataFrame(predictions)
        X = X.reindex(actuals.index).dropna()
        y = actuals.reindex(X.index)

        if self.use_regime and regime is not None:
            regime_aligned = regime.reindex(X.index)
            # Add regime dummies
            for state in regime_aligned.unique():
                if not pd.isna(state):
                    X[f'regime_{int(state)}'] = (
                        regime_aligned == state
                    ).astype(float)
                    # Interaction terms
                    for model_name in self.model_names:
                        X[f'{model_name}_x_regime_{int(state)}'] = (
                            X[model_name] * X[f'regime_{int(state)}']
                        )

        if self.method == 'ridge':
            self.meta_model = Ridge(alpha=self.alpha)
            self.meta_model.fit(X.fillna(0), y)
        elif self.method == 'linear':
            from sklearn.linear_model import LinearRegression
            self.meta_model = LinearRegression()
            self.meta_model.fit(X.fillna(0), y)
        else:
            # Default: equal weight
            self.meta_model = None

        # Log model weights
        if self.meta_model is not None and hasattr(self.meta_model, 'coef_'):
            weights = pd.Series(
                self.meta_model.coef_[:len(self.model_names)],
                index=self.model_names
            )
            logger.info(f"Ensemble weights: {weights.to_dict()}")

        # Evaluate ensemble IC
        ensemble_pred = self.predict(predictions, regime)
        ic, _ = spearmanr(actuals.reindex(ensemble_pred.index),
                          ensemble_pred)
        logger.info(f"Ensemble training IC: {ic:.4f}")

    def predict(self, predictions: Dict[str, pd.Series],
                regime: Optional[pd.Series] = None) -> pd.Series:
        """
        Generate ensemble predictions.

        Parameters
        ----------
        predictions : dict
            {model_name: pd.Series of predictions} for each base model.
        regime : pd.Series, optional
            Regime labels.

        Returns
        -------
        pd.Series
            Combined prediction.
        """
        X = pd.DataFrame(predictions)

        if self.use_regime and regime is not None:
            regime_aligned = regime.reindex(X.index)
            for state in regime_aligned.unique():
                if not pd.isna(state):
                    X[f'regime_{int(state)}'] = (
                        regime_aligned == state
                    ).astype(float)
                    for model_name in self.model_names:
                        if model_name in X.columns:
                            X[f'{model_name}_x_regime_{int(state)}'] = (
                                X[model_name] * X[f'regime_{int(state)}']
                            )

        if self.meta_model is not None:
            # Ensure columns match training
            for col in self.meta_model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.meta_model.feature_names_in_]
            preds = self.meta_model.predict(X.fillna(0))
            return pd.Series(preds, index=X.index)
        else:
            # Equal weight fallback
            base_preds = pd.DataFrame({
                name: predictions[name] for name in self.model_names
            })
            return base_preds.mean(axis=1)

    def get_model_contributions(self, predictions: Dict[str, pd.Series],
                                 regime: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Decompose ensemble prediction into model contributions.

        Returns
        -------
        pd.DataFrame
            Contribution of each base model to the ensemble signal.
        """
        if self.meta_model is None or not hasattr(self.meta_model, 'coef_'):
            return pd.DataFrame()

        weights = self.meta_model.coef_[:len(self.model_names)]
        contributions = pd.DataFrame()
        for i, name in enumerate(self.model_names):
            if name in predictions:
                contributions[name] = predictions[name] * weights[i]

        return contributions
