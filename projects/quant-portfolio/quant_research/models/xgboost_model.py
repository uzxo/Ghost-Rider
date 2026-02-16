"""
Cross-Sectional Gradient Boosting Model
========================================
XGBoost/LightGBM for predicting cross-sectional excess returns.
This is the primary alpha model in the ensemble.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)


class CrossSectionalGBM:
    """
    Cross-sectional gradient boosting model.

    Predicts next-period cross-sectional return rank from
    a multi-layer feature stack.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.feature_names = None
        self.feature_importance = None

    def build_model(self):
        """Initialize the gradient boosting model."""
        try:
            import lightgbm as lgb
            params = self.config.get('lightgbm', {})
            self.model = lgb.LGBMRegressor(
                n_estimators=params.get('n_estimators', 500),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.7),
                reg_alpha=params.get('reg_alpha', 0.1),
                reg_lambda=params.get('reg_lambda', 1.0),
                min_child_samples=params.get('min_child_samples', 20),
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            self._model_type = 'lgbm'
            logger.info("Using LightGBM model")
        except ImportError:
            import xgboost as xgb
            params = self.config.get('xgboost', {})
            self.model = xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 500),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.7),
                reg_alpha=params.get('reg_alpha', 0.1),
                reg_lambda=params.get('reg_lambda', 1.0),
                min_child_weight=params.get('min_child_weight', 10),
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            self._model_type = 'xgb'
            logger.info("Using XGBoost model (LightGBM not available)")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None):
        """
        Train the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features (cross-sectional, multiple dates stacked).
        y_train : pd.Series
            Training target (forward return rank).
        X_val : pd.DataFrame, optional
            Validation features for early stopping.
        y_val : pd.Series, optional
            Validation target.
        """
        if self.model is None:
            self.build_model()

        self.feature_names = list(X_train.columns)

        # Handle missing values
        X_train = X_train.fillna(X_train.median())

        fit_params = {}
        if X_val is not None and y_val is not None:
            X_val = X_val.fillna(X_val.median())
            early_stop = self.config.get(
                self._model_type, {}
            ).get('early_stopping_rounds', 50)

            if self._model_type == 'lgbm':
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['callbacks'] = [
                    __import__('lightgbm').early_stopping(early_stop, verbose=False),
                    __import__('lightgbm').log_evaluation(0),
                ]
            else:
                fit_params['eval_set'] = [(X_val, y_val)]
                fit_params['verbose'] = False

        self.model.fit(X_train, y_train, **fit_params)

        # Store feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=self.feature_names
        ).sort_values(ascending=False)

        logger.info(f"Model trained. Top features: "
                     f"{self.feature_importance.head(5).to_dict()}")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions.

        Parameters
        ----------
        X : pd.DataFrame
            Features for prediction.

        Returns
        -------
        pd.Series
            Predicted return ranks.
        """
        X = X[self.feature_names].fillna(X[self.feature_names].median())
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def get_shap_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute SHAP values for interpretability.

        Returns
        -------
        pd.DataFrame
            SHAP values for each feature and observation.
        """
        try:
            import shap
            X_filled = X[self.feature_names].fillna(
                X[self.feature_names].median()
            )
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_filled)
            return pd.DataFrame(
                shap_values,
                columns=self.feature_names,
                index=X.index
            )
        except ImportError:
            logger.warning("SHAP not installed; skipping SHAP analysis")
            return pd.DataFrame()

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       n_splits: int = 5) -> Dict:
        """
        Time-series cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target.
        n_splits : int
            Number of CV folds.

        Returns
        -------
        dict
            CV metrics (IC, ICIR, RMSE per fold).
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {'ic': [], 'rmse': []}

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            self.build_model()
            self.train(X_tr, y_tr)
            preds = self.predict(X_te)

            # Information Coefficient (Spearman rank correlation)
            from scipy.stats import spearmanr
            ic, _ = spearmanr(y_te, preds)
            metrics['ic'].append(ic)

            # RMSE
            rmse = np.sqrt(np.mean((y_te - preds) ** 2))
            metrics['rmse'].append(rmse)

        metrics['ic_mean'] = np.mean(metrics['ic'])
        metrics['ic_std'] = np.std(metrics['ic'])
        metrics['icir'] = (
            metrics['ic_mean'] / metrics['ic_std']
            if metrics['ic_std'] > 0 else 0
        )

        logger.info(f"CV Results — IC: {metrics['ic_mean']:.4f} "
                     f"+/- {metrics['ic_std']:.4f}, "
                     f"ICIR: {metrics['icir']:.4f}")
        return metrics
