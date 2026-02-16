"""
Hidden Markov Model for Regime Detection
==========================================
Classifies market regimes using macro/market features.
Regime classification conditions other models' predictions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeHMM:
    """
    Hidden Markov Model for market regime detection.

    Identifies 3-4 latent states from market observables
    (VIX, yield curve, credit spreads, market breadth).
    """

    def __init__(self, config: dict):
        hmm_config = config.get('hmm', {})
        self.n_states = hmm_config.get('n_states', 4)
        self.n_iter = hmm_config.get('n_iter', 200)
        self.covariance_type = hmm_config.get('covariance_type', 'full')
        self.feature_cols = hmm_config.get('features', [
            'vix', 'yield_curve_2s10s', 'credit_spread_hy', 'sp500_ret_21d'
        ])
        self.model = None
        self.scaler = StandardScaler()
        self._fitted = False

    def build_model(self):
        """Initialize the HMM."""
        try:
            from hmmlearn.hmm import GaussianHMM
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=42,
            )
            logger.info(f"HMM initialized with {self.n_states} states")
        except ImportError:
            logger.warning("hmmlearn not installed; using fallback regime detection")
            self.model = None

    def train(self, macro_data: pd.DataFrame):
        """
        Fit the HMM on macro features.

        Parameters
        ----------
        macro_data : pd.DataFrame
            Macro data with columns matching self.feature_cols.
        """
        available_cols = [c for c in self.feature_cols if c in macro_data.columns]
        if len(available_cols) < 2:
            logger.warning("Insufficient macro features for HMM; "
                           "using rule-based fallback")
            self._fitted = False
            return

        X = macro_data[available_cols].dropna()
        if len(X) < 100:
            logger.warning("Insufficient data for HMM training")
            self._fitted = False
            return

        X_scaled = self.scaler.fit_transform(X)

        if self.model is None:
            self.build_model()

        if self.model is not None:
            try:
                self.model.fit(X_scaled)
                self._fitted = True
                logger.info("HMM fitted successfully")

                # Log state properties
                for i in range(self.n_states):
                    means = self.model.means_[i]
                    logger.info(f"  State {i}: means = "
                                 f"{dict(zip(available_cols, means.round(2)))}")
            except Exception as e:
                logger.error(f"HMM fitting failed: {e}")
                self._fitted = False

    def predict_regime(self, macro_data: pd.DataFrame) -> pd.Series:
        """
        Predict the current regime.

        Parameters
        ----------
        macro_data : pd.DataFrame
            Macro features.

        Returns
        -------
        pd.Series
            Regime labels (0 to n_states-1).
        """
        if not self._fitted or self.model is None:
            return self._fallback_regime(macro_data)

        available_cols = [c for c in self.feature_cols if c in macro_data.columns]
        X = macro_data[available_cols].dropna()
        X_scaled = self.scaler.transform(X)

        try:
            states = self.model.predict(X_scaled)
            return pd.Series(states, index=X.index, name='regime')
        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            return self._fallback_regime(macro_data)

    def get_regime_probabilities(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Get posterior probabilities of each regime."""
        if not self._fitted or self.model is None:
            return pd.DataFrame()

        available_cols = [c for c in self.feature_cols if c in macro_data.columns]
        X = macro_data[available_cols].dropna()
        X_scaled = self.scaler.transform(X)

        try:
            probs = self.model.predict_proba(X_scaled)
            return pd.DataFrame(
                probs,
                index=X.index,
                columns=[f'regime_{i}_prob' for i in range(self.n_states)]
            )
        except Exception:
            return pd.DataFrame()

    def get_transition_matrix(self) -> np.ndarray:
        """Get the state transition probability matrix."""
        if self._fitted and self.model is not None:
            return self.model.transmat_
        return np.eye(self.n_states) / self.n_states

    def _fallback_regime(self, macro_data: pd.DataFrame) -> pd.Series:
        """Rule-based regime fallback when HMM unavailable."""
        regime = pd.Series(0, index=macro_data.index, name='regime')

        if 'vix' in macro_data.columns:
            high_vol = macro_data['vix'] > 25
        else:
            high_vol = pd.Series(False, index=macro_data.index)

        if 'sp500_ret_63d' in macro_data.columns:
            bear = macro_data['sp500_ret_63d'] < -0.05
        elif 'sp500_ret_21d' in macro_data.columns:
            bear = macro_data['sp500_ret_21d'] < -0.03
        else:
            bear = pd.Series(False, index=macro_data.index)

        regime = np.where(
            ~bear & ~high_vol, 0,  # Bull / Low Vol
            np.where(
                ~bear & high_vol, 1,  # Bull / High Vol
                np.where(
                    bear & ~high_vol, 2,  # Bear / Low Vol
                    3  # Bear / High Vol
                )
            )
        )
        return pd.Series(regime, index=macro_data.index, name='regime')
