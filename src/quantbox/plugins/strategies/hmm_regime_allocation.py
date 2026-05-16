"""HMM-based Regime Portfolio Allocation.

Fits a 2-3 state Gaussian Hidden Markov Model on asset returns (and optionally
volatility/volume features) to classify the market into regimes (bull/bear/crisis).
Allocation shifts dynamically based on the detected regime.

Core alpha: identifying regime transitions earlier than simple trend/momentum indicators.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbox.contracts import PluginMeta


@dataclass
class HmmRegimeAllocation:
    """HMM latent regime detection with dynamic portfolio allocation.

    Fits a Gaussian HMM on rolling windows of asset features to classify
    market regimes. Each regime maps to a predefined allocation mix.
    Refits periodically and applies transition smoothing to avoid whipsaws.
    """

    meta: PluginMeta = PluginMeta(
        name="strategy.hmm_regime_allocation.v1",
        kind="strategy",
        version="0.1.0",
        core_compat=">=0.1.0",
        description=(
            "HMM regime detection: fits Gaussian HMM on returns/vol/volume to classify "
            "bull/bear/crisis regimes, allocates portfolio accordingly"
        ),
        tags=("crypto", "regime", "hmm", "portfolio-allocation", "daily"),
        capabilities=("backtest", "paper"),
        params_schema={
            "type": "object",
            "properties": {
                "n_regimes": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of HMM hidden states (regimes)",
                },
                "hmm_lookback_days": {
                    "type": "integer",
                    "default": 252,
                    "description": "Rolling window size in days for HMM fitting",
                },
                "refit_frequency_days": {
                    "type": "integer",
                    "default": 21,
                    "description": "How often to refit the HMM (days)",
                },
                "covariance_type": {
                    "type": "string",
                    "default": "full",
                    "description": "HMM covariance type: full, diag, spherical, tied",
                },
                "features": {
                    "type": "array",
                    "default": ["returns", "volatility_20d", "volume_ratio"],
                    "description": "Feature set for HMM observation vectors",
                },
                "regime_allocation": {
                    "type": "object",
                    "default": {
                        "bull": {"equities": 0.7, "bonds": 0.1, "commodities": 0.1, "crypto": 0.1},
                        "bear": {"equities": 0.2, "bonds": 0.5, "commodities": 0.2, "crypto": 0.1},
                        "crisis": {"equities": 0.0, "bonds": 0.7, "commodities": 0.3, "crypto": 0.0},
                    },
                    "description": "Target allocation per regime (by asset class)",
                },
                "transition_smoothing": {
                    "type": "boolean",
                    "default": True,
                    "description": "Apply smoothing to avoid rapid regime flips",
                },
                "min_regime_duration_days": {
                    "type": "integer",
                    "default": 5,
                    "description": "Minimum days before allowing regime transition",
                },
                "n_iterations": {
                    "type": "integer",
                    "default": 100,
                    "description": "Max EM iterations for HMM fitting",
                },
                "random_state": {
                    "type": "integer",
                    "default": 42,
                    "description": "Random seed for reproducibility",
                },
                "volatility_window": {
                    "type": "integer",
                    "default": 20,
                    "description": "Window for rolling volatility feature",
                },
                "volume_ratio_window": {
                    "type": "integer",
                    "default": 20,
                    "description": "Window for volume ratio feature (current / rolling mean)",
                },
                "symbol_asset_class": {
                    "type": "object",
                    "default": {"BTCUSDT": "crypto", "ETHUSDT": "crypto"},
                    "description": "Mapping of symbols to asset classes for allocation",
                },
            },
        },
        inputs=("prices",),
        outputs=("weights",),
        examples=(
            """
strategy:
  plugin: strategy.hmm_regime_allocation.v1
  params:
    n_regimes: 3
    hmm_lookback_days: 252
    refit_frequency_days: 21
    covariance_type: full
    features: [returns, volatility_20d, volume_ratio]
""",
        ),
    )

    def run(self, data: dict, params: dict) -> dict:
        """Compute target weights based on HMM regime detection.

        Args:
            data: Dict containing 'prices' DataFrame (date index x symbol columns).
                  Optionally 'volumes' DataFrame for volume_ratio feature.
            params: Strategy parameters.

        Returns:
            Dict with 'weights' DataFrame (date index x symbol columns),
            'regimes' Series, and diagnostic info.
        """
        try:
            import hmmlearn.hmm  # noqa: F401
        except ImportError:
            return self._fallback_run(data, params, error="hmmlearn not installed; pip install hmmlearn")

        # Extract parameters
        n_regimes = params.get("n_regimes", 3)
        hmm_lookback = params.get("hmm_lookback_days", 252)
        refit_freq = params.get("refit_frequency_days", 21)
        cov_type = params.get("covariance_type", "full")
        features_list = params.get("features", ["returns", "volatility_20d", "volume_ratio"])
        regime_alloc = params.get(
            "regime_allocation", self.meta.params_schema["properties"]["regime_allocation"]["default"]
        )
        smoothing = params.get("transition_smoothing", True)
        min_duration = params.get("min_regime_duration_days", 5)
        n_iter = params.get("n_iterations", 100)
        seed = params.get("random_state", 42)
        vol_window = params.get("volatility_window", 20)
        vol_ratio_window = params.get("volume_ratio_window", 20)
        symbol_class_map = params.get("symbol_asset_class", {"BTCUSDT": "crypto", "ETHUSDT": "crypto"})

        prices_df = data["prices"]
        volumes_df = data.get("volumes")

        # Use first symbol as the regime indicator (BTC as market proxy)
        regime_symbol = prices_df.columns[0]
        regime_prices = prices_df[regime_symbol].dropna()

        if len(regime_prices) < hmm_lookback + vol_window + 1:
            weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
            return {"weights": weights, "regimes": pd.Series(dtype=int), "error": "Insufficient data"}

        # Build feature matrix
        features_data = self._build_features(
            regime_prices, volumes_df, regime_symbol, features_list, vol_window, vol_ratio_window
        )

        # Align all data to feature availability
        valid_idx = features_data.dropna().index
        features_clean = features_data.loc[valid_idx].values

        # Run HMM with periodic refitting
        regime_series = self._fit_and_predict(
            features_clean, valid_idx, n_regimes, hmm_lookback, refit_freq, cov_type, n_iter, seed
        )

        # Label regimes by mean return (highest mean return = bull, lowest = crisis)
        regime_labels = self._label_regimes(regime_series, regime_prices, n_regimes)

        # Apply transition smoothing
        if smoothing:
            regime_labels = self._smooth_regimes(regime_labels, min_duration)

        # Map regimes to allocations
        weights = self._allocate(regime_labels, regime_alloc, prices_df, symbol_class_map)

        return {
            "weights": weights,
            "regimes": regime_labels,
            "n_refits": max(1, (len(valid_idx) - hmm_lookback) // refit_freq),
            "regime_counts": regime_labels.value_counts().to_dict() if len(regime_labels) > 0 else {},
        }

    def _build_features(
        self,
        prices: pd.Series,
        volumes_df: pd.DataFrame | None,
        symbol: str,
        features_list: list[str],
        vol_window: int,
        vol_ratio_window: int,
    ) -> pd.DataFrame:
        """Build observation feature matrix for HMM."""
        feats = pd.DataFrame(index=prices.index)

        if "returns" in features_list:
            feats["returns"] = prices.pct_change()

        if "volatility_20d" in features_list:
            feats["volatility_20d"] = prices.pct_change().rolling(vol_window).std()

        if "volume_ratio" in features_list and volumes_df is not None and symbol in volumes_df.columns:
            vol = volumes_df[symbol]
            feats["volume_ratio"] = vol / vol.rolling(vol_ratio_window).mean()

        # If volume not available, drop that feature silently
        if "volume_ratio" in features_list and "volume_ratio" not in feats.columns:
            pass  # proceed without it

        return feats

    def _fit_and_predict(
        self,
        features: np.ndarray,
        index: pd.DatetimeIndex,
        n_regimes: int,
        lookback: int,
        refit_freq: int,
        cov_type: str,
        n_iter: int,
        seed: int,
    ) -> pd.Series:
        """Fit HMM on rolling windows and predict regimes."""
        from hmmlearn.hmm import GaussianHMM

        regimes = pd.Series(np.nan, index=index, dtype=float)
        n = len(features)

        if n < lookback:
            return regimes

        last_model = None
        last_fit_idx = 0

        for i in range(lookback, n):
            # Refit periodically
            if last_model is None or (i - last_fit_idx) >= refit_freq:
                window = features[max(0, i - lookback) : i]
                try:
                    model = GaussianHMM(
                        n_components=n_regimes,
                        covariance_type=cov_type,
                        n_iter=n_iter,
                        random_state=seed,
                    )
                    model.fit(window)
                    last_model = model
                    last_fit_idx = i
                except Exception:
                    # HMM convergence failure — keep last model
                    if last_model is None:
                        continue

            # Predict current regime using recent observations
            try:
                recent = features[max(0, i - lookback) : i + 1]
                predicted = last_model.predict(recent)
                regimes.iloc[i] = predicted[-1]
            except Exception:
                pass

        return regimes

    def _label_regimes(self, regimes: pd.Series, prices: pd.Series, n_regimes: int) -> pd.Series:
        """Label numeric regimes as bull/bear/crisis based on mean returns per state."""
        returns = prices.pct_change()
        valid = regimes.dropna()

        if len(valid) == 0:
            return regimes.map(lambda x: "crisis")

        # Compute mean return per regime
        regime_returns = {}
        for r in range(n_regimes):
            mask = valid == r
            if mask.any():
                aligned_returns = returns.reindex(valid[mask].index)
                regime_returns[r] = aligned_returns.mean()
            else:
                regime_returns[r] = 0.0

        # Sort by mean return: highest = bull, lowest = crisis
        sorted_regimes = sorted(regime_returns.keys(), key=lambda r: regime_returns[r], reverse=True)

        label_map = {}
        labels = ["bull", "bear", "crisis"][:n_regimes]
        for i, r in enumerate(sorted_regimes):
            label_map[r] = labels[i] if i < len(labels) else f"regime_{r}"

        return regimes.map(label_map)

    def _smooth_regimes(self, regimes: pd.Series, min_duration: int) -> pd.Series:
        """Enforce minimum regime duration to avoid rapid switching."""
        smoothed = regimes.copy()
        current_regime = None
        current_start = 0

        for i in range(len(smoothed)):
            val = smoothed.iloc[i]
            if pd.isna(val):
                continue

            if val != current_regime:
                if current_regime is not None and (i - current_start) < min_duration:
                    # Revert short regime back to previous
                    smoothed.iloc[current_start:i] = smoothed.iloc[current_start - 1] if current_start > 0 else val
                current_regime = val
                current_start = i

        return smoothed

    def _allocate(
        self,
        regimes: pd.Series,
        regime_alloc: dict,
        prices_df: pd.DataFrame,
        symbol_class_map: dict[str, str],
    ) -> pd.DataFrame:
        """Convert regime labels to per-symbol weights based on asset class allocation."""
        weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)

        for idx in regimes.dropna().index:
            regime = regimes[idx]
            alloc = regime_alloc.get(regime, {})

            for col in prices_df.columns:
                col_upper = col.upper() if isinstance(col, str) else str(col)
                asset_class = symbol_class_map.get(col_upper, symbol_class_map.get(col, "crypto"))
                class_weight = alloc.get(asset_class, 0.0)

                # Split class weight evenly among symbols of same class
                n_same_class = sum(
                    1
                    for c in prices_df.columns
                    if symbol_class_map.get(
                        c.upper() if isinstance(c, str) else str(c), symbol_class_map.get(c, "crypto")
                    )
                    == asset_class
                )
                weights.loc[idx, col] = class_weight / max(n_same_class, 1)

        return weights

    def _fallback_run(self, data: dict, params: dict, error: str) -> dict:
        """Return zero weights with error when dependencies are missing."""
        prices_df = data["prices"]
        weights = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        return {"weights": weights, "regimes": pd.Series(dtype=object), "error": error}
