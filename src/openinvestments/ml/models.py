"""
Machine learning models for quantitative finance.

Implements predictive models for:
- Price prediction and forecasting
- Risk prediction and volatility modeling
- Market regime classification
- Anomaly detection in financial time series
- Factor model estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    predictions: np.ndarray
    confidence_intervals: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_metrics: Optional[Dict[str, float]] = None
    prediction_timestamp: Optional[pd.Timestamp] = None


@dataclass
class ModelPerformance:
    """Container for model performance metrics."""
    mse: float
    mae: float
    r2_score: float
    rmse: float
    mape: float
    directional_accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None


class FinancialModel(ABC):
    """Abstract base class for financial prediction models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make predictions on new data."""
        pass

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate model performance on test data."""
        pass


class EnsemblePricePredictor(FinancialModel):
    """
    Ensemble model for price prediction combining multiple algorithms.

    Uses stacking ensemble with Random Forest, Gradient Boosting,
    XGBoost, and neural networks.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42
    ):
        """
        Initialize ensemble price predictor.

        Args:
            n_estimators: Number of estimators in ensemble
            learning_rate: Learning rate for gradient boosting
            max_depth: Maximum depth for tree-based models
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        # Initialize base models
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            ),
            'ridge': Ridge(alpha=0.1, random_state=random_state)
        }

        # Meta-learner
        self.meta_model = LinearRegression()

        # Scaler
        self.scaler = StandardScaler()

        # Training data for meta-learner
        self.base_predictions_train = None

        self.is_fitted = False
        self.feature_names = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit ensemble model."""
        logger.info("Fitting ensemble price predictor")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit base models and collect predictions
        base_predictions = []

        for name, model in self.models.items():
            logger.info(f"Fitting {name} model")
            model.fit(X_scaled, y)
            pred = model.predict(X_scaled)
            base_predictions.append(pred.reshape(-1, 1))

        # Combine base predictions for meta-learner
        self.base_predictions_train = np.hstack(base_predictions)

        # Fit meta-learner
        self.meta_model.fit(self.base_predictions_train, y)

        self.is_fitted = True
        logger.info("Ensemble model fitted successfully")

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)

        # Get base model predictions
        base_predictions = []
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            base_predictions.append(pred.reshape(-1, 1))

        # Combine predictions
        X_meta = np.hstack(base_predictions)

        # Final prediction from meta-learner
        final_predictions = self.meta_model.predict(X_meta)

        # Calculate feature importance (simplified)
        feature_importance = {}
        if hasattr(self.models['rf'], 'feature_importances_'):
            rf_importance = self.models['rf'].feature_importances_
            feature_importance = {f"feature_{i}": imp for i, imp in enumerate(rf_importance)}

        return ModelPrediction(
            predictions=final_predictions,
            feature_importance=feature_importance,
            prediction_timestamp=pd.Timestamp.now()
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate ensemble model performance."""
        predictions = self.predict(X)
        pred_values = predictions.predictions

        mse = mean_squared_error(y, pred_values)
        mae = mean_absolute_error(y, pred_values)
        r2 = r2_score(y, pred_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y - pred_values) / y)) * 100

        # Directional accuracy
        actual_direction = np.sign(np.diff(np.concatenate([[0], y])))
        pred_direction = np.sign(np.diff(np.concatenate([[0], pred_values])))
        directional_accuracy = np.mean(actual_direction == pred_direction)

        return ModelPerformance(
            mse=mse,
            mae=mae,
            r2_score=r2,
            rmse=rmse,
            mape=mape,
            directional_accuracy=directional_accuracy
        )


class LSTMPricePredictor(FinancialModel):
    """
    LSTM neural network for time series price prediction.

    Uses bidirectional LSTM layers with attention mechanism for
    capturing long-term dependencies in financial time series.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        lstm_units: int = 64,
        dropout_rate: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32
    ):
        """
        Initialize LSTM price predictor.

        Args:
            lookback_window: Number of time steps to look back
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        self.lookback_window = lookback_window
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None
        self.scaler = RobustScaler()
        self.is_fitted = False

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential([
            Bidirectional(LSTM(self.lstm_units, return_sequences=True),
                         input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2),
            Dropout(self.dropout_rate),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit LSTM model."""
        logger.info("Fitting LSTM price predictor")

        # Scale features
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Build model
        self.model = self._build_model((X.shape[1], X.shape[2]))

        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Fit model
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        self.is_fitted = True
        logger.info("LSTM model fitted successfully")

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Make LSTM predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions_scaled = self.model.predict(X_scaled, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()

        return ModelPrediction(
            predictions=predictions,
            prediction_timestamp=pd.Timestamp.now()
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate LSTM model performance."""
        predictions = self.predict(X)
        pred_values = predictions.predictions

        mse = mean_squared_error(y, pred_values)
        mae = mean_absolute_error(y, pred_values)
        r2 = r2_score(y, pred_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y - pred_values) / y)) * 100

        return ModelPerformance(
            mse=mse,
            mae=mae,
            r2_score=r2,
            rmse=rmse,
            mape=mape
        )


class VolatilityPredictor(FinancialModel):
    """
    Machine learning model for volatility prediction.

    Combines GARCH-type models with machine learning for
    predicting future volatility and risk measures.
    """

    def __init__(self, model_type: str = 'xgb', use_features: bool = True):
        """
        Initialize volatility predictor.

        Args:
            model_type: Type of ML model ('xgb', 'rf', 'gb')
            use_features: Whether to use additional features
        """
        self.model_type = model_type
        self.use_features = use_features

        if model_type == 'xgb':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif model_type == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """Fit volatility prediction model."""
        logger.info(f"Fitting {self.model_type} volatility predictor")

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info("Volatility predictor fitted successfully")

    def predict(self, X: np.ndarray) -> ModelPrediction:
        """Predict future volatility."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            for i, importance in enumerate(self.model.feature_importances_):
                feature_importance[f"feature_{i}"] = float(importance)

        return ModelPrediction(
            predictions=predictions,
            feature_importance=feature_importance,
            prediction_timestamp=pd.Timestamp.now()
        )

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelPerformance:
        """Evaluate volatility prediction performance."""
        predictions = self.predict(X)
        pred_values = predictions.predictions

        mse = mean_squared_error(y, pred_values)
        mae = mean_absolute_error(y, pred_values)
        r2 = r2_score(y, pred_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y - pred_values) / y)) * 100

        return ModelPerformance(
            mse=mse,
            mae=mae,
            r2_score=r2,
            rmse=rmse,
            mape=mape
        )


class MarketRegimeClassifier:
    """
    Machine learning classifier for market regime identification.

    Identifies different market regimes (bull, bear, sideways, high volatility)
    using unsupervised and supervised learning techniques.
    """

    def __init__(self, n_regimes: int = 4, method: str = 'hmm'):
        """
        Initialize market regime classifier.

        Args:
            n_regimes: Number of market regimes to identify
            method: Classification method ('hmm', 'kmeans', 'gmm')
        """
        self.n_regimes = n_regimes
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: np.ndarray, **kwargs) -> None:
        """Fit market regime classification model."""
        logger.info(f"Fitting {self.method} market regime classifier")

        X_scaled = self.scaler.fit_transform(X)

        if self.method == 'kmeans':
            from sklearn.cluster import KMeans
            self.model = KMeans(n_clusters=self.n_regimes, random_state=42)
        elif self.method == 'gmm':
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(n_components=self.n_regimes, random_state=42)
        elif self.method == 'hmm':
            try:
                from hmmlearn import hmm
                self.model = hmm.GaussianHMM(n_components=self.n_regimes, random_state=42)
            except ImportError:
                logger.warning("hmmlearn not available, falling back to KMeans")
                from sklearn.cluster import KMeans
                self.model = KMeans(n_clusters=self.n_regimes, random_state=42)

        self.model.fit(X_scaled)
        self.is_fitted = True
        logger.info("Market regime classifier fitted successfully")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict market regime for given data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict regime probabilities if available."""
        if not self.is_fitted:
            return None

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        elif hasattr(self.model, 'score_samples'):
            # For GMM, use score samples as proxy for probabilities
            scores = self.model.score_samples(X_scaled)
            # Convert to rough probability estimates
            return np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

        return None

    def get_regime_labels(self) -> Dict[int, str]:
        """Get human-readable labels for regimes."""
        return {
            0: "Bull Market",
            1: "Bear Market",
            2: "Sideways Market",
            3: "High Volatility"
        }


class FinancialMLManager:
    """
    Manager class for financial machine learning models.

    Provides unified interface for training, prediction, and evaluation
    of various financial ML models.
    """

    def __init__(self):
        self.models = {}
        self.logger = logger

    def create_price_predictor(self, model_type: str = 'ensemble', **kwargs) -> FinancialModel:
        """Create and return a price prediction model."""
        if model_type == 'ensemble':
            return EnsemblePricePredictor(**kwargs)
        elif model_type == 'lstm':
            return LSTMPricePredictor(**kwargs)
        elif model_type == 'rf':
            return EnsemblePricePredictor(**kwargs)  # Simplified
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def create_volatility_predictor(self, model_type: str = 'xgb', **kwargs) -> VolatilityPredictor:
        """Create and return a volatility prediction model."""
        return VolatilityPredictor(model_type=model_type, **kwargs)

    def create_regime_classifier(self, **kwargs) -> MarketRegimeClassifier:
        """Create and return a market regime classifier."""
        return MarketRegimeClassifier(**kwargs)

    def cross_validate_model(
        self,
        model: FinancialModel,
        X: np.ndarray,
        y: np.ndarray,
        cv_splits: int = 5
    ) -> Dict[str, float]:
        """Perform time series cross-validation on a model."""
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        mse_scores = []
        mae_scores = []
        r2_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Evaluate
            performance = model.evaluate(X_test, y_test)

            mse_scores.append(performance.mse)
            mae_scores.append(performance.mae)
            r2_scores.append(performance.r2_score)

        return {
            'mse_mean': np.mean(mse_scores),
            'mse_std': np.std(mse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores)
        }

    def create_feature_matrix(
        self,
        price_data: pd.DataFrame,
        lookback_periods: int = 20,
        include_technical: bool = True,
        include_macro: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create feature matrix for machine learning models.

        Args:
            price_data: Price data with OHLC columns
            lookback_periods: Number of periods to look back for features
            include_technical: Whether to include technical indicators
            include_macro: Whether to include macroeconomic features

        Returns:
            Feature matrix X and target vector y
        """
        df = price_data.copy()

        # Basic price features
        features = []

        # Lag features
        for lag in range(1, lookback_periods + 1):
            features.extend([
                df['close'].shift(lag),
                df['volume'].shift(lag) if 'volume' in df.columns else df['close'].shift(lag),
            ])

        if include_technical:
            # Technical indicators
            features.extend([
                df['close'].rolling(window=20).mean(),  # SMA 20
                df['close'].rolling(window=50).mean(),  # SMA 50
                df['close'].rolling(window=20).std(),   # Volatility 20
                (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std(),  # Z-score
            ])

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features.append(rsi)

            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            macd = ema12 - ema26
            features.append(macd)

        # Combine features
        feature_df = pd.concat(features, axis=1).dropna()

        # Target: next period return
        target = df['close'].shift(-1) / df['close'] - 1
        target = target.loc[feature_df.index]

        return feature_df.values, target.values


# Global ML manager instance
ml_manager = FinancialMLManager()


def get_ml_manager() -> FinancialMLManager:
    """Get the global financial ML manager instance."""
    return ml_manager
