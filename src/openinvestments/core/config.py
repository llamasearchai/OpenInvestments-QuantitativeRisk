"""
Core configuration module for OpenInvestments platform.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field


class PlatformSettings(BaseSettings):
    """Platform-wide configuration settings."""

    # Application
    app_name: str = "OpenInvestments Quantitative Risk Platform"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "data")
    models_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "models")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "logs")
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent.parent / "reports")

    # Database
    database_url: str = Field(default="sqlite:///openinvestments.db", env="DATABASE_URL")

    # OpenAI
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    embedding_model: str = Field(default="text-embedding-3-large", env="EMBEDDING_MODEL")

    # MLflow
    mlflow_tracking_uri: str = Field(default="sqlite:///mlflow.db", env="MLFLOW_TRACKING_URI")
    mlflow_artifact_uri: Optional[str] = Field(default=None, env="MLFLOW_ARTIFACT_URI")

    # Security
    secret_key: str = Field(default_factory=lambda: os.urandom(32).hex(), env="SECRET_KEY")
    jwt_secret_key: str = Field(default_factory=lambda: os.urandom(32).hex(), env="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # Risk Engine
    default_confidence_level: float = 0.95
    default_var_horizon: int = 1  # days
    default_es_horizon: int = 1  # days
    max_portfolio_size: int = 10000
    max_simulation_paths: int = 100000

    # Monte Carlo
    default_mc_paths: int = 10000
    default_mc_steps: int = 252  # trading days
    default_mc_timestep: float = 1.0 / 252.0

    # PDE Solver
    default_pde_grid_points: int = 1000
    default_pde_time_steps: int = 100
    default_pde_relaxation: float = 1.0

    # Bayesian Modeling
    default_mcmc_samples: int = 1000
    default_mcmc_tune: int = 1000
    default_mcmc_chains: int = 4

    # Survival Analysis
    default_survival_horizon: int = 365  # days
    default_censoring_threshold: float = 0.95

    # Anomaly Detection
    default_contamination: float = 0.1
    default_anomaly_threshold: float = 0.95

    # Stress Testing
    default_stress_scenarios: int = 100
    default_macro_scenarios: int = 50

    # Validation
    min_test_coverage: float = 0.95
    max_validation_runtime: int = 3600  # seconds
    default_validation_confidence: float = 0.99

    # Performance
    enable_gpu_acceleration: bool = Field(default=False, env="ENABLE_GPU")
    max_workers: int = Field(default_factory=lambda: os.cpu_count() or 4)
    chunk_size: int = 1000

    # Observability
    enable_telemetry: bool = Field(default=True, env="ENABLE_TELEMETRY")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    metrics_port: int = 8001

    # CLI
    cli_theme: str = "monokai"
    cli_progress_style: str = "bar"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


class Config:
    """Main configuration class providing access to all settings."""

    _instance: Optional[PlatformSettings] = None

    @classmethod
    def get(cls) -> PlatformSettings:
        """Get the singleton configuration instance."""
        if cls._instance is None:
            cls._instance = PlatformSettings()
        return cls._instance

    @classmethod
    def reload(cls) -> PlatformSettings:
        """Reload configuration from environment."""
        cls._instance = PlatformSettings()
        return cls._instance

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist."""
        config = cls.get()
        for directory in [config.data_dir, config.models_dir, config.logs_dir, config.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)


# Initialize configuration
config = Config.get()
Config.ensure_directories()
