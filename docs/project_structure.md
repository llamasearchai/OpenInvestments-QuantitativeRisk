# OpenInvestments Project Structure

This document provides a comprehensive overview of the OpenInvestments Quantitative Risk Platform project structure and organization.

## 📁 Root Directory Structure

```
openinvestments-quantitative-risk/
├── 📄 main.py                     # Main application entry point
├── 📄 setup.py                    # Python package configuration
├── 📄 requirements.txt            # Python dependencies
├── 📄 pyproject.toml              # Modern Python project configuration
├── 📄 Dockerfile                  # Docker container definition
├── 📄 docker-compose.yml          # Multi-service Docker orchestration
├── 📄 cleanup_report.json         # Repository cleanup report (generated)
├── 📄 config.env.example          # Environment configuration template
├── 📄 test_installation.py        # Installation verification script
├── 📄 LICENSE                     # Project license
├── 📄 .gitignore                  # Git ignore patterns
├── 📁 src/                        # Source code directory
├── 📁 scripts/                    # Utility scripts
├── 📁 docs/                       # Documentation
├── 📁 tests/                      # Test suite
├── 📁 data/                       # Data storage (generated)
├── 📁 models/                     # Trained models (generated)
├── 📁 logs/                       # Application logs (generated)
└── 📁 reports/                    # Generated reports (generated)
```

## 🔧 Core Source Code Structure (`src/openinvestments/`)

```
src/openinvestments/
├── 📄 __init__.py                 # Package initialization
├── 📁 core/                       # Core utilities and configuration
│   ├── 📄 __init__.py
│   ├── 📄 config.py              # Platform configuration management
│   ├── 📄 logging.py             # Structured logging system
│   └── 📄 market_data.py         # Market data integration
├── 📁 valuation/                 # Option pricing and derivatives
│   ├── 📄 __init__.py
│   ├── 📄 monte_carlo.py         # Monte Carlo pricing engine
│   ├── 📄 trees.py               # Binomial/trinomial trees
│   ├── 📄 greeks.py              # Greeks calculation with AAD
│   └── 📄 pde_solver.py          # PDE solvers for exotics
├── 📁 risk/                      # Risk analytics and measurement
│   ├── 📄 __init__.py
│   ├── 📄 var_es.py              # VaR and Expected Shortfall
│   ├── 📄 distributions.py       # Heavy-tail distributions
│   └── 📄 portfolio_risk.py      # Portfolio risk analysis
├── 📁 portfolio/                 # Portfolio management
│   ├── 📄 __init__.py
│   ├── 📄 analysis.py            # Portfolio analysis tools
│   ├── 📄 optimization.py        # Advanced optimization algorithms
│   └── 📄 backtesting.py         # Backtesting framework
├── 📁 ml/                        # Machine learning components
│   ├── 📄 __init__.py
│   ├── 📄 models.py              # ML models for finance
│   └── 📄 feature_engineering.py # Financial feature engineering
├── 📁 data_quality/              # Data quality and cleansing
│   ├── 📄 __init__.py
│   └── 📄 validator.py           # Data validation and cleansing
├── 📁 monitoring/                # Monitoring and alerting
│   ├── 📄 __init__.py
│   ├── 📄 alerts.py              # Automated alerting system
│   └── 📄 audit.py               # Audit trail and logging
├── 📁 cli/                       # Command-line interface
│   ├── 📄 __init__.py
│   ├── 📄 valuation_cli.py       # Valuation commands
│   ├── 📄 risk_cli.py            # Risk analysis commands
│   ├── 📄 portfolio_cli.py       # Portfolio commands
│   ├── 📄 validation_cli.py      # Validation commands
│   └── 📄 reporting_cli.py       # Reporting commands
└── 📁 api/                       # REST API
    ├── 📄 __init__.py
    ├── 📁 routes/
    │   ├── 📄 __init__.py
    │   ├── 📄 valuation.py        # Valuation API endpoints
    │   ├── 📄 risk.py             # Risk API endpoints
    │   ├── 📄 portfolio.py        # Portfolio API endpoints
    │   ├── 📄 validation.py       # Validation API endpoints
    │   └── 📄 reports.py          # Reporting API endpoints
    └── 📁 middleware/             # API middleware
        ├── 📄 __init__.py
        └── 📄 observability.py     # Observability middleware
```

## 📊 Data and Configuration Files

### Environment Configuration (`config.env.example`)

```bash
# Application Settings
DEBUG=false
APP_NAME="OpenInvestments Quantitative Risk Platform"
VERSION="1.0.0"

# OpenAI Integration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-large

# Database
DATABASE_URL=sqlite:///./data/openinvestments.db

# MLflow
MLFLOW_TRACKING_URI=sqlite:///./data/mlflow.db
MLFLOW_ARTIFACT_STORE_URI=./data/mlruns

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_key_here

# GPU Acceleration
ENABLE_GPU=false

# Logging
LOG_LEVEL=INFO

# FastAPI Server
HOST=0.0.0.0
PORT=8000
```

### Dependencies (`requirements.txt`)

```txt
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Machine Learning & Statistics
scikit-learn>=1.0.0
statsmodels>=0.13.0
pymc>=4.0.0
numpyro>=0.8.0
jax>=0.3.0
jaxlib>=0.3.0

# Survival Analysis
scikit-survival>=0.17.0

# Anomaly Detection
pyod>=1.0.0
river>=0.14.0

# Optimization
cvxpy>=1.3.0

# Model Registry & MLOps
mlflow>=1.25.0
feast>=0.19.0

# CLI & UI
click>=8.0.0
rich>=12.0.0
tqdm>=4.62.0
questionary>=1.10.0

# Web Framework
fastapi>=0.85.0
uvicorn>=0.18.0
pydantic>=1.9.0

# OpenAI Integration
openai>=1.0.0
openai-agents>=0.1.0

# Observability
opentelemetry-distro>=0.34b0
opentelemetry-instrumentation-fastapi>=0.34b0
prometheus-client>=0.14.0
structlog>=22.0.0

# Security
cryptography>=38.0.0
python-jose[cryptography]>=3.3.0
bcrypt>=4.0.0
passlib>=1.7.0

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-xdist>=2.5.0
pytest-mock>=3.8.0
pytest-benchmark>=4.0.0
hypothesis>=6.50.0
mutmut>=2.4.0

# Code Quality
black>=22.0.0
ruff>=0.0.215
mypy>=0.981
pre-commit>=2.20.0
isort>=5.10.0

# Data Storage & Processing
duckdb>=0.6.0
pyarrow>=8.0.0
polars>=0.15.0
sqlalchemy>=1.4.0

# GPU Acceleration
numba>=0.56.0
cupy>=11.0.0

# Configuration & Secrets
python-dotenv>=0.19.0
pydantic-settings>=2.0.0

# Documentation
mkdocs>=1.4.0
mkdocs-material>=8.5.0
mkdocstrings>=0.19.0

# Development Tools
jupyter>=1.0.0
ipykernel>=6.0.0
notebook>=6.4.0
```

## 🧪 Test Structure (`tests/`)

```
tests/
├── 📄 __init__.py
├── 📁 unit/                     # Unit tests
│   ├── 📄 test_valuation.py     # Valuation engine tests
│   ├── 📄 test_risk.py          # Risk analytics tests
│   ├── 📄 test_portfolio.py     # Portfolio tests
│   └── 📄 test_ml.py            # ML model tests
├── 📁 integration/              # Integration tests
│   ├── 📄 test_api.py           # API integration tests
│   └── 📄 test_cli.py           # CLI integration tests
├── 📁 performance/              # Performance tests
│   ├── 📄 test_monte_carlo_perf.py
│   └── 📄 test_risk_calc_perf.py
├── 📁 fixtures/                 # Test data and fixtures
│   ├── 📄 sample_market_data.csv
│   └── 📄 test_portfolio.json
├── 📄 conftest.py               # Pytest configuration
└── 📄 test_installation.py      # Installation verification
```

## 📜 Scripts Directory (`scripts/`)

```
scripts/
├── 📄 cleanup_repo.py           # Repository cleanup and organization
├── 📄 setup_dev_env.py          # Development environment setup
├── 📄 generate_docs.py          # Documentation generation
├── 📄 run_performance_tests.py  # Performance testing
├── 📄 backup_data.py            # Data backup utilities
├── 📄 migrate_database.py       # Database migration scripts
└── 📄 health_check.py           # System health monitoring
```

## 📚 Documentation Structure (`docs/`)

```
docs/
├── 📄 index.md                  # Documentation home
├── 📄 installation.md           # Installation guide
├── 📄 quickstart.md             # Quick start guide
├── 📄 api_reference.md          # API reference
├── 📄 cli_reference.md          # CLI reference
├── 📄 configuration.md          # Configuration guide
├── 📄 project_structure.md      # This file
├── 📁 api/                      # API documentation
│   ├── 📄 valuation.md
│   ├── 📄 risk.md
│   ├── 📄 portfolio.md
│   ├── 📄 validation.md
│   └── 📄 reports.md
├── 📁 examples/                 # Usage examples
│   ├── 📄 option_pricing.py
│   ├── 📄 risk_analysis.py
│   ├── 📄 portfolio_optimization.py
│   └── 📄 ml_model_training.py
└── 📁 tutorials/                # Step-by-step tutorials
    ├── 📄 getting_started.md
    ├── 📄 advanced_features.md
    └── 📄 production_deployment.md
```

## 🗃️ Data and Model Storage

### Data Directory (`data/`)

```
data/
├── 📁 raw/                      # Raw input data
├── 📁 processed/                # Processed/cleaned data
├── 📁 market_data/              # Market data cache
├── 📁 audit.db                  # Audit trail database
├── 📁 mlflow.db                 # MLflow tracking database
├── 📁 mlruns/                   # MLflow artifacts
└── 📁 temp/                     # Temporary files
```

### Models Directory (`models/`)

```
models/
├── 📁 valuation/                # Valuation models
├── 📁 risk/                     # Risk models
├── 📁 portfolio/                # Portfolio models
├── 📁 ml/                       # Machine learning models
├── 📁 checkpoints/              # Model checkpoints
└── 📁 registry/                 # Model registry
```

### Logs Directory (`logs/`)

```
logs/
├── 📄 application.log           # Main application logs
├── 📄 audit.log                 # Audit trail logs
├── 📄 api.log                   # API access logs
├── 📄 error.log                 # Error logs
├── 📄 performance.log           # Performance metrics
└── 📁 archived/                 # Archived log files
```

### Reports Directory (`reports/`)

```
reports/
├── 📁 valuation/                # Valuation reports
├── 📁 risk/                     # Risk analysis reports
├── 📁 portfolio/                # Portfolio reports
├── 📁 validation/               # Validation reports
├── 📁 compliance/               # Compliance reports
├── 📁 performance/              # Performance reports
└── 📁 custom/                   # Custom reports
```

## 🔧 Development Workflow

### Code Organization Principles

1. **Modular Architecture**: Each component has clear responsibilities
2. **Separation of Concerns**: Business logic, data access, and presentation layers
3. **Dependency Injection**: Loose coupling between components
4. **Configuration Management**: Environment-based configuration
5. **Error Handling**: Comprehensive error handling and logging
6. **Testing**: Unit, integration, and performance tests
7. **Documentation**: Inline documentation and API docs

### File Naming Conventions

- **Python files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Test files**: `test_*.py`
- **Config files**: `*.env`, `*.json`, `*.yaml`

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
from fastapi import FastAPI

# Local imports
from ..core.config import config
from ..valuation.monte_carlo import MonteCarloPricer
```

## 🚀 Deployment Structure

### Docker Multi-stage Build

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /app/venv ./venv
COPY src/ ./src/
COPY main.py .
ENV PATH="/app/venv/bin:$PATH"
EXPOSE 8000
CMD ["python", "main.py", "api"]
```

### Kubernetes Deployment

```
k8s/
├── 📄 deployment.yaml           # Main deployment
├── 📄 service.yaml              # Service configuration
├── 📄 configmap.yaml            # Configuration
├── 📄 secret.yaml               # Secrets
├── 📄 ingress.yaml              # Ingress rules
└── 📄 hpa.yaml                  # Horizontal pod autoscaling
```

## 📊 Monitoring and Observability

### Metrics Collection

- **Application Metrics**: Response times, error rates, throughput
- **Business Metrics**: Valuation accuracy, risk calculations, portfolio performance
- **System Metrics**: CPU, memory, disk usage, network I/O
- **Custom Metrics**: Model performance, data quality scores

### Logging Structure

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "logger": "valuation.engine",
  "message": "Option priced successfully",
  "user_id": "user123",
  "session_id": "session456",
  "request_id": "req789",
  "operation": "price_option",
  "parameters": {
    "S0": 100,
    "K": 100,
    "T": 1.0
  },
  "duration_ms": 150,
  "result": {
    "price": 10.45,
    "confidence_interval": [10.2, 10.7]
  }
}
```

### Alert Configuration

- **Risk Thresholds**: VaR breaches, drawdown limits
- **Model Performance**: Accuracy degradation, drift detection
- **System Health**: Service availability, performance degradation
- **Security Events**: Failed authentication, suspicious activity
- **Data Quality**: Missing data, outliers, stale data

## 🔒 Security Considerations

### Data Protection

- **Encryption**: Data at rest and in transit
- **Access Control**: Role-based access control (RBAC)
- **Audit Trail**: Comprehensive logging of all operations
- **Data Sanitization**: Input validation and sanitization

### Secure Configuration

- **Secrets Management**: Environment variables for sensitive data
- **API Security**: JWT tokens, rate limiting, CORS
- **Database Security**: Connection pooling, query parameterization
- **File System Security**: Proper permissions, secure file handling

## 📈 Performance Optimization

### Code Optimization

- **Vectorization**: NumPy operations for numerical computations
- **Caching**: Redis/memcached for frequently accessed data
- **Async Processing**: Asynchronous operations for I/O bound tasks
- **Memory Management**: Efficient data structures and garbage collection

### Infrastructure Optimization

- **Load Balancing**: Distribute load across multiple instances
- **Database Optimization**: Indexing, query optimization, connection pooling
- **CDN**: Content delivery network for static assets
- **Caching Layers**: Multiple caching layers (browser, CDN, application, database)

This comprehensive project structure ensures maintainability, scalability, and professional development practices for the OpenInvestments Quantitative Risk Platform.
