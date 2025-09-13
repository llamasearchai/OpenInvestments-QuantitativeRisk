"""
Data quality validation and cleansing for financial time series.

Provides comprehensive data quality checks including:
- Missing data detection and imputation
- Outlier detection and treatment
- Data consistency validation
- Time series integrity checks
- Cross-sectional validation
- Statistical quality metrics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

from ..core.logging import get_logger
from ..core.config import config

logger = get_logger(__name__)


class DataQualityIssue(Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    INCONSISTENT_DATA = "inconsistent_data"
    TIME_SERIES_GAPS = "time_series_gaps"
    NEGATIVE_PRICES = "negative_prices"
    STALE_DATA = "stale_data"
    VOLUME_SPIKES = "volume_spikes"
    PRICE_JUMPS = "price_jumps"


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    overall_score: float  # 0-100 scale
    total_issues: int
    issues_by_type: Dict[DataQualityIssue, int]
    detailed_issues: List[Dict[str, Any]]
    recommendations: List[str]
    data_statistics: Dict[str, Any]
    validation_timestamp: datetime


@dataclass
class CleansingResult:
    """Result of data cleansing operations."""
    original_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    cleansing_operations: List[Dict[str, Any]]
    data_quality_improvement: Dict[str, float]
    cleansing_timestamp: datetime


class DataValidator(ABC):
    """Abstract base class for data validators."""

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate data and return list of issues."""
        pass


class MissingDataValidator(DataValidator):
    """Validator for missing data detection."""

    def validate(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect missing values in data."""
        issues = []

        # Check for NaN values
        missing_summary = data.isnull().sum()
        for column in data.columns:
            missing_count = missing_summary[column]
            if missing_count > 0:
                missing_percentage = (missing_count / len(data)) * 100

                severity = "low" if missing_percentage < 1 else "medium" if missing_percentage < 5 else "high"

                issues.append({
                    "type": DataQualityIssue.MISSING_VALUES,
                    "column": column,
                    "severity": severity,
                    "count": int(missing_count),
                    "percentage": missing_percentage,
                    "description": f"Missing {missing_percentage:.1f}% of values in column {column}",
                    "recommendation": self._get_missing_data_recommendation(missing_percentage)
                })

        return issues

    def _get_missing_data_recommendation(self, percentage: float) -> str:
        """Get recommendation for missing data handling."""
        if percentage < 1:
            return "Consider forward-fill or interpolation for small gaps"
        elif percentage < 5:
            return "Use interpolation or regression imputation"
        else:
            return "Consider excluding this column or using advanced imputation methods"


class OutlierValidator(DataValidator):
    """Validator for outlier detection."""

    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        contamination: float = 0.1
    ):
        """
        Initialize outlier validator.

        Args:
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for IQR/Z-score methods
            contamination: Expected proportion of outliers for isolation forest
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination

    def validate(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect outliers in data."""
        issues = []

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            series = data[column].dropna()

            if len(series) < 10:
                continue

            if self.method == "iqr":
                outliers = self._detect_iqr_outliers(series)
            elif self.method == "zscore":
                outliers = self._detect_zscore_outliers(series)
            elif self.method == "isolation_forest":
                outliers = self._detect_isolation_forest_outliers(series)
            else:
                continue

            if len(outliers) > 0:
                outlier_percentage = (len(outliers) / len(series)) * 100

                severity = "low" if outlier_percentage < 1 else "medium" if outlier_percentage < 5 else "high"

                issues.append({
                    "type": DataQualityIssue.OUTLIERS,
                    "column": column,
                    "severity": severity,
                    "count": len(outliers),
                    "percentage": outlier_percentage,
                    "description": f"Detected {len(outliers)} outliers ({outlier_percentage:.1f}%) in column {column}",
                    "method": self.method,
                    "outlier_indices": outliers.index.tolist()[:10],  # First 10 indices
                    "recommendation": self._get_outlier_recommendation(outlier_percentage)
                })

        return issues

    def _detect_iqr_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - self.threshold * IQR
        upper_bound = Q3 + self.threshold * IQR

        return series[(series < lower_bound) | (series > upper_bound)]

    def _detect_zscore_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        return series[z_scores > self.threshold]

    def _detect_isolation_forest_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Isolation Forest."""
        try:
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)

            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Fit isolation forest
            clf = IsolationForest(contamination=self.contamination, random_state=42)
            outliers = clf.fit_predict(X_scaled)

            # Return outlier values
            return series[outliers == -1]
        except Exception:
            return pd.Series(dtype=series.dtype)

    def _get_outlier_recommendation(self, percentage: float) -> str:
        """Get recommendation for outlier handling."""
        if percentage < 1:
            return "Consider winsorizing or robust statistical methods"
        elif percentage < 5:
            return "Review outliers manually and consider transformation"
        else:
            return "Significant outlier presence - consider data source validation"


class TimeSeriesValidator(DataValidator):
    """Validator for time series data integrity."""

    def validate(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate time series data integrity."""
        issues = []

        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append({
                "type": DataQualityIssue.INCONSISTENT_DATA,
                "column": "index",
                "severity": "high",
                "description": "DataFrame does not have DatetimeIndex",
                "recommendation": "Convert index to DatetimeIndex for time series analysis"
            })
            return issues

        # Check for time gaps
        if len(data.index) > 1:
            time_diffs = data.index.to_series().diff().dropna()
            expected_diff = time_diffs.mode().iloc[0] if len(time_diffs) > 0 else pd.Timedelta(days=1)

            gaps = time_diffs[time_diffs > expected_diff * 2]
            if len(gaps) > 0:
                gap_percentage = (len(gaps) / len(data)) * 100

                severity = "low" if gap_percentage < 1 else "medium" if gap_percentage < 5 else "high"

                issues.append({
                    "type": DataQualityIssue.TIME_SERIES_GAPS,
                    "column": "timestamp",
                    "severity": severity,
                    "count": len(gaps),
                    "percentage": gap_percentage,
                    "description": f"Detected {len(gaps)} time gaps in series",
                    "recommendation": "Consider interpolation or forward-fill for missing time periods"
                })

        # Check for duplicates
        duplicates = data.index.duplicated()
        if duplicates.any():
            dup_count = duplicates.sum()
            dup_percentage = (dup_count / len(data)) * 100

            severity = "medium" if dup_percentage < 1 else "high"

            issues.append({
                "type": DataQualityIssue.DUPLICATES,
                "column": "timestamp",
                "severity": severity,
                "count": int(dup_count),
                "percentage": dup_percentage,
                "description": f"Found {dup_count} duplicate timestamps",
                "recommendation": "Remove duplicates or aggregate conflicting values"
            })

        # Check for stale data (no updates for extended period)
        if len(data.index) > 1:
            latest_timestamp = data.index.max()
            days_since_update = (datetime.now() - latest_timestamp).days

            if days_since_update > 7:  # More than a week old
                severity = "medium" if days_since_update < 30 else "high"

                issues.append({
                    "type": DataQualityIssue.STALE_DATA,
                    "column": "timestamp",
                    "severity": severity,
                    "days_stale": days_since_update,
                    "description": f"Data is {days_since_update} days old",
                    "recommendation": "Update data source or verify data feed"
                })

        return issues


class FinancialDataValidator(DataValidator):
    """Validator specifically for financial data characteristics."""

    def validate(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate financial data characteristics."""
        issues = []

        # Check for negative prices
        price_columns = ['close', 'high', 'low', 'open', 'price']
        for col in price_columns:
            if col in data.columns:
                negative_prices = data[data[col] <= 0]
                if len(negative_prices) > 0:
                    neg_percentage = (len(negative_prices) / len(data)) * 100

                    issues.append({
                        "type": DataQualityIssue.NEGATIVE_PRICES,
                        "column": col,
                        "severity": "high",
                        "count": len(negative_prices),
                        "percentage": neg_percentage,
                        "description": f"Found {len(negative_prices)} negative or zero prices in {col}",
                        "recommendation": "Remove or correct invalid price observations"
                    })

        # Check for price jumps (sudden large movements)
        if 'close' in data.columns and len(data) > 1:
            returns = data['close'].pct_change().abs()
            price_jumps = returns[returns > 0.5]  # More than 50% daily move

            if len(price_jumps) > 0:
                jump_percentage = (len(price_jumps) / len(data)) * 100

                severity = "low" if jump_percentage < 0.1 else "medium" if jump_percentage < 1 else "high"

                issues.append({
                    "type": DataQualityIssue.PRICE_JUMPS,
                    "column": "close",
                    "severity": severity,
                    "count": len(price_jumps),
                    "percentage": jump_percentage,
                    "description": f"Detected {len(price_jumps)} large price jumps (>50%)",
                    "recommendation": "Investigate price jump causes and consider data filtering"
                })

        # Check for volume spikes
        if 'volume' in data.columns:
            volume_mean = data['volume'].mean()
            volume_std = data['volume'].std()
            volume_spikes = data[data['volume'] > volume_mean + 3 * volume_std]

            if len(volume_spikes) > 0:
                spike_percentage = (len(volume_spikes) / len(data)) * 100

                severity = "low" if spike_percentage < 0.1 else "medium" if spike_percentage < 1 else "high"

                issues.append({
                    "type": DataQualityIssue.VOLUME_SPIKES,
                    "column": "volume",
                    "severity": severity,
                    "count": len(volume_spikes),
                    "percentage": spike_percentage,
                    "description": f"Detected {len(volume_spikes)} volume spikes (3+ std dev)",
                    "recommendation": "Review volume spikes for data quality issues"
                })

        # Check OHLC consistency
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            inconsistent_rows = data[
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            ]

            if len(inconsistent_rows) > 0:
                inconsistent_percentage = (len(inconsistent_rows) / len(data)) * 100

                issues.append({
                    "type": DataQualityIssue.INCONSISTENT_DATA,
                    "column": "ohlc",
                    "severity": "high",
                    "count": len(inconsistent_rows),
                    "percentage": inconsistent_percentage,
                    "description": f"Found {len(inconsistent_rows)} inconsistent OHLC observations",
                    "recommendation": "Correct OHLC relationships (high >= max(open,close) >= min(open,close) >= low)"
                })

        return issues


class DataCleanser:
    """Data cleansing and imputation class."""

    def __init__(self):
        self.logger = logger

    def cleanse_missing_data(
        self,
        data: pd.DataFrame,
        method: str = "interpolate",
        max_missing_pct: float = 0.1
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Cleanse missing data from DataFrame.

        Args:
            data: Input DataFrame
            method: Imputation method ('interpolate', 'forward_fill', 'mean', 'median')
            max_missing_pct: Maximum allowed missing percentage per column

        Returns:
            Cleaned DataFrame and list of operations performed
        """
        cleaned_data = data.copy()
        operations = []

        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count == 0:
                continue

            missing_pct = missing_count / len(data)

            # Skip columns with too much missing data
            if missing_pct > max_missing_pct:
                operations.append({
                    "operation": "skipped",
                    "column": column,
                    "reason": f"Missing percentage ({missing_pct:.1%}) exceeds threshold ({max_missing_pct:.1%})",
                    "missing_count": int(missing_count)
                })
                continue

            if method == "interpolate":
                cleaned_data[column] = data[column].interpolate(method='linear')
            elif method == "forward_fill":
                cleaned_data[column] = data[column].fillna(method='ffill')
            elif method == "backward_fill":
                cleaned_data[column] = data[column].fillna(method='bfill')
            elif method == "mean":
                mean_value = data[column].mean()
                cleaned_data[column] = data[column].fillna(mean_value)
            elif method == "median":
                median_value = data[column].median()
                cleaned_data[column] = data[column].fillna(median_value)

            operations.append({
                "operation": method,
                "column": column,
                "missing_count": int(missing_count),
                "missing_percentage": missing_pct
            })

        return cleaned_data, operations

    def cleanse_outliers(
        self,
        data: pd.DataFrame,
        method: str = "winsorize",
        threshold: float = 0.05
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Cleanse outliers from DataFrame.

        Args:
            data: Input DataFrame
            method: Outlier treatment method ('winsorize', 'remove', 'cap')
            threshold: Quantile threshold for winsorizing

        Returns:
            Cleaned DataFrame and list of operations performed
        """
        cleaned_data = data.copy()
        operations = []

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            series = data[column].dropna()

            if len(series) < 10:
                continue

            # Calculate bounds
            lower_bound = series.quantile(threshold)
            upper_bound = series.quantile(1 - threshold)

            outliers_lower = data[column] < lower_bound
            outliers_upper = data[column] > upper_bound
            outliers_count = (outliers_lower | outliers_upper).sum()

            if outliers_count == 0:
                continue

            if method == "winsorize":
                cleaned_data[column] = np.where(
                    data[column] < lower_bound,
                    lower_bound,
                    np.where(data[column] > upper_bound, upper_bound, data[column])
                )
            elif method == "remove":
                cleaned_data = cleaned_data[
                    (data[column] >= lower_bound) & (data[column] <= upper_bound)
                ]
            elif method == "cap":
                cleaned_data[column] = np.clip(data[column], lower_bound, upper_bound)

            operations.append({
                "operation": method,
                "column": column,
                "outliers_count": int(outliers_count),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            })

        return cleaned_data, operations

    def cleanse_time_series(
        self,
        data: pd.DataFrame,
        freq: str = "D"
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Cleanse time series data by fixing gaps and duplicates.

        Args:
            data: Input DataFrame with DatetimeIndex
            freq: Expected frequency ('D' for daily, 'H' for hourly, etc.)

        Returns:
            Cleaned DataFrame and list of operations performed
        """
        operations = []

        # Remove duplicates
        duplicates_removed = len(data) - len(data[~data.index.duplicated()])
        if duplicates_removed > 0:
            data = data[~data.index.duplicated(keep='first')]
            operations.append({
                "operation": "remove_duplicates",
                "duplicates_removed": duplicates_removed
            })

        # Reindex to fill gaps
        if isinstance(data.index, pd.DatetimeIndex):
            full_index = pd.date_range(
                start=data.index.min(),
                end=data.index.max(),
                freq=freq
            )

            data_reindexed = data.reindex(full_index)

            # Interpolate missing values
            gaps_filled = data_reindexed.isnull().sum().sum()
            if gaps_filled > 0:
                data_reindexed = data_reindexed.interpolate(method='time')
                operations.append({
                    "operation": "fill_time_gaps",
                    "gaps_filled": int(gaps_filled),
                    "frequency": freq
                })

            data = data_reindexed

        return data, operations


class DataQualityManager:
    """
    Central manager for data quality validation and cleansing.
    """

    def __init__(self):
        self.validators = [
            MissingDataValidator(),
            OutlierValidator(),
            TimeSeriesValidator(),
            FinancialDataValidator()
        ]
        self.cleanser = DataCleanser()
        self.logger = logger

    def validate_data(
        self,
        data: pd.DataFrame,
        validator_types: Optional[List[str]] = None
    ) -> DataQualityReport:
        """
        Comprehensive data quality validation.

        Args:
            data: DataFrame to validate
            validator_types: Specific validator types to run (optional)

        Returns:
            Comprehensive data quality report
        """
        all_issues = []

        # Run all validators
        for validator in self.validators:
            validator_name = validator.__class__.__name__

            if validator_types is None or any(vt in validator_name.lower() for vt in validator_types):
                try:
                    issues = validator.validate(data)
                    all_issues.extend(issues)
                except Exception as e:
                    self.logger.error(f"Validator {validator_name} failed: {e}")

        # Calculate overall score
        total_issues = len(all_issues)
        severity_weights = {
            "low": 1,
            "medium": 3,
            "high": 5,
            "critical": 10
        }

        weighted_score = sum(severity_weights.get(issue["severity"], 1) for issue in all_issues)
        overall_score = max(0, 100 - weighted_score)

        # Group issues by type
        issues_by_type = {}
        for issue in all_issues:
            issue_type = issue["type"]
            issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

        # Generate recommendations
        recommendations = self._generate_recommendations(all_issues, data)

        # Calculate data statistics
        data_statistics = self._calculate_data_statistics(data)

        report = DataQualityReport(
            overall_score=overall_score,
            total_issues=total_issues,
            issues_by_type=issues_by_type,
            detailed_issues=all_issues,
            recommendations=recommendations,
            data_statistics=data_statistics,
            validation_timestamp=datetime.now()
        )

        self.logger.info(f"Data quality validation completed. Score: {overall_score:.1f}/100")
        return report

    def cleanse_data(
        self,
        data: pd.DataFrame,
        cleansing_config: Dict[str, Any] = None
    ) -> CleansingResult:
        """
        Comprehensive data cleansing.

        Args:
            data: DataFrame to cleanse
            cleansing_config: Configuration for cleansing operations

        Returns:
            Cleansing result with operations performed
        """
        if cleansing_config is None:
            cleansing_config = {
                "missing_data": {"method": "interpolate", "max_missing_pct": 0.1},
                "outliers": {"method": "winsorize", "threshold": 0.05},
                "time_series": {"freq": "D"}
            }

        original_data = data.copy()
        all_operations = []

        # Cleanse missing data
        if "missing_data" in cleansing_config:
            config = cleansing_config["missing_data"]
            data, operations = self.cleanser.cleanse_missing_data(
                data, **config
            )
            all_operations.extend(operations)

        # Cleanse outliers
        if "outliers" in cleansing_config:
            config = cleansing_config["outliers"]
            data, operations = self.cleanser.cleanse_outliers(
                data, **config
            )
            all_operations.extend(operations)

        # Cleanse time series
        if "time_series" in cleansing_config:
            config = cleansing_config["time_series"]
            data, operations = self.cleanser.cleanse_time_series(
                data, **config
            )
            all_operations.extend(operations)

        # Calculate improvement metrics
        original_quality = self.validate_data(original_data)
        cleaned_quality = self.validate_data(data)

        improvement = {
            "score_improvement": cleaned_quality.overall_score - original_quality.overall_score,
            "issues_reduced": original_quality.total_issues - cleaned_quality.total_issues,
            "issues_reduced_pct": (
                (original_quality.total_issues - cleaned_quality.total_issues) /
                max(1, original_quality.total_issues) * 100
            )
        }

        result = CleansingResult(
            original_data=original_data,
            cleaned_data=data,
            cleansing_operations=all_operations,
            data_quality_improvement=improvement,
            cleansing_timestamp=datetime.now()
        )

        self.logger.info(f"Data cleansing completed. Score improvement: {improvement['score_improvement']:.1f} points")
        return result

    def _generate_recommendations(
        self,
        issues: List[Dict[str, Any]],
        data: pd.DataFrame
    ) -> List[str]:
        """Generate recommendations based on data quality issues."""
        recommendations = []

        issue_counts = {}
        for issue in issues:
            issue_type = issue["type"]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

        # Generate recommendations based on issue types
        if DataQualityIssue.MISSING_VALUES in issue_counts:
            missing_pct = sum(issue["percentage"] for issue in issues
                            if issue["type"] == DataQualityIssue.MISSING_VALUES)
            if missing_pct > 10:
                recommendations.append("Consider data source validation - high missing data percentage")
            else:
                recommendations.append("Implement appropriate missing data imputation strategy")

        if DataQualityIssue.OUTLIERS in issue_counts:
            outlier_pct = sum(issue["percentage"] for issue in issues
                            if issue["type"] == DataQualityIssue.OUTLIERS)
            if outlier_pct > 5:
                recommendations.append("Review outlier treatment strategy - significant outlier presence")
            else:
                recommendations.append("Implement robust statistical methods for outlier handling")

        if DataQualityIssue.TIME_SERIES_GAPS in issue_counts:
            recommendations.append("Establish regular data update schedule to minimize time gaps")

        if DataQualityIssue.STALE_DATA in issue_counts:
            recommendations.append("Verify data feed connectivity and update frequency")

        if len(recommendations) == 0:
            recommendations.append("Data quality is good - continue monitoring regularly")

        return recommendations

    def _calculate_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data statistics."""
        stats = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(data.select_dtypes(include=['object']).columns),
            "datetime_columns": len(data.select_dtypes(include=['datetime']).columns),
            "memory_usage_mb": data.memory_usage(deep=True).sum() / 1024 / 1024
        }

        # Add numeric column statistics
        if stats["numeric_columns"] > 0:
            numeric_data = data.select_dtypes(include=[np.number])
            stats["numeric_stats"] = {
                "mean": numeric_data.mean().to_dict(),
                "std": numeric_data.std().to_dict(),
                "min": numeric_data.min().to_dict(),
                "max": numeric_data.max().to_dict(),
                "missing_pct": (numeric_data.isnull().sum() / len(numeric_data) * 100).to_dict()
            }

        return stats


# Global data quality manager instance
data_quality_manager = DataQualityManager()


def get_data_quality_manager() -> DataQualityManager:
    """Get the global data quality manager instance."""
    return data_quality_manager
