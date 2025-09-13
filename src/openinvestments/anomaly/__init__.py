"""
Anomaly detection module for financial time series and portfolio monitoring.
Implements multiple detection algorithms for market anomalies and fraud detection.
"""

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
from typing import Dict, Any, Optional
from ..core.logging import get_logger

logger = get_logger(__name__)

class FinancialAnomalyDetector:
    """Anomaly detection for financial data using multiple algorithms."""
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination)
        self.lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination)

    def detect_anomalies_isolation_forest(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using Isolation Forest."""
        predictions = self.isolation_forest.fit_predict(data)
        anomaly_scores = self.isolation_forest.decision_function(data)
        
        return {
            "anomaly_labels": predictions.tolist(),
            "anomaly_scores": anomaly_scores.tolist(),
            "method": "isolation_forest",
            "contamination": self.contamination,
            "num_anomalies": np.sum(predictions == -1)
        }

    def detect_anomalies_lof(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using Local Outlier Factor."""
        anomaly_scores = self.lof.fit_predict(data)
        anomaly_distances = self.lof.negative_outlier_factor_
        
        return {
            "anomaly_labels": anomaly_scores.tolist(),
            "anomaly_distances": anomaly_distances.tolist(),
            "method": "local_outlier_factor",
            "contamination": self.contamination,
            "num_anomalies": np.sum(anomaly_scores == -1)
        }

    def detect_market_anomalies(self, returns: np.ndarray, volumes: np.ndarray = None) -> Dict[str, Any]:
        """Detect market anomalies in returns and volume data."""
        # Combine returns and volumes if available
        if volumes is not None:
            data = np.column_stack((returns, volumes))
        else:
            data = returns.reshape(-1, 1)
        
        # Use ensemble approach
        iso_result = self.detect_anomalies_isolation_forest(data)
        lof_result = self.detect_anomalies_lof(data)
        
        # Combine results (simple majority vote)
        combined_anomalies = np.where(
            (iso_result["anomaly_labels"] == -1) | (lof_result["anomaly_labels"] == -1),
            -1, 1
        )
        
        return {
            "combined_anomaly_labels": combined_anomalies.tolist(),
            "isolation_forest_results": iso_result,
            "local_outlier_factor_results": lof_result,
            "ensemble_method": "majority_vote",
            "total_anomalies": np.sum(combined_anomalies == -1)
        }

# OpenAI integration for anomaly explanation
def explain_anomaly(openai_client, anomaly_data: Dict[str, Any], context: str):
    """Use OpenAI to explain detected anomalies in financial context."""
    prompt = f"""
    Explain this financial anomaly detection result in a professional manner:
    
    Context: {context}
    
    Detection Results: {anomaly_data}
    
    Provide insight on what this means for risk management and potential actions.
    """
    
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    # Example usage
    detector = FinancialAnomalyDetector(contamination=0.1)
    returns = np.random.normal(0.001, 0.02, 1000)
    volumes = np.random.normal(1000, 200, 1000)
    
    result = detector.detect_market_anomalies(returns, volumes)
    print(result)