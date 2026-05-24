import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Any

class StructuralAnomalyDetector:
    """
    Performs unsupervised anomaly detection on structural node embeddings
    learned by the Graph Neural Network to discover emerging zero-day fraud structures.
    """
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=150,
            n_jobs=-1
        )
        
    def fit(self, node_embeddings: np.ndarray) -> "StructuralAnomalyDetector":
        """
        Fits the Isolation Forest on structural node embeddings.
        """
        self.model.fit(node_embeddings)
        return self
        
    def predict(self, node_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomaly status and calculates anomaly scores.
        
        Returns:
            - anomaly_labels (np.ndarray): Binary tags (1 = Anomaly, 0 = Normal)
            - anomaly_scores (np.ndarray): Raw anomaly scores (higher means more anomalous)
        """
        # IsolationForest predict returns -1 for outliers and 1 for inliers.
        preds = self.model.predict(node_embeddings)
        anomaly_labels = np.where(preds == -1, 1, 0)
        
        # decision_function returns negative values for outliers, positive for inliers.
        # We invert it so that higher scores mean MORE anomalous.
        raw_scores = self.model.decision_function(node_embeddings)
        anomaly_scores = -raw_scores # shift so higher is more anomalous
        
        # Scale to [0, 1] range for intuitive dashboard visualization
        min_s = anomaly_scores.min()
        max_s = anomaly_scores.max()
        if max_s - min_s > 1e-8:
            anomaly_scores = (anomaly_scores - min_s) / (max_s - min_s)
        else:
            anomaly_scores = np.zeros_like(anomaly_scores)
            
        return anomaly_labels, anomaly_scores
        
    def analyze_anomalies(
        self,
        df_nodes: pd.DataFrame,
        anomaly_labels: np.ndarray,
        anomaly_scores: np.ndarray
    ) -> pd.DataFrame:
        """
        Merges unsupervised anomaly predictions back into the nodes DataFrame
        for deep analytics.
        """
        df_res = df_nodes.copy()
        df_res['is_anomaly'] = anomaly_labels
        df_res['anomaly_score'] = anomaly_scores
        return df_res

if __name__ == "__main__":
    # Quick test
    detector = StructuralAnomalyDetector()
    mock_embs = np.random.randn(100, 16)
    # inject an outlier
    mock_embs[0] = mock_embs[0] * 10.0
    
    detector.fit(mock_embs)
    labels, scores = detector.predict(mock_embs)
    
    print("Outliers found:", labels.sum())
    print("Outlier node 0 is anomaly:", labels[0] == 1)
    print("Detector tests passed!")
