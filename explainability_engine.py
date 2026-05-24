import numpy as np
import pandas as pd
import shap
import pickle
from typing import List, Dict, Any, Tuple
from gnn_model import NumPyFraudGNNClassifier

class GNNTransactionExplainer:
    """
    Explainability engine that bridges structural GNN embeddings with SHAP values.
    It takes raw SHAP values computed over the GNN-MLP input space and maps
    them back to human-interpretable dimensions (Transaction features + Source/Target Network Context).
    """
    def __init__(self, model: NumPyFraudGNNClassifier, background_embeddings: np.ndarray):
        """
        model: Trained NumPyFraudGNNClassifier model.
        background_embeddings: A representation matrix of size [num_samples, 37]
                               where 37 = source_emb(16) + target_emb(16) + edge_features(5)
        """
        self.model = model
        
        # Define a wrapper prediction function that SHAP can call.
        # It takes numpy arrays of size [N, 37] and returns logits (log-odds).
        def predict_mlp_logits(x_np: np.ndarray) -> np.ndarray:
            probs = self.model.mlp.predict_proba(x_np)[:, 1]
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            logits = np.log(probs / (1.0 - probs))
            return logits
            
        self.predict_mlp = predict_mlp_logits
        
        # Select background samples to keep Kernel SHAP fast
        bg_samples = shap.sample(background_embeddings, 50) if len(background_embeddings) > 50 else background_embeddings
        self.explainer = shap.KernelExplainer(self.predict_mlp, bg_samples)
        
        # Names for the 37 input features to SHAP
        self.raw_feature_names = (
            [f"src_emb_{i}" for i in range(16)] +
            [f"dst_emb_{i}" for i in range(16)] +
            ["Amount (log)", "Hour", "Day of Week", "Distance", "Online Flag"]
        )
        
        # Interpretability group names
        self.grouped_feature_names = [
            "Transaction Amount",
            "Hour of Day",
            "Day of Week",
            "Geographic Velocity",
            "Online Channel",
            "Source Account Network Context (GNN)",
            "Target Account Network Context (GNN)"
        ]

    def explain_transaction(
        self,
        src_emb: np.ndarray,
        dst_emb: np.ndarray,
        edge_feat: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generates grouped, interpretable SHAP values for a single transaction.
        
        Inputs:
            - src_emb (np.ndarray): Embedding of the source node [16]
            - dst_emb (np.ndarray): Embedding of the target node [16]
            - edge_feat (np.ndarray): Transaction features [5]
            
        Returns:
            - base_value (float): Model's base logit prediction
            - prediction (float): Predicted logit for this transaction
            - probability (float): Fraud probability (sigmoid(prediction))
            - shap_values (Dict[str, float]): Interpretable SHAP contribution per group feature
            - raw_shap_values (np.ndarray): The 37 raw SHAP values
        """
        # Form single input row: shape [1, 37]
        x_input = np.concatenate([src_emb, dst_emb, edge_feat]).reshape(1, -1)
        
        # Compute SHAP values
        shap_res = self.explainer.shap_values(x_input)
        
        if isinstance(shap_res, list):
            shap_vals = shap_res[0]
        else:
            shap_vals = shap_res
            
        shap_vals = shap_vals.flatten() # [37]
        
        # Calculate raw logits and probability
        pred_logit = self.predict_mlp(x_input)[0]
        pred_prob = 1.0 / (1.0 + np.exp(-pred_logit))
        
        # Aggregate the 37 features into 7 interpretable groups:
        # Indices 0 to 15: Source account embedding (GNN)
        # Indices 16 to 31: Target account embedding (GNN)
        # Indices 32 to 36: Edge features
        src_gnn_val = float(np.sum(shap_vals[0:16]))
        dst_gnn_val = float(np.sum(shap_vals[16:32]))
        
        grouped_vals = {
            "Source Account Network Context (GNN)": src_gnn_val,
            "Target Account Network Context (GNN)": dst_gnn_val,
            "Transaction Amount": float(shap_vals[32]),
            "Hour of Day": float(shap_vals[33]),
            "Day of Week": float(shap_vals[34]),
            "Geographic Velocity": float(shap_vals[35]),
            "Online Channel": float(shap_vals[36])
        }
        
        base_val = float(self.explainer.expected_value)
        
        return {
            "base_value": base_val,
            "prediction_logit": float(pred_logit),
            "probability": float(pred_prob),
            "shap_values": grouped_vals,
            "raw_shap_values": shap_vals,
            "raw_features": x_input.flatten()
        }

if __name__ == "__main__":
    # Test explainer
    model = NumPyFraudGNNClassifier()
    # Mock data to fit internal MLP representation quickly
    X = np.random.randn(10, 5)
    edge_index = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]], dtype=np.int64)
    edge_feats = np.random.randn(10, 5)
    labels = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    mask = np.ones(10, dtype=bool)
    
    model.fit(X, edge_index, edge_feats, labels, mask)
    
    bg_data = np.random.randn(5, 37)
    explainer = GNNTransactionExplainer(model, bg_data)
    
    src = np.random.randn(16)
    dst = np.random.randn(16)
    edge = np.random.randn(5)
    
    res = explainer.explain_transaction(src, dst, edge)
    print("Probability of fraud:", res['probability'])
    print("Explainer tests passed successfully!")
