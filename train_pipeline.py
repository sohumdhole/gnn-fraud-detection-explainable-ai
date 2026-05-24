import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
from graph_data_generator import generate_transaction_graph
from gnn_model import NumPyFraudGNNClassifier
from anomaly_detector import StructuralAnomalyDetector
from explainability_engine import GNNTransactionExplainer

def train_pipeline():
    print("====================================================")
    print("      NUMPY GNN FRAUD DETECTION TRAINING PIPELINE   ")
    print("====================================================")
    
    # ----------------------------------------------------
    # 1. GENERATE FINANCIAL TRANSACTION GRAPH
    # ----------------------------------------------------
    print("\n[Step 1] Generating high-fidelity transaction graph...")
    data = generate_transaction_graph(
        num_users=800,
        num_merchants=200,
        num_normal_transactions=5000,
        seed=42
    )
    
    # Extract data as NumPy arrays
    node_features = data['node_features'].numpy()
    node_types = data['node_types'].numpy()
    edge_index = data['edge_index'].numpy()
    edge_features = data['edge_features'].numpy()
    edge_labels = data['edge_labels'].numpy()
    
    df_transactions = data['df_transactions']
    df_nodes = data['df_nodes']
    nx_graph = data['nx_graph']
    
    num_edges = edge_labels.shape[0]
    print(f"Generated Graph: {node_features.shape[0]} nodes, {num_edges} transaction edges.")
    print(f"Fraud transactions count: {int(edge_labels.sum())} ({edge_labels.mean() * 100:.2f}%)")
    
    # ----------------------------------------------------
    # 2. CHRONOLOGICAL TRAIN-TEST SPLIT
    # ----------------------------------------------------
    split_idx = int(num_edges * 0.8)
    
    train_mask = np.zeros(num_edges, dtype=bool)
    train_mask[:split_idx] = True
    
    test_mask = np.zeros(num_edges, dtype=bool)
    test_mask[split_idx:] = True
    
    print(f"Chronological Train Set: {train_mask.sum()} edges")
    print(f"Chronological Test Set:  {test_mask.sum()} edges")
    
    # ----------------------------------------------------
    # 3. INITIALIZE & TRAIN GRAPH NEURAL NETWORK
    # ----------------------------------------------------
    print("\n[Step 2] Initializing NumPy FraudGNN Classifier...")
    model = NumPyFraudGNNClassifier(emb_dim=16, random_state=42)
    
    print("\n[Step 3] Training Multi-Layer GNN Edge Classifier...")
    # Train the MLP on the GNN topological embeddings + transaction edge features
    model.fit(node_features, edge_index, edge_features, edge_labels, train_mask)
    
    # ----------------------------------------------------
    # 4. MODEL PERFORMANCE AUDIT
    # ----------------------------------------------------
    print("\n[Step 4] Performing final model performance audit...")
    all_probs = model.predict_proba(node_features, edge_index, edge_features)
    
    y_train = edge_labels[train_mask]
    train_probs = all_probs[train_mask]
    train_auc = roc_auc_score(y_train, train_probs)
    
    y_test = edge_labels[test_mask]
    test_probs = all_probs[test_mask]
    test_auc = roc_auc_score(y_test, test_probs)
    test_preds = (test_probs >= 0.5).astype(int)
    test_f1 = f1_score(y_test, test_preds)
    
    print(f"\nTraining Results | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Test F1-Score: {test_f1:.4f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, test_preds, digits=4))
    
    # ----------------------------------------------------
    # 5. GNN NODE EMBEDDINGS & UNSUPERVISED ANOMALY AUDIT
    # ----------------------------------------------------
    print("\n[Step 5] Extracting node structural embeddings...")
    node_embeddings_np = model.gnn.get_node_embeddings(node_features, edge_index)
    print(f"Node embeddings extracted of shape: {node_embeddings_np.shape}")
    
    print("\n[Step 6] Running Unsupervised Anomaly Detection (Isolation Forest)...")
    anomaly_detector = StructuralAnomalyDetector(contamination=0.04)
    anomaly_detector.fit(node_embeddings_np)
    
    anomaly_labels, anomaly_scores = anomaly_detector.predict(node_embeddings_np)
    df_nodes_analyzed = anomaly_detector.analyze_anomalies(df_nodes, anomaly_labels, anomaly_scores)
    
    print(f"Unsupervised outliers flagged: {anomaly_labels.sum()} out of {len(df_nodes)}")
    
    # ----------------------------------------------------
    # 6. CONSTRUCT BACKGROUND SAMPLES FOR SHAP
    # ----------------------------------------------------
    print("\n[Step 7] Setting up SHAP explainability engine background...")
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    
    src_embs = node_embeddings_np[src_nodes]
    dst_embs = node_embeddings_np[dst_nodes]
    
    transaction_representations = np.concatenate([src_embs, dst_embs, edge_features], axis=-1)
    
    train_indices = np.where(train_mask)[0]
    train_normal_indices = [idx for idx in train_indices if edge_labels[idx] == 0]
    bg_indices = np.random.choice(train_normal_indices, size=min(100, len(train_normal_indices)), replace=False)
    
    bg_representations = transaction_representations[bg_indices]
    print(f"Compiled SHAP explainer with background size: {bg_representations.shape}")
    
    # ----------------------------------------------------
    # 7. SAVE PIPELINE ARTIFACTS
    # ----------------------------------------------------
    print("\n[Step 8] Exporting model, explainer, and analytics artifacts...")
    
    df_transactions['fraud_score'] = all_probs
    df_transactions['is_flagged'] = (all_probs >= 0.5).astype(int)
    
    # Save best GNN-MLP model weights
    with open("gnn_model_best.pth", "wb") as f:
        pickle.dump(model, f)
        
    artifacts = {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'edge_labels': edge_labels,
        'node_embeddings': node_embeddings_np,
        'df_nodes': df_nodes_analyzed,
        'df_transactions': df_transactions,
        'nx_graph': nx_graph,
        'shap_background': bg_representations,
        'classification_metrics': {
            'auc': roc_auc_score(y_test, test_probs),
            'f1': f1_score(y_test, test_preds),
            'precision': precision_score(y_test, test_preds),
            'recall': recall_score(y_test, test_preds)
        }
    }
    
    with open("pipeline_artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Saved 'pipeline_artifacts.pkl' and 'gnn_model_best.pth' successfully!")
    print("Full NumPy GNN training pipeline run completed successfully.")
    print("====================================================")

if __name__ == "__main__":
    train_pipeline()
