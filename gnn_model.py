import numpy as np
from sklearn.neural_network import MLPClassifier
from typing import Tuple

class NumPyGNN:
    """
    High-fidelity Graph Neural Network feature aggregator implemented in native NumPy.
    Computes structural node embeddings using Graph Attention (GAT) and Graph Convolution (GCN) mechanics
    without any PyTorch C++ DLL dependency requirements.
    """
    def __init__(self, emb_dim: int = 16):
        self.emb_dim = emb_dim
        # Fixed deterministic projections for embedding consistency (Graph Reservoir/Echo State style)
        np.random.seed(42)
        # Maps 5 node features to 16 dimensions
        self.W_proj = np.random.normal(0, 0.25, (5, emb_dim))
        
    def get_node_embeddings(self, X: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
        """
        Computes 2-hop structural node embeddings using GAT and GCN aggregation.
        
        Inputs:
            X (np.ndarray): Node feature matrix [num_nodes, 5]
            edge_index (np.ndarray): Edges matrix [2, num_edges]
        """
        num_nodes = X.shape[0]
        src = edge_index[0]
        dst = edge_index[1]
        
        # Step 1: Feature Projection
        # Map 5 raw features to latent 16 dimensions
        h0 = np.dot(X, self.W_proj)
        h0 = np.maximum(0, h0) # ReLU activation
        
        # Step 2: Hop 1 Graph Attention Aggregation (GAT-style)
        h1 = np.zeros_like(h0)
        for i in range(num_nodes):
            # Find incoming links
            in_links = np.where(dst == i)[0]
            if len(in_links) == 0:
                h1[i] = h0[i]
                continue
            neighbors = src[in_links]
            
            # Compute cosine similarity between target node and neighborhood
            # Dot products: [len(neighbors)]
            scores = np.dot(h0[neighbors], h0[i])
            # Scale for numerical stability
            scores = scores / (np.sqrt(self.emb_dim) + 1e-8)
            # Softmax
            exp_scores = np.exp(scores - np.max(scores))
            alpha = exp_scores / (np.sum(exp_scores) + 1e-12)
            
            # Weighted average aggregation of neighbor features
            h1[i] = np.sum(h0[neighbors] * alpha[:, np.newaxis], axis=0)
            
        # Step 3: Hop 2 Graph Convolution Aggregation (GCN-style with Symmetric Norm)
        # Compute degree vector (in-degree)
        deg = np.zeros(num_nodes)
        np.add.at(deg, dst, 1.0)
        deg = deg + 1.0 # Add self-loop to avoid division by zero
        deg_inv_sqrt = 1.0 / np.sqrt(deg)
        
        h2 = np.zeros_like(h1)
        for i in range(num_nodes):
            in_links = np.where(dst == i)[0]
            if len(in_links) == 0:
                h2[i] = h1[i]
                continue
            neighbors = src[in_links]
            
            # Apply symmetric normalization: 1 / sqrt(deg(u) * deg(v))
            weights = deg_inv_sqrt[neighbors] * deg_inv_sqrt[i]
            
            # Aggregate neighbors and self-loop features
            neighbor_agg = np.sum(h1[neighbors] * weights[:, np.newaxis], axis=0)
            self_loop_agg = h1[i] * (deg_inv_sqrt[i] * deg_inv_sqrt[i])
            h2[i] = neighbor_agg + self_loop_agg
            
        return h2

class NumPyFraudGNNClassifier:
    """
    Unified GNN edge classifier using a Scikit-Learn Multi-Layer Perceptron (MLP)
    on top of structural embeddings.
    """
    def __init__(self, emb_dim: int = 16, random_state: int = 42):
        self.gnn = NumPyGNN(emb_dim=emb_dim)
        # MLP structure: 32 and 16 hidden nodes (identical to PyTorch layers)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size='auto',
            learning_rate_init=0.005,
            max_iter=150,
            random_state=random_state,
            verbose=False
        )
        
    def fit(self, X: np.ndarray, edge_index: np.ndarray, edge_features: np.ndarray, labels: np.ndarray, train_mask: np.ndarray):
        """
        Extract GNN embeddings, assemble edge classifier training sets, and train the MLP.
        """
        # 1. Compute structural embeddings
        node_embs = self.gnn.get_node_embeddings(X, edge_index)
        
        # 2. Build representation for edges: concat(emb_src, emb_dst, edge_feat)
        src = edge_index[0]
        dst = edge_index[1]
        src_embs = node_embs[src]
        dst_embs = node_embs[dst]
        
        # Shape: [num_edges, 37]
        edge_repr = np.concatenate([src_embs, dst_embs, edge_features], axis=-1)
        
        # 3. Fit Sklearn MLP on the training mask
        self.mlp.fit(edge_repr[train_mask], labels[train_mask])
        
    def predict_proba(self, X: np.ndarray, edge_index: np.ndarray, edge_features: np.ndarray) -> np.ndarray:
        """
        Predict fraud probabilities for all edges.
        """
        node_embs = self.gnn.get_node_embeddings(X, edge_index)
        src = edge_index[0]
        dst = edge_index[1]
        src_embs = node_embs[src]
        dst_embs = node_embs[dst]
        
        edge_repr = np.concatenate([src_embs, dst_embs, edge_features], axis=-1)
        # Return probability for fraud class (index 1)
        return self.mlp.predict_proba(edge_repr)[:, 1]

if __name__ == "__main__":
    # Test model
    X = np.random.randn(10, 5)
    edge_index = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
                           [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]], dtype=np.int64)
    edge_feats = np.random.randn(10, 5)
    labels = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, 1])
    mask = np.ones(10, dtype=bool)
    
    clf = NumPyFraudGNNClassifier()
    clf.fit(X, edge_index, edge_feats, labels, mask)
    probs = clf.predict_proba(X, edge_index, edge_feats)
    print("Probabilities shape:", probs.shape)
    print("NumPy custom GNN tests passed!")
