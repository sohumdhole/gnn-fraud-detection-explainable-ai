# GuardianGraph: Advanced GNN Fraud Detection Platform

GuardianGraph is a state-of-the-art, production-grade fraud detection platform combining Graph Neural Networks (GNNs) implemented in native PyTorch, network structure analysis with NetworkX, unsupervised anomaly isolation with Isolation Forest, and model explainability with SHAP. It is equipped with a stunning, high-fidelity real-time Streamlit dashboard and a FastAPI microservice.

---

## 🔬 Scientific & Mathematical Overview

Traditional fraud models (like XGBoost or LightGBM) treat transactions as isolated points in space, failing to capture complex relational structures such as credit card velocity rings, laundering chains, or bipartite merchants spikes. GuardianGraph solves this by utilizing **Graph Deep Learning** to aggregate spatial-structural context directly from transaction graphs.

### 1. Graph Structure Representation
We model financial activities as a homogeneous graph $G = (V, E, X, E_{feat})$ where:
- $V$ represent accounts (cardholders, merchants, entities).
- $E$ represents transaction edges between source account $u$ and target account $v$.
- $X \in \mathbb{R}^{|V| \times d_N}$ represents account-specific node features (balance, age, location risk, behavioral multipliers).
- $E_{feat} \in \mathbb{R}^{|E| \times d_E}$ represents transactional edge features (amount, cyclical hour, day of week, distance velocity, channel flag).

---

### 2. Graph Attention Network (GAT) Formulation
To weigh node-neighbor relationships dynamically, GuardianGraph employs custom multi-head Graph Attention Network (GAT) layers. The attention coefficient $\alpha_{ij}^k$ for attention head $k$ between account $i$ and neighbor $j$ is calculated as:

$$\alpha_{ij}^k = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T [W_k h_i \parallel W_k h_j]\right)\right)}{\sum_{l \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}_k^T [W_k h_i \parallel W_k h_l]\right)\right)}$$

where:
- $W_k \in \mathbb{R}^{F' \times F}$ is a head-specific linear transformation weight.
- $\mathbf{a}_k \in \mathbb{R}^{2F'}$ is the attention vector.
- $\parallel$ denotes concatenation.
- $\mathcal{N}(i)$ is the direct 1-hop topological neighborhood of account $i$.

The features are then aggregated via multi-head concatenation:

$$h_i^{(l+1)} = \parallel_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^k W_k h_j^{(l)} \right)$$

---

### 3. Graph Convolutional Network (GCN) Symmetric Normalization
For downstream representation learning, we apply a Graph Convolutional Network (GCN) layer to map hidden representations to final latent embeddings $z_i$. The symmetric normalized propagation is calculated as:

$$z_i^{(l+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \frac{1}{\sqrt{\tilde{d}_i \tilde{d}_j}} h_j^{(l)} W^{(l)} \right)$$

where $\tilde{d}_i = d_i + 1$ represents the node degree including self-loops, ensuring structural scale invariance.

---

### 4. Relational Edge Classification (Transaction Scoring)
To score individual transactions $e_{uv}$ between source account $u$ and target account $v$, we construct a high-dimensional edge representation fusing structural GNN node embeddings $z_u, z_v$ and the transaction features $e_{uv}$:

$$\text{Representation}(e_{uv}) = [ z_u \parallel z_v \parallel e_{uv} ] \in \mathbb{R}^{2 \cdot \text{emb\_dim} + d_E}$$

This concatenated vector is fed into a Multi-Layer Perceptron (MLP) classification network:

$$\hat{y}_{uv} = \text{Sigmoid}\left( \text{MLP}\left( [ z_u \parallel z_v \parallel e_{uv} ] \right) \right)$$

This architectural choice allows the model to decide transaction risk by combining the global topological health of the accounts (GNN) with the immediate characteristics of the transaction.

---

### 5. Explainable AI with Grouped SHAP
To ensure regulatory compliance (e.g. Fair Lending, GDPR right to explanation), we compute Shapley values on the MLP classifier input features. The local model prediction is represented as a linear combination of feature attributions:

$$g(z') = \phi_0 + \sum_{i=1}^M \phi_i z'_i$$

Since the GNN node embeddings $z_u, z_v$ have 16 dimensions each, they are latent factors and lack direct semantic meaning to human risk analysts. GuardianGraph maps these 37 raw SHAP values into **7 highly interpretable feature categories** by grouping them:

1. **Transaction Amount** ($\phi_{32}$)
2. **Hour of Day** ($\phi_{33}$)
3. **Day of Week** ($\phi_{34}$)
4. **Geographic Velocity** ($\phi_{35}$)
5. **Online Channel** ($\phi_{36}$)
6. **Source Account Network Context** ($\sum_{i=0}^{15} \phi_i$) — representing the structural risk of the sender's GNN neighborhood.
7. **Target Account Network Context** ($\sum_{i=16}^{31} \phi_i$) — representing the structural risk of the recipient's GNN neighborhood.

---

### 6. Unsupervised Anomaly Isolation
To discover emerging, zero-day fraud networks (e.g., newly formed money-laundering mule rings that lack historical labels), GuardianGraph runs an unsupervised **Isolation Forest** directly on the learned structural GNN node embeddings $Z \in \mathbb{R}^{|V| \times d_{emb}}$:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Nodes that isolate quickly (low average path length $E(h(x))$ across isolation trees) represent structural topological anomalies and are flagged instantly on the dashboard.

---

## 🛠️ Architecture and Deployment Setup

### Directory Blueprint
```
gnn_fraud_detection/
├── requirements.txt           # Python dependencies
├── setup_env.ps1              # Environment configuration script
├── graph_data_generator.py    # Transaction graph simulator
├── gnn_model.py               # Custom GCN/GAT native PyTorch layer implementation
├── anomaly_detector.py        # Unsupervised Isolation Forest wrapper
├── explainability_engine.py   # SHAP translation mapping engine
├── train_pipeline.py          # Unified model training & serialization script
├── api_service.py             # FastAPI microservice for low-latency scoring
├── app.py                     # Sleek Streamlit interactive dashboard
├── deploy_architecture.json   # Enterprise AWS ECS & Redis specifications
└── README.md                  # Deep technical specification documentation
```

### Production Deployment Schema
The FastAPI endpoint (`api_service.py`) is designed for sub-10ms response times by caching GNN node embeddings in memory or a Redis lookup table, bypassing GNN re-propagation on every API query:

1. **Transaction Request** is posted to `/score` or `/explain`.
2. **Redis Embedding Check**: The service looks up pre-calculated structural embeddings $z_{source}$ and $z_{target}$ from the daily GNN batch training cache (latency < 2ms).
3. **Instant MLP Classifier Inference**: The edge MLP classifier scores the transaction using the cached embeddings + live request parameters (latency < 1ms).
4. **Asynchronous Explainability**: If `/explain` is called, the SHAP engine runs in a Celery background queue to keep the API responsive, pushing explanation charts to the dashboard.
