import os
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from gnn_model import NumPyFraudGNNClassifier
from explainability_engine import GNNTransactionExplainer

# Initialize FastAPI App
app = FastAPI(
    title="Advanced GNN Fraud Detection Service",
    description="Real-time transaction scoring and structural explainability API powered by Graph Neural Networks (GNN) and SHAP.",
    version="1.0.0"
)

# Global model and artifact placeholders
model = None
artifacts = None
explainer = None
node_embeddings = None

# Input schemas for API requests
class TransactionRequest(BaseModel):
    source_id: int = Field(..., description="Source Account Node ID (0-799 for Users)")
    target_id: int = Field(..., description="Target Account Node ID (800-999 for Merchants, or 0-799 for Users)")
    amount: float = Field(..., gt=0, description="Transaction Amount in USD")
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of the week (0-6, where 0=Monday)")
    distance_km: float = Field(..., ge=0, description="Geographic distance from last transaction in km")
    online_flag: int = Field(..., ge=0, le=1, description="Online channel indicator (1=Online, 0=Card Present)")

class ScoreResponse(BaseModel):
    source_id: int
    target_id: int
    fraud_probability: float = Field(..., description="Predicted GNN fraud probability (0.0 to 1.0)")
    is_flagged: bool = Field(..., description="Binary risk decision based on 0.5 threshold")
    source_structural_anomaly_score: float = Field(..., description="Unsupervised GNN outlier score for source account (0.0 to 1.0)")
    target_structural_anomaly_score: float = Field(..., description="Unsupervised GNN outlier score for target account (0.0 to 1.0)")

class ExplainResponse(BaseModel):
    score_details: ScoreResponse
    base_value: float = Field(..., description="Model baseline logit prediction")
    prediction_logit: float = Field(..., description="Calculated logit for this transaction")
    shap_contributions: dict = Field(..., description="Grouped SHAP contributions explaining the classification logit")

@app.on_event("startup")
def load_models_and_artifacts():
    """
    Load trained GNN weights, unsupervised anomaly metrics, and NetworkX topological features on startup.
    """
    global model, artifacts, explainer, node_embeddings
    
    model_path = "gnn_model_best.pth"
    artifacts_path = "pipeline_artifacts.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(artifacts_path):
        print("Warning: Model weights or pipeline artifacts not found! Run 'train_pipeline.py' first.")
        return
        
    print("Loading pipeline artifacts...")
    with open(artifacts_path, "rb") as f:
        artifacts = pickle.load(f)
        
    node_embeddings = artifacts['node_embeddings']
    
    print("Loading GNN model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print("Initializing SHAP explainer...")
    explainer = GNNTransactionExplainer(model, artifacts['shap_background'])
    print("API Service successfully loaded and ready for inference!")

@app.get("/health")
def health_check():
    if model is None or artifacts is None:
        return {"status": "degraded", "message": "Model artifacts are missing. Please execute the training pipeline."}
    return {"status": "healthy", "message": "GNN Fraud Service is active."}

@app.get("/metrics")
def get_performance_metrics():
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Service not initialized. Model artifacts are missing.")
    return {
        "model_architecture": "GAT-GCN Homogeneous Edge Classifier",
        "performance_audit": artifacts['classification_metrics']
    }

@app.post("/score", response_model=ScoreResponse)
def score_transaction(request: TransactionRequest):
    if model is None or artifacts is None:
        raise HTTPException(status_code=503, detail="Service not initialized. Model artifacts are missing.")
        
    num_nodes = node_embeddings.shape[0]
    if request.source_id >= num_nodes or request.target_id >= num_nodes:
        raise HTTPException(status_code=400, detail=f"Invalid source_id or target_id. Graph node boundary is {num_nodes}.")
        
    # Retrieve precomputed structural node embeddings
    src_emb = node_embeddings[request.source_id]
    dst_emb = node_embeddings[request.target_id]
    
    # Scale/Format transaction features to match GNN training
    scaled_edge_feat = np.array([
        np.log10(request.amount),
        request.hour / 23.0,
        request.day_of_week / 6.0,
        min(request.distance_km / 500.0, 1.0),
        float(request.online_flag)
    ], dtype=np.float32)
    
    # Form representation for edge MLP classifier
    edge_repr = np.concatenate([src_emb, dst_emb, scaled_edge_feat]).reshape(1, -1)
    
    # Run inference
    prob = model.mlp.predict_proba(edge_repr)[0, 1]
    
    # Retrieve unsupervised anomaly stats
    df_nodes = artifacts['df_nodes']
    src_anomaly_score = float(df_nodes.loc[df_nodes['node_id'] == request.source_id, 'anomaly_score'].values[0])
    dst_anomaly_score = float(df_nodes.loc[df_nodes['node_id'] == request.target_id, 'anomaly_score'].values[0])
    
    return ScoreResponse(
        source_id=request.source_id,
        target_id=request.target_id,
        fraud_probability=prob,
        is_flagged=bool(prob >= 0.5),
        source_structural_anomaly_score=src_anomaly_score,
        target_structural_anomaly_score=dst_anomaly_score
    )

@app.post("/explain", response_model=ExplainResponse)
def explain_transaction(request: TransactionRequest):
    if model is None or artifacts is None or explainer is None:
        raise HTTPException(status_code=503, detail="Service not initialized. Model artifacts are missing.")
        
    num_nodes = node_embeddings.shape[0]
    if request.source_id >= num_nodes or request.target_id >= num_nodes:
        raise HTTPException(status_code=400, detail=f"Invalid source_id or target_id. Graph node boundary is {num_nodes}.")
        
    # Get basic scores
    score_res = score_transaction(request)
    
    # Retrieve structural node embeddings and format transaction features
    src_emb = node_embeddings[request.source_id]
    dst_emb = node_embeddings[request.target_id]
    scaled_edge_feat = np.array([
        np.log10(request.amount),
        request.hour / 23.0,
        request.day_of_week / 6.0,
        min(request.distance_km / 500.0, 1.0),
        float(request.online_flag)
    ], dtype=np.float32)
    
    # Run SHAP attribution
    shap_res = explainer.explain_transaction(src_emb, dst_emb, scaled_edge_feat)
    
    return ExplainResponse(
        score_details=score_res,
        base_value=shap_res['base_value'],
        prediction_logit=shap_res['prediction_logit'],
        shap_contributions=shap_res['shap_values']
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
