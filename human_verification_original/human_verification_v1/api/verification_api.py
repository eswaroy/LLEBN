from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import torch
from typing import List, Dict
import sys
import os
sys.path.append('..')

from inference.verify_user import HumanVerificationSystem

app = FastAPI(title="Human Verification API")

# Load system at startup
verifier = None

@app.get("/")
async def root():
    """Root endpoint - returns API information."""
    return {
        "name": "Human Verification API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "extract_embedding": "POST /extract_embedding",
            "verify_uniqueness": "POST /verify_uniqueness"
        }
    }

# Get the absolute path to the checkpoints directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'training', 'checkpoints')

@app.on_event("startup")
async def load_model():
    global verifier
    verifier = HumanVerificationSystem(
        model_path=os.path.join(CHECKPOINT_DIR, 'best_model.pth'),
        scaler_path=os.path.join(CHECKPOINT_DIR, 'scaler.pkl')
    )
    print("Human Verification System loaded!")

class InteractionData(BaseModel):
    user_id: str
    feature_vector: List[float]  # Pre-extracted features

class VerificationRequest(BaseModel):
    new_user_features: List[float]
    existing_user_embeddings: List[List[float]]

@app.post("/extract_embedding")
async def extract_embedding(data: InteractionData):
    """Extract embedding from user interaction features."""
    try:
        feature_vector = np.array(data.feature_vector)
        embedding = verifier.extract_embedding(feature_vector)
        
        return {
            "user_id": data.user_id,
            "embedding": embedding.tolist(),
            "embedding_norm": float(np.linalg.norm(embedding))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify_uniqueness")
async def verify_uniqueness(request: VerificationRequest):
    """Check if new user is unique compared to existing users."""
    try:
        new_embedding = np.array(request.new_user_features)
        existing_embeddings = np.array(request.existing_user_embeddings)
        
        scores = verifier.compute_verification_scores(
            new_embedding, 
            existing_embeddings
        )
        
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": verifier is not None}
