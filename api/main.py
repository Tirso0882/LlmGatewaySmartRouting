"""
LLM Gateway API - Fine-tuned Routing
API for testing the fine-tuned DistilBERT routing model
"""

import os
import sys
import time
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import model classes with fallback
try:
    from distilbert_finetuner import DistilBERTFineTuner
    print("✅ Full DistilBERTFineTuner loaded (training environment)")
except ImportError:
    try:
        from distilbert_inference import DistilBERTFineTuner
        print("✅ Lightweight DistilBERTFineTuner loaded (deployment environment)")
    except ImportError:
        print("⚠️ No DistilBERTFineTuner available - falling back to rule-based routing")
        DistilBERTFineTuner = None
from mock_llm_responses import mock_llm
from real_llm_integration_example import RealLLMIntegration

# Initialize FastAPI app
app = FastAPI(
    title="LLM Gateway API",
    description="Smart routing API using fine-tuned DistilBERT model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the fine-tuned model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'distilbert_llm_router')
fine_tuner = None

# Pydantic models for API
class RoutingRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

class RoutingResponse(BaseModel):
    prompt: str
    recommended_model: str
    confidence: float
    inference_time_ms: float
    reasoning: str
    alternative_models: List[str]
    llm_response: str
    llm_response_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_models: List[str]

class BatchRoutingRequest(BaseModel):
    prompts: List[str]

class BatchRoutingResponse(BaseModel):
    results: List[RoutingResponse]
    total_time_ms: float

# Available models for routing
AVAILABLE_MODELS = {
    'o3': {
        'name': 'o3',
        'description': 'High-accuracy model for complex reasoning',
        'cost_per_1k_tokens': 0.060,
        'avg_response_time_ms': 5000,
        'best_for': ['complex_reasoning', 'advanced_analysis', 'stem_problems']
    },
    'gpt-4o-mini': {
        'name': 'gpt-4o-mini', 
        'description': 'Fast and cost-effective for simple tasks',
        'cost_per_1k_tokens': 0.0001,
        'avg_response_time_ms': 400,
        'best_for': ['simple_queries', 'quick_responses', 'cost_optimization']
    },
    'o4-mini': {
        'name': 'o4-mini',
        'description': 'Balanced model for moderate complexity',
        'cost_per_1k_tokens': 0.015,
        'avg_response_time_ms': 2000,
        'best_for': ['moderate_complexity', 'balanced_performance']
    }
}

@app.on_event("startup")
async def startup_event():
    """Load the fine-tuned model on startup with Azure Blob Storage fallback"""
    global fine_tuner
    
    # Try to load DistilBERT from local path first
    try:
        print("🤖 Loading DistilBERT model from local path...")
        if DistilBERTFineTuner:
            fine_tuner = DistilBERTFineTuner()
            fine_tuner.load_model(model_path)
        else:
            raise ImportError("No model loader available")
        print("✅ DistilBERT model loaded successfully from local path!")
        return
    except Exception as e:
        print(f"⚠️ Local DistilBERT model loading failed: {e}")
    
    # Try to download from Azure Blob Storage
    try:
        print("📥 Attempting to download DistilBERT model from Azure Blob Storage...")
        
        # Check if Azure Storage SDK is available
        try:
            from azure_model_storage import AzureModelStorage
        except ImportError:
            print("⚠️ Azure Storage SDK not available, skipping download")
            fine_tuner = None
            return
        
        # Check if connection string is available
        import os
        if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
            storage = AzureModelStorage()
            blob_name = 'distilbert_llm_router_v1.zip'
            
            if storage.model_exists(blob_name):
                print("📥 Downloading DistilBERT model from Azure Blob Storage...")
                storage.download_model(blob_name, model_path)
                
                # Try to load the downloaded model
                try:
                    if DistilBERTFineTuner:
                        fine_tuner = DistilBERTFineTuner()
                        fine_tuner.load_model(model_path)
                    else:
                        raise ImportError("No model loader available")
                    print("✅ DistilBERT model downloaded and loaded successfully!")
                    return
                except Exception as e:
                    print(f"❌ Downloaded model loading failed: {e}")
            else:
                print("⚠️ DistilBERT model not found in Azure Blob Storage")
        else:
            print("⚠️ AZURE_STORAGE_CONNECTION_STRING not found, skipping download")
            
    except Exception as e:
        print(f"❌ Azure Blob Storage download failed: {e}")
    
    print("📋 Falling back to rule-based routing")
    fine_tuner = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Gateway API is running!", 
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "route": "/route",
            "batch": "/route/batch",
            "models": "/models",
            "stats": "/stats",
            "dashboard": "/dashboard"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=fine_tuner is not None,
        available_models=list(AVAILABLE_MODELS.keys())
    )

@app.get("/dashboard")
async def dashboard():
    """Serve the interactive dashboard"""
    return FileResponse("static/index.html")

@app.get("/models")
async def get_models():
    """Get available models and their configurations"""
    return {
        "models": AVAILABLE_MODELS,
        "total_models": len(AVAILABLE_MODELS)
    }

@app.post("/route", response_model=RoutingResponse)
async def route_prompt(request: RoutingRequest):
    """Route a single prompt to the best LLM"""
    
    try:
        # Measure inference time
        start_time = time.time()
        
        if fine_tuner is not None:
            # Use DistilBERT model
            predictions, confidences = fine_tuner.predict([request.prompt])
            recommended_model = predictions[0]
            confidence = confidences[0]
            routing_method = "DistilBERT"
        else:
            # Fallback to rule-based routing
            recommended_model, confidence = rule_based_routing(request.prompt)
            routing_method = "Rule-based"
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to ms
        
        # Get alternative models (all except the recommended one)
        alternative_models = [model for model in AVAILABLE_MODELS.keys() 
                            if model != recommended_model]
        
        # Generate reasoning based on model characteristics
        model_info = AVAILABLE_MODELS[recommended_model]
        reasoning = f"[{routing_method}] Selected {recommended_model} ({model_info['description']}) with {confidence:.2%} confidence"
        
        # Generate real LLM response
        try:
            real_llm = RealLLMIntegration()
            llm_response, llm_response_time = real_llm.call_real_llm(recommended_model, request.prompt)
        except Exception as e:
            # Fallback to mock response if real LLM fails
            llm_response, llm_response_time = mock_llm.generate_response(recommended_model, request.prompt)
        
        return RoutingResponse(
            prompt=request.prompt,
            recommended_model=recommended_model,
            confidence=confidence,
            inference_time_ms=inference_time,
            reasoning=reasoning,
            alternative_models=alternative_models,
            llm_response=llm_response,
            llm_response_time_ms=llm_response_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")

@app.post("/route/batch", response_model=BatchRoutingResponse)
async def route_batch(request: BatchRoutingRequest):
    """Route multiple prompts in batch"""
    
    try:
        start_time = time.time()
        
        if fine_tuner is not None:
            # Use DistilBERT model for batch predictions
            predictions, confidences = fine_tuner.predict(request.prompts)
            routing_method = "DistilBERT"
        else:
            # Fallback to rule-based routing for each prompt
            predictions = []
            confidences = []
            for prompt in request.prompts:
                model, conf = rule_based_routing(prompt)
                predictions.append(model)
                confidences.append(conf)
            routing_method = "Rule-based"
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        # Create results
        results = []
        for i, prompt in enumerate(request.prompts):
            recommended_model = predictions[i]
            confidence = confidences[i]
            
            # Get alternative models
            alternative_models = [model for model in AVAILABLE_MODELS.keys() 
                                if model != recommended_model]
            
            # Generate reasoning
            model_info = AVAILABLE_MODELS[recommended_model]
            reasoning = f"[{routing_method}] Selected {recommended_model} ({model_info['description']}) with {confidence:.2%} confidence"
            
            # Generate LLM response for each prompt
            try:
                real_llm = RealLLMIntegration()
                llm_response, llm_response_time = real_llm.call_real_llm(recommended_model, prompt)
            except Exception as e:
                # Fallback to mock response if real LLM fails
                llm_response, llm_response_time = mock_llm.generate_response(recommended_model, prompt)
            
            results.append(RoutingResponse(
                prompt=prompt,
                recommended_model=recommended_model,
                confidence=confidence,
                inference_time_ms=total_time / len(request.prompts),  # Average time per prompt
                reasoning=reasoning,
                alternative_models=alternative_models,
                llm_response=llm_response,
                llm_response_time_ms=llm_response_time
            ))
        
        return BatchRoutingResponse(
            results=results,
            total_time_ms=total_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch routing error: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get model statistics and performance info"""
    return {
        "model_info": {
            "name": "DistilBERT Fine-tuned Router",
            "architecture": "DistilBERT + Classification Head",
            "training_accuracy": "100% (on test set)",
            "model_size_mb": 255,
            "inference_device": "CPU/GPU"
        },
        "available_models": len(AVAILABLE_MODELS),
        "routing_capabilities": [
            "Semantic understanding of prompts",
            "Confidence-based routing",
            "Batch processing support",
            "Real-time inference"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
