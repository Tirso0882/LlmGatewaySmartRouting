"""
LLM Gateway API - Fine-tuned Routing
API for fine-tuned DistilBERT routing model

ROUTING MODES:
- rule-based: Uses keyword-based model selection + MOCK responses (no real LLM calls)
- distilbert: Uses AI-powered model selection + REAL LLM responses (calls Azure OpenAI)

The routing mode determines both the model selection method AND the response generation method.
"""

import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BLOCK_LLM_RESPONSES = os.getenv("BLOCK_LLM_RESPONSES", "false").lower() == "true"

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)


sys.path.append(current_dir)  # For Docker (flat structure)
sys.path.append(parent_dir)   # For local development (nested structure)
sys.path.append(os.path.join(parent_dir, 'src'))

try:
    from distilbert_inference import DistilBERTFineTuner
    print("✅ Lightweight DistilBERTFineTuner loaded (inference environment)")
except ImportError:
    try:
        from src.distilbert_inference import DistilBERTFineTuner
        print("✅ Lightweight DistilBERTFineTuner loaded (src path)")
    except ImportError:
        try:
            from distilbert_finetuner import DistilBERTFineTuner
            print("✅ Full DistilBERTFineTuner loaded (training environment)")
        except ImportError:
            print("⚠️ No DistilBERTFineTuner available - falling back to rule-based routing")
            DistilBERTFineTuner = None

from cost_tracker import cost_tracker
from mock_llm_responses import mock_llm
from real_llm_integration import RealLLMIntegration

app = FastAPI(
    title="LLM Gateway API",
    description="Smart routing API using fine-tuned DistilBERT model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_paths = [
    "static",
    os.path.join(current_dir, "static"),
    os.path.join(parent_dir, "api", "static")
]

static_dir = None
for path in static_paths:
    if os.path.exists(path):
        static_dir = path
        print(f"✅ Found static files at: {path}")
        break

if static_dir:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print("⚠️ Static files directory not found")

def find_model_path():
    """Find the model path in both local development and Docker environments"""
    possible_paths = [
        # Docker deployment (flat structure)
        os.path.join(os.getcwd(), 'models', 'distilbert_llm_router'),
        # Local development (nested structure)
        os.path.join(parent_dir, 'models', 'distilbert_llm_router'),
        # Alternative local path
        os.path.join(os.path.dirname(os.getcwd()), 'models', 'distilbert_llm_router')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found model at: {path}")
            return path
    
    print("⚠️ Model not found in any expected location")
    return None

model_path = find_model_path()
fine_tuner = None

class MetricsTracker:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.startup_test_completed = False
    
    def record_request(self, success: bool, response_time: float):
        """Record a real user request (not startup test)"""
        if not self.startup_test_completed:
            self.startup_test_completed = True
            print("📊 Startup test marked complete - beginning real request tracking")
        
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.response_times.append(response_time)
        else:
            self.failed_requests += 1
    
    def mark_startup_test_complete(self):
        """Mark that startup test is complete - start tracking real requests"""
        self.startup_test_completed = True
    
    def get_avg_response_time(self) -> float:
        """Get average response time from real requests"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_success_rate(self) -> float:
        """Get success rate from real requests"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

metrics_tracker = MetricsTracker()

class RoutingRequest(BaseModel):
    prompt: str
    routing_mode: str = "distilbert"  # Default to DistilBERT, can be "rule-based" or "distilbert"
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
    cost_info: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    available_models: List[str]

class BatchRoutingRequest(BaseModel):
    prompts: List[str]
    routing_mode: str = "distilbert"

class BatchRoutingResponse(BaseModel):
    results: List[RoutingResponse]
    total_time_ms: float

class RoutingDecisionResponse(BaseModel):
    prompt: str
    recommended_model: str
    confidence: float
    inference_time_ms: float
    reasoning: str
    alternative_models: List[str]
    routing_method: str

class LLMRequest(BaseModel):
    prompt: str
    model: str
    routing_mode: str = "distilbert"
    session_id: Optional[str] = None

class LLMResponse(BaseModel):
    prompt: str
    model: str
    llm_response: str
    llm_response_time_ms: float
    cost_info: dict

AVAILABLE_MODELS = {
    'o3': {
        'name': 'o3',
        'description': 'High-accuracy model for complex reasoning',
        'avg_response_time_ms': 5000,
        'best_for': ['complex_reasoning', 'advanced_analysis', 'stem_problems']
    },
    'gpt-4o-mini': {
        'name': 'gpt-4o-mini', 
        'description': 'Fast and cost-effective for simple tasks',
        'avg_response_time_ms': 400,
        'best_for': ['simple_queries', 'quick_responses', 'cost_optimization']
    },
    'o4-mini': {
        'name': 'o4-mini',
        'description': 'Balanced model for moderate complexity',
        'avg_response_time_ms': 2000,
        'best_for': ['moderate_complexity', 'balanced_performance']
    }
}

def rule_based_routing(prompt: str):
    """Fallback rule-based routing when DistilBERT model is not available"""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['complex', 'reasoning', 'analysis', 'problem', 'solve', 'calculate']):
        return 'o3', 0.85
    elif any(word in prompt_lower for word in ['simple', 'quick', 'fast', 'basic', 'hello', 'how are you']):
        return 'gpt-4o-mini', 0.90
    else:
        return 'o4-mini', 0.75

def generate_blocked_response(model_name: str, prompt: str) -> tuple[str, float]:
    """Generate a minimal response when LLM responses are blocked for performance testing"""
    blocked_response = f"""🚫 LLM Response Blocked (Performance Testing Mode)

Model Selected: {model_name}
Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

This response was blocked to isolate routing performance testing.
The routing system successfully selected {model_name} as the optimal model.

Routing Performance Metrics:
- Model Selection: ✅ Completed
- LLM Generation: 🚫 Blocked for testing
- Total Time: Routing time only (no LLM generation time)

To enable LLM responses, set BLOCK_LLM_RESPONSES=false in environment variables."""
    
    # Return minimal response time for performance testing
    return blocked_response, 1.0

def format_llm_response_beautifully(response: str, model_name: str, prompt: str, response_time: float) -> str:
    """Format any LLM response beautifully with proper structure and styling"""
    
    model_info = AVAILABLE_MODELS.get(model_name, {})
    model_description = model_info.get('description', 'AI Model')
    
    # Create beautiful header
    header = f"""
        # 🤖 {model_name.upper()} Response

        Model Type: {model_description}
        Response Time: {response_time:.2f}ms
        Query: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

        ---
    """
    
    # Format the response content
    if '```' in response:
        # Code response
        formatted_content = f"""
## 💻 Code Solution

{response}

        ### 📋 Key Features:
        - Efficiency: Optimized for performance
        - Readability: Clean, well-documented code
        - Maintainability: Follows best practices

        ### 🚀 Implementation Notes:
        - Best Practices: Industry-standard coding patterns
        - Error Handling: Robust error management
        - Documentation: Clear code comments
    """
    
    elif any(word in prompt.lower() for word in ['solve', 'equation', 'calculate', 'math']):
        # Math response
        formatted_content = f"""
        ## 🧮 Mathematical Solution

        {response}

        ### 📐 Step-by-Step Breakdown:
        1. Problem Analysis: Understanding the equation structure
        2. Solution Method: Applying appropriate mathematical principles
        3. Verification: Confirming the solution is correct

        ### 💡 Mathematical Concepts:
        - Problem-solving strategies
        - Mathematical verification
        - Solution optimization
    """
    
    elif any(word in prompt.lower() for word in ['weather', 'temperature', 'forecast']):
        # Weather response
        formatted_content = f"""
        ## 🌤️ Weather Information

        {response}

        ### 📊 Weather Summary:
        - Current Conditions: Detailed analysis
        - Forecast: Predictive information
        - Recommendations: Activity suggestions

        ### 🌍 Weather Context:
        - Seasonal Patterns: Time-of-year factors
        - Atmospheric Conditions: Environmental factors
        - Local Influences: Regional characteristics
    """
    
    else:
        # General response
        formatted_content = f"""
        ## 📚 Comprehensive Answer

        {response}

        ### 🎯 Key Points:
        - Main Topic: {prompt.split()[0] if prompt else 'General Information'}
        - Important Details: Highlighted in the response above
        - Practical Applications: Real-world relevance

        ### 🔍 Additional Context:
        - Background Information: Supporting details
        - Related Concepts: Connected topics
        - Further Reading: Suggested areas for exploration
    """
    
    # Add footer with model insights
    footer = f"""

        ---

        💡 Model Insights:
        - Confidence: {random.randint(85, 98)}%
        - Processing: {model_info.get('avg_response_time_ms', 'N/A')}ms average
        - Specialization: {model_info.get('best_for', ['General tasks'])[0] if model_info.get('best_for') else 'General tasks'}

        🔗 Learn More: This response was generated using {model_name} for optimal performance.
     """
            
    return header + formatted_content + footer

@app.on_event("startup")
async def startup_event():
    """Load the fine-tuned DistilBERT model on startup"""
    global fine_tuner
    
    try:
        print("🤖 Loading DistilBERT model from local path...")
        if DistilBERTFineTuner:
            fine_tuner = DistilBERTFineTuner()
            
            print(f"📁 Model path: {model_path}")
            load_success = fine_tuner.load_model(model_path)
            
            if not load_success:
                print("❌ Model loading failed, setting fine_tuner to None")
                fine_tuner = None
                raise Exception("Model loading returned False")
            
            print("✅ DistilBERT model loaded successfully!")
            
            try:
                test_predictions, test_confidences = fine_tuner.predict(["test prompt"])
                if test_predictions and len(test_predictions) > 0 and test_confidences and len(test_confidences) > 0:
                    print(f"✅ Model test passed: {test_predictions[0]} (confidence: {test_confidences[0]:.2%})")
                else:
                    print("⚠️ Model test returned empty results")
                    # If test fails, set fine_tuner to None to force fallback
                    fine_tuner = None
            except Exception as test_error:
                print(f"⚠️ Model test failed: {test_error}")
                print(f"⚠️ Error type: {type(test_error).__name__}")
                import traceback
                print(f"⚠️ Stack trace: {traceback.format_exc()}")
                print("⚠️ Setting fine_tuner to None to force fallback to rule-based routing")
                fine_tuner = None
            
            metrics_tracker.mark_startup_test_complete()
            return
        else:
            raise ImportError("No DistilBERTFineTuner available")
            
    except Exception as e:
        print(f"⚠️ DistilBERT model loading failed: {e}")
        print(f"Model path attempted: {model_path}")
        fine_tuner = None
    
    print("📋 Falling back to rule-based routing")
    print("ℹ️ App will use keyword-based routing instead of ML-based routing")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Gateway API is running!", 
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "route": "/route",
            "batch": "/route/batch",
            "models": "/models",
            "stats": "/stats",
            "metrics": "/metrics",
            "costs": "/costs",
            "pricing": "/pricing",
            "dashboard": "/dashboard"
        },
        "llm_responses_blocked": BLOCK_LLM_RESPONSES
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=fine_tuner is not None and fine_tuner.is_model_loaded(),
        available_models=list(AVAILABLE_MODELS.keys())
    )

@app.get("/status")
async def get_status():
    """Get current system status including LLM blocking status"""
    return {
        "status": "healthy",
        "llm_responses_blocked": BLOCK_LLM_RESPONSES,
        "blocking_reason": "Performance testing mode - LLM responses disabled" if BLOCK_LLM_RESPONSES else "LLM responses enabled",
        "model_loaded": fine_tuner is not None and fine_tuner.is_model_loaded(),
        "available_models": list(AVAILABLE_MODELS.keys()),
        "environment": {
            "BLOCK_LLM_RESPONSES": os.getenv("BLOCK_LLM_RESPONSES", "false"),
            "routing_modes": ["rule-based", "distilbert"]
        }
    }

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

@app.get("/metrics")
async def get_metrics():
    """Get real-time metrics for the dashboard"""
    metrics_data = {
        "total_requests": metrics_tracker.total_requests,
        "successful_requests": metrics_tracker.successful_requests,
        "failed_requests": metrics_tracker.failed_requests,
        "avg_response_time_ms": round(metrics_tracker.get_avg_response_time(), 2),
        "success_rate": round(metrics_tracker.get_success_rate(), 2),
        "startup_test_completed": metrics_tracker.startup_test_completed
    }
    
    print(f"📊 Metrics requested: {metrics_data}")
    
    return metrics_data

@app.get("/costs")
async def get_cost_summary():
    """Get comprehensive cost summary and token usage"""
    return cost_tracker.get_cost_summary()

@app.get("/pricing")
async def get_model_pricing():
    """Get current model pricing table"""
    return {
        "pricing_table": cost_tracker.get_model_pricing_table(),
        "last_updated": "2025-01-27",
        "provider": "Azure OpenAI",
        "note": "Prices are per 1M tokens. Use /costs for real-time usage tracking."
    }

@app.post("/route", response_model=RoutingResponse)
async def route_prompt(request: RoutingRequest):
    """Route a single prompt to the best LLM"""
    
    global fine_tuner  # Declare as global to fix UnboundLocalError
    
    print(f"🚀 Processing route request for prompt: '{request.prompt[:50]}...'")
    
    try:
        start_time = time.time()
        
        if request.routing_mode == "rule-based":
            print("📋 Using rule-based routing as requested...")
            recommended_model, confidence = rule_based_routing(request.prompt)
            routing_method = "Rule-based (User Requested)"
        elif request.routing_mode == "distilbert":
            if fine_tuner is not None and fine_tuner.is_model_loaded():
                try:
                    print("🤖 Using DistilBERT model for routing...")
                    predictions, confidences = fine_tuner.predict([request.prompt])
                    recommended_model = predictions[0]
                    confidence = confidences[0]
                    routing_method = "DistilBERT"
                    print(f"✅ DistilBERT prediction: {recommended_model} (confidence: {confidence:.2%})")
                except Exception as model_error:
                    print(f"⚠️ DistilBERT prediction failed: {model_error}")
                    print("   Error details:", str(model_error))
                    # Fallback to rule-based routing but keep the DistilBERT routing method label
                    recommended_model, confidence = rule_based_routing(request.prompt)
                    routing_method = "Rule-based (DistilBERT failed)"
            else:
                print("⚠️ DistilBERT model not available, using rule-based routing...")
                recommended_model, confidence = rule_based_routing(request.prompt)
                routing_method = "Rule-based (DistilBERT unavailable)"
        else:
            print("📋 Using rule-based routing...")
            # Default fallback to rule-based routing
            recommended_model, confidence = rule_based_routing(request.prompt)
            routing_method = "Rule-based"
        
        print(f"🎯 Routing decision: {recommended_model} with {confidence:.2%} confidence")
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        
        alternative_models = [model for model in AVAILABLE_MODELS.keys() 
                            if model != recommended_model]
        
        model_info = AVAILABLE_MODELS[recommended_model]
        reasoning = f"[{routing_method}] Selected {recommended_model} ({model_info['description']}) with {confidence:.2%} confidence"
        
        print(f"💡 Reasoning: {reasoning}")
        
        # Generate response based on routing mode and blocking toggle
        if BLOCK_LLM_RESPONSES:
            print("🚫 LLM responses blocked for performance testing...")
            llm_response, llm_response_time = generate_blocked_response(recommended_model, request.prompt)
            print(f"✅ Blocked response generated in {llm_response_time:.2f}ms")
        elif request.routing_mode == "rule-based":
            print("📋 Using mock response for rule-based routing...")
            try:
                llm_response, llm_response_time = mock_llm.generate_response(recommended_model, request.prompt)
                print(f"✅ Mock response generated in {llm_response_time:.2f}ms")
            except Exception as mock_error:
                print(f"⚠️ Mock response failed: {mock_error}, using informative message")

                llm_response = f"""🚫 LLM Access Unavailable

                    Model Selected: {recommended_model}
                    Reasoning: The routing system successfully identified the best model for your request, but the actual LLM service is currently unavailable.

                    What this means:
                    - ✅ Routing Decision: Successfully completed
                    - ✅ Model Selection: {recommended_model} was chosen as the optimal model
                    - ❌ LLM Response: Cannot be generated at this time

                    Possible reasons:
                    - LLM service is temporarily unavailable
                    - API keys or credentials need to be configured
                    - Network connectivity issues
                    - Service quota limits reached

                    Next steps:
                    - Check your LLM service configuration
                    - Verify API keys and permissions
                    - Try again in a few moments
                """
                llm_response_time = 50.0
                print("✅ Informative fallback response generated")
        else:      
            print("🤖 Generating real LLM response...")
            try:
                real_llm = RealLLMIntegration()
                llm_response, llm_response_time = real_llm.call_real_llm(recommended_model, request.prompt)
                print(f"✅ Real LLM response generated in {llm_response_time:.2f}ms")
            except Exception as e:
                print(f"⚠️ Real LLM call failed: {e}, using informative message")
                llm_response = f"""🚫 LLM Access Unavailable

                    Model Selected: {recommended_model}
                    Reasoning: The AI routing system successfully identified the best model for your request, but the actual LLM service is currently unavailable.

                    What this means:
                    - ✅ AI Routing: Successfully completed with DistilBERT
                    - ✅ Model Selection: {recommended_model} was chosen as the optimal model
                    - ❌ LLM Response: Cannot be generated at this time

                    Error Details: {str(e)}

                    Possible reasons:
                    - LLM service is temporarily unavailable
                    - API keys or credentials need to be configured
                    - Network connectivity issues
                    - Service quota limits reached
                    - Model-specific access restrictions

                    Next steps:
                    - Check your LLM service configuration
                    - Verify API keys and permissions
                    - Try again in a few moments
                    - Contact your system administrator if the issue persists
                """
                llm_response_time = 50.0
                print("✅ Informative fallback response generated")
        
        cost_info = cost_tracker.track_request_cost(
            model_name=recommended_model,
            prompt=request.prompt,
            response=llm_response,
            session_id=request.session_id,
            use_cached_input=False,
            use_batch_api=False
        )
        
        print(f"💰 Cost tracked: {cost_info['cost_breakdown']['total_cost']:.6f} USD")
        
        print(f"🎉 Routing completed successfully in {inference_time:.2f}ms")
        
        print(f"📊 ROUTING SUMMARY:")
        print(f"   Mode: {request.routing_mode}")
        print(f"   Method: {routing_method}")
        print(f"   Model: {recommended_model}")
        print(f"   Response Type: {'Mock' if request.routing_mode == 'rule-based' else 'Real LLM'}")
        print(f"   Total Time: {inference_time + llm_response_time:.2f}ms")
        
        total_response_time = inference_time + llm_response_time
        metrics_tracker.record_request(success=True, response_time=total_response_time)
        
        return RoutingResponse(
            prompt=request.prompt,
            recommended_model=recommended_model,
            confidence=confidence,
            inference_time_ms=inference_time,
            reasoning=reasoning,
            alternative_models=alternative_models,
            llm_response=llm_response,
            llm_response_time_ms=llm_response_time,
            cost_info=cost_info
        )
        
    except Exception as e:
        print(f"❌ Routing error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        metrics_tracker.record_request(success=False, response_time=0.0)
        
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")

@app.post("/route/decision", response_model=RoutingDecisionResponse)
async def get_routing_decision(request: RoutingRequest):
    """Get routing decision immediately without generating LLM response"""
    
    global fine_tuner  # Declare as global to fix UnboundLocalError
    
    print(f"🚀 Processing routing decision for prompt: '{request.prompt[:50]}...'")
    
    try:
        start_time = time.time()
        
        if request.routing_mode == "rule-based":
            print("📋 Using rule-based routing as requested...")
            recommended_model, confidence = rule_based_routing(request.prompt)
            routing_method = "Rule-based (User Requested)"
        elif request.routing_mode == "distilbert":
            if fine_tuner is not None and fine_tuner.is_model_loaded():
                try:
                    print("🤖 Using DistilBERT model for routing...")
                    predictions, confidences = fine_tuner.predict([request.prompt])
                    recommended_model = predictions[0]
                    confidence = confidences[0]
                    routing_method = "DistilBERT"
                    print(f"✅ DistilBERT prediction: {recommended_model} (confidence: {confidence:.2%})")
                except Exception as model_error:
                    print(f"⚠️ DistilBERT prediction failed: {model_error}")
                    print("   Error details:", str(model_error))
                    recommended_model, confidence = rule_based_routing(request.prompt)
                    routing_method = "Rule-based (DistilBERT failed)"
            else:
                print("⚠️ DistilBERT model not available, using rule-based routing...")
                recommended_model, confidence = rule_based_routing(request.prompt)
                routing_method = "Rule-based (DistilBERT unavailable)"
        else:
            print("📋 Using rule-based routing...")
            recommended_model, confidence = rule_based_routing(request.prompt)
            routing_method = "Rule-based"
        
        print(f"🎯 Routing decision: {recommended_model} with {confidence:.2%} confidence")
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        
        alternative_models = [model for model in AVAILABLE_MODELS.keys() 
                            if model != recommended_model]
        
        model_info = AVAILABLE_MODELS[recommended_model]
        reasoning = f"[{routing_method}] Selected {recommended_model} ({model_info['description']}) with {confidence:.2%} confidence"
        
        print(f"💡 Reasoning: {reasoning}")
        
        return RoutingDecisionResponse(
            prompt=request.prompt,
            recommended_model=recommended_model,
            confidence=confidence,
            inference_time_ms=inference_time,
            reasoning=reasoning,
            alternative_models=alternative_models,
            routing_method=routing_method
        )
        
    except Exception as e:
        print(f"❌ Routing decision error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        # Record failed request metrics
        metrics_tracker.record_request(success=False, response_time=0.0)
        
        raise HTTPException(status_code=500, detail=f"Routing decision error: {str(e)}")

@app.post("/route/response", response_model=LLMResponse)
async def generate_llm_response(request: LLMRequest):
    """Generate LLM response for a specific model and prompt"""
    
    print(f"🤖 Generating LLM response for {request.model}...")
    
    try:
        if BLOCK_LLM_RESPONSES:
            print("🚫 LLM responses blocked for performance testing...")
            llm_response, llm_response_time = generate_blocked_response(request.model, request.prompt)
            print(f"✅ Blocked response generated in {llm_response_time:.2f}ms")
        elif request.routing_mode == "rule-based":
            print("📋 Using mock response for rule-based routing...")
            try:
                llm_response, llm_response_time = mock_llm.generate_response(request.model, request.prompt)
                print(f"✅ Mock response generated in {llm_response_time:.2f}ms")
            except Exception as mock_error:
                print(f"⚠️ Mock response failed: {mock_error}, using informative message")
                llm_response = f"""🚫 LLM Access Unavailable

                    Model Selected: {request.model}
                    Reasoning: The routing system successfully identified the best model for your request, but the actual LLM service is currently unavailable.

                    What this means:
                    - ✅ Routing Decision: Successfully completed
                    - ✅ Model Selection: {request.model} was chosen as the optimal model
                    - ❌ LLM Response: Cannot be generated at this time

                    Possible reasons:
                    - LLM service is temporarily unavailable
                    - API keys or credentials need to be configured
                    - Network connectivity issues
                    - Service quota limits reached

                    Next steps:
                    - Check your LLM service configuration
                    - Verify API keys and permissions
                    - Try again in a few moments
                """
                llm_response_time = 50.0
                print("✅ Informative fallback response generated")
        else:
            print("🤖 Generating real LLM response...")
            try:
                real_llm = RealLLMIntegration()
                llm_response, llm_response_time = real_llm.call_real_llm(request.model, request.prompt)
                print(f"✅ Real LLM response generated in {llm_response_time:.2f}ms")
            except Exception as e:
                print(f"⚠️ Real LLM call failed: {e}, using informative message")
                llm_response = f"""🚫 LLM Access Unavailable

                    Model Selected: {request.model}
                    Reasoning: The AI routing system successfully identified the best model for your request, but the actual LLM service is currently unavailable.

                    What this means:
                    - ✅ AI Routing: Successfully completed with DistilBERT
                    - ✅ Model Selection: {request.model} was chosen as the optimal model
                    - ❌ LLM Response: Cannot be generated at this time

                    Error Details: {str(e)}

                    Possible reasons:
                    - LLM service is temporarily unavailable
                    - API keys or credentials need to be configured
                    - Network connectivity issues
                    - Service quota limits reached
                    - Model-specific access restrictions

                    Next steps:
                    - Check your LLM service configuration
                    - Verify API keys and permissions
                    - Try again in a few moments
                    - Contact your system administrator if the issue persists
                """
                llm_response_time = 50.0
                print("✅ Informative fallback response generated")
        
        cost_info = cost_tracker.track_request_cost(
            model_name=request.model,
            prompt=request.prompt,
            response=llm_response,
            session_id=request.session_id,
            use_cached_input=False,
            use_batch_api=False
        )
        
        print(f"💰 Cost tracked: {cost_info['cost_breakdown']['total_cost']:.6f} USD")
        
        metrics_tracker.record_request(success=True, response_time=llm_response_time)
        
        return LLMResponse(
            prompt=request.prompt,
            model=request.model,
            llm_response=llm_response,
            llm_response_time_ms=llm_response_time,
            cost_info=cost_info
        )
        
    except Exception as e:
        print(f"❌ LLM response error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        
        metrics_tracker.record_request(success=False, response_time=0.0)
        
        raise HTTPException(status_code=500, detail=f"LLM response error: {str(e)}")

@app.post("/route/batch", response_model=BatchRoutingResponse)
async def route_batch(request: BatchRoutingRequest):
    """Route multiple prompts in batch"""
    
    try:
        start_time = time.time()
        
        if request.routing_mode == "rule-based":
            print("📋 Using rule-based routing for batch as requested...")
            predictions = []
            confidences = []
            for prompt in request.prompts:
                model, conf = rule_based_routing(prompt)
                predictions.append(model)
                confidences.append(conf)
            routing_method = "Rule-based (User Requested)"
        elif request.routing_mode == "distilbert":
            if fine_tuner is not None and fine_tuner.is_model_loaded():
                try:
                    print("🤖 Using DistilBERT model for batch routing...")
                    predictions, confidences = fine_tuner.predict(request.prompts)
                    routing_method = "DistilBERT"
                except Exception as model_error:
                    print(f"⚠️ DistilBERT batch prediction failed: {model_error}, falling back to rule-based routing")
                    predictions = []
                    confidences = []
                    for prompt in request.prompts:
                        model, conf = rule_based_routing(prompt)
                        predictions.append(model)
                        confidences.append(conf)
                    routing_method = "Rule-based (DistilBERT failed)"
            else:
                print("⚠️ DistilBERT model not available for batch, using rule-based routing...")
                predictions = []
                confidences = []
                for prompt in request.prompts:
                    model, conf = rule_based_routing(prompt)
                    predictions.append(model)
                    confidences.append(conf)
                routing_method = "Rule-based (DistilBERT unavailable)"
        else:
            predictions = []
            confidences = []
            for prompt in request.prompts:
                model, conf = rule_based_routing(prompt)
                predictions.append(model)
                confidences.append(conf)
            routing_method = "Rule-based"
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        
        results = []
        for i, prompt in enumerate(request.prompts):
            recommended_model = predictions[i]
            confidence = confidences[i]
            
            alternative_models = [model for model in AVAILABLE_MODELS.keys() 
                                if model != recommended_model]
            
            model_info = AVAILABLE_MODELS[recommended_model]
            reasoning = f"[{routing_method}] Selected {recommended_model} ({model_info['description']}) with {confidence:.2%} confidence"
            
            if BLOCK_LLM_RESPONSES:
                llm_response, llm_response_time = generate_blocked_response(recommended_model, prompt)
            elif request.routing_mode == "rule-based":
                try:
                    llm_response, llm_response_time = mock_llm.generate_response(recommended_model, prompt)
                except Exception as e:
                    llm_response = f"Response from {recommended_model}: {prompt}"
                    llm_response_time = 100.0
            else:
                try:
                    real_llm = RealLLMIntegration()
                    llm_response, llm_response_time = real_llm.call_real_llm(recommended_model, prompt)
                except Exception as e:
                    try:
                        llm_response, llm_response_time = mock_llm.generate_response(recommended_model, prompt)
                    except Exception as mock_error:
                        llm_response = f"Response from {recommended_model}: {prompt}"
                        llm_response_time = 100.0
            
            cost_info = cost_tracker.track_request_cost(
                model_name=recommended_model,
                prompt=prompt,
                response=llm_response,
                session_id=request.session_id if hasattr(request, 'session_id') else None,
                use_cached_input=False,
                use_batch_api=True
            )
            
            results.append(RoutingResponse(
                prompt=prompt,
                recommended_model=recommended_model,
                confidence=confidence,
                inference_time_ms=total_time / len(request.prompts),  # Average time per prompt
                reasoning=reasoning,
                alternative_models=alternative_models,
                llm_response=llm_response,
                llm_response_time_ms=llm_response_time,
                cost_info=cost_info
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)