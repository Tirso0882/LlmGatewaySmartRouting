# LLM Gateway API - Production-Ready Smart Routing

A FastAPI application that provides intelligent LLM routing using a fine-tuned DistilBERT model with full production deployment capabilities.

## 🚀 Features

### Core Routing Capabilities
- **AI-Powered Routing**: Uses fine-tuned DistilBERT for intelligent model selection
- **Dual Routing Modes**: Choose between AI-powered (DistilBERT) or rule-based routing
- **High Accuracy**: 100% accuracy on test set with DistilBERT model
- **Real-time Inference**: Fast routing decisions (~15-25ms per prompt)
- **Batch Processing**: Efficient handling of multiple prompts simultaneously
- **Confidence Scoring**: Get confidence levels for all routing decisions
- **Alternative Models**: See other model options with reasoning

### Production Features
- **Interactive Dashboard**: Real-time web interface for testing and monitoring
- **Cost Tracking**: Cost monitoring and token usage analytics
- **Metrics & Monitoring**: Real-time performance metrics and health monitoring
- **Dual Response Modes**: Mock responses (rule-based) or real LLM calls (DistilBERT)
- **Fallback Mechanisms**: Automatic fallback to rule-based routing if AI model fails
- **Health Checks**: Built-in health monitoring and status reporting
- **CORS Support**: Cross-origin resource sharing for web applications

## 📋 Available Models

| Model | Description | Cost/1K tokens | Response Time | Best For |
|-------|-------------|----------------|---------------|----------|
| `o3` | High-accuracy for complex reasoning | $0.060 | 5000ms | Complex analysis, STEM problems |
| `gpt-4o-mini` | Fast and cost-effective | $0.0001 | 400ms | Simple queries, quick responses |
| `o4-mini` | Balanced performance | $0.015 | 2000ms | Moderate complexity tasks |

## 🏗️ Architecture

### Production & Development Architecture
The system uses a **containerized monolithic architecture** optimized for both production and development:

#### **Production (Azure App Service)**
```
┌─────────────────────────────────────────────────────────────┐
│                    Azure App Service                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              LLM Gateway Container                      │ │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │ │
│  │  │   FastAPI App   │  │    DistilBERT Model         │   │ │
│  │  │   (Port 8000)   │  │    (~250MB optimized)       │   │ │
│  │  └─────────────────┘  └─────────────────────────────┘   │ │
│  │           │                        │                    │ │
│  │           └────────┬─────────────────┘                    │ │
│  │                    │                                      │ │
│  │  ┌─────────────────▼─────────────────────────────────┐   │ │
│  │  │           Routing Engine                          │   │ │
│  │  │  • DistilBERT Inference                          │   │ │
│  │  │  • Rule-based Fallback                           │   │ │
│  │  │  • Cost Tracking                                 │   │ │
│  │  │  • Metrics & Monitoring                          │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

#### **Local Development (Docker)**
```
┌─────────────────────────────────────────────────────────────┐
│                    Local Docker Container                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              LLM Gateway Application                    │ │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐   │ │
│  │  │   FastAPI App   │  │    DistilBERT Model         │   │ │
│  │  │   (Port 8000)   │  │    (~250MB optimized)       │   │ │
│  │  └─────────────────┘  └─────────────────────────────┘   │ │
│  │           │                        │                    │ │
│  │           └────────┬─────────────────┘                    │ │
│  │                    │                                      │ │
│  │  ┌─────────────────▼─────────────────────────────────┐   │ │
│  │  │           Routing Engine                          │   │ │
│  │  │  • DistilBERT Inference                          │   │ │
│  │  │  • Rule-based Fallback                           │   │ │
│  │  │  • Cost Tracking                                 │   │ │
│  │  │  • Metrics & Monitoring                          │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ **Consistent Environment**: Same container runs locally and in production
- ✅ **Simple Deployment**: Azure App Service handles scaling and management
- ✅ **Fast Development**: Docker provides quick local setup
- ✅ **Production Features**: SSL, custom domains, auto-scaling
- ✅ **Cost-effective**: Pay only for what you use

## 🛠️ Setup

### Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_api.txt
   ```

2. **Start the API**:
   ```bash
   python main.py
   ```

3. **Access the API**:
   - **API**: http://localhost:8000
   - **Dashboard**: http://localhost:8000/dashboard
   - **Health Check**: http://localhost:8000/health
   - **API Documentation**: http://localhost:8000/docs
   - **Models**: http://localhost:8000/models
   - **Stats**: http://localhost:8000/stats
   - **Metrics**: http://localhost:8000/metrics
   - **Costs**: http://localhost:8000/costs
   - **Pricing**: http://localhost:8000/pricing

### Production Deployment

#### Azure App Service (Production)
```bash
# Deploy to Azure App Service for production
chmod +x deploy-to-azure-app-service.sh
./deploy-to-azure-app-service.sh
```

### Local Development

#### Docker (Local Development)
```bash
# Build and run locally with Docker
docker build -t llm-gateway .
docker run -p 8000:8000 llm-gateway
```

### Quick Start Workflow

#### **1. Local Development**
```bash
# Start local development environment
docker build -t llm-gateway .
docker run -p 8000:8000 llm-gateway

# Test your changes
curl http://localhost:8000/health
```

#### **2. Production Deployment**
```bash
# Deploy to Azure App Service
chmod +x deploy-to-azure-app-service.sh
./deploy-to-azure-app-service.sh

# Test production deployment
curl https://llm-gateway.azurewebsites.net/health
```

### Environment Configuration

For production deployment, configure these environment variables:

```bash
# Model Configuration
MODEL_PATH=/app/models/distilbert_llm_router
PYTHONPATH=/app

# Azure OpenAI API Keys (for real LLM responses)
O3_ENDPOINT=your-o3-endpoint
O3_API_KEY=your-o3-api-key
GPT4O_MINI_ENDPOINT=your-gpt4o-mini-endpoint
GPT4O_MINI_API_KEY=your-gpt4o-mini-api-key
O4_MINI_ENDPOINT=your-o4-mini-endpoint
O4_MINI_API_KEY=your-o4-mini-api-key

# Application Settings
DEBUG=false
WEBSITES_PORT=8000
PORT=8000
```

## 📡 API Endpoints

### Core Routing Endpoints

#### Health Check
```bash
GET /
GET /health
```
Returns API status, model availability, and system health.

#### Get Available Models
```bash
GET /models
```
Returns list of available models and their configurations.

#### Route Single Prompt (Full Response)
```bash
POST /route
```
```json
{
  "prompt": "What is the weather like today?",
  "routing_mode": "distilbert",  // or "rule-based"
  "user_id": "optional",
  "session_id": "optional"
}
```

#### Route Multiple Prompts (Batch)
```bash
POST /route/batch
```
```json
{
  "prompts": [
    "What is 2+2?",
    "Explain quantum computing",
    "Write a haiku"
  ],
  "routing_mode": "distilbert"
}
```

### Advanced Routing Endpoints

#### Get Routing Decision Only
```bash
POST /route/decision
```
Get routing decision without generating LLM response (faster for testing).

#### Generate LLM Response
```bash
POST /route/response
```
Generate LLM response for a specific model and prompt.

### Monitoring & Analytics

#### Interactive Dashboard
```bash
GET /dashboard
```
Real-time web interface for testing and monitoring.

#### Real-time Metrics
```bash
GET /metrics
```
Returns current performance metrics and statistics.

#### Cost Tracking
```bash
GET /costs
```
Returns comprehensive cost summary and token usage.

#### Model Pricing
```bash
GET /pricing
```
Returns current model pricing table.

#### Model Statistics
```bash
GET /stats
```
Returns model performance and capability information.

## 🧪 Testing

### Automated Testing
Run the test suite to see the API in action:

```bash
python test_api.py
```

This will test:
- Health check
- Model listing
- Single prompt routing
- Batch routing
- Statistics

### Interactive Testing
Use the built-in dashboard for interactive testing:

1. **Access Dashboard**: http://localhost:8000/dashboard
2. **Test Routing**: Enter prompts and see real-time routing decisions
3. **View Metrics**: Monitor performance and cost in real-time
4. **Compare Modes**: Switch between DistilBERT and rule-based routing

### API Testing Examples

#### Test DistilBERT Routing
```bash
curl -X POST "http://localhost:8000/route" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Solve this complex mathematical equation: 2x^2 + 5x - 3 = 0",
       "routing_mode": "distilbert"
     }'
```

#### Test Rule-based Routing
```bash
curl -X POST "http://localhost:8000/route" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is the weather like today?",
       "routing_mode": "rule-based"
     }'
```

#### Test Batch Processing
```bash
curl -X POST "http://localhost:8000/route/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "prompts": [
         "Hello, how are you?",
         "Explain quantum computing principles",
         "Write a Python function to sort a list"
       ],
       "routing_mode": "distilbert"
     }'
```

## 📊 Example Responses

### Single Routing Response (DistilBERT Mode)
```json
{
  "prompt": "What is the weather like today?",
  "recommended_model": "gpt-4o-mini",
  "confidence": 0.95,
  "inference_time_ms": 15.2,
  "reasoning": "[DistilBERT] Selected gpt-4o-mini (Fast and cost-effective for simple tasks) with 95.00% confidence",
  "alternative_models": ["o3", "o4-mini"],
  "llm_response": "I'd be happy to help with weather information! However, I don't have access to real-time weather data...",
  "llm_response_time_ms": 450.3,
  "cost_info": {
    "cost_breakdown": {
      "total_cost": 0.000123,
      "input_tokens": 8,
      "output_tokens": 45,
      "input_cost": 0.000008,
      "output_cost": 0.000115
    },
    "model": "gpt-4o-mini",
    "session_id": "session_123"
  }
}
```

### Batch Routing Response
```json
{
  "results": [
    {
      "prompt": "What is 2+2?",
      "recommended_model": "gpt-4o-mini",
      "confidence": 0.98,
      "inference_time_ms": 8.5,
      "reasoning": "[DistilBERT] Selected gpt-4o-mini (Fast and cost-effective for simple tasks) with 98.00% confidence",
      "alternative_models": ["o3", "o4-mini"],
      "llm_response": "2 + 2 = 4",
      "llm_response_time_ms": 320.1,
      "cost_info": {
        "cost_breakdown": {
          "total_cost": 0.000045,
          "input_tokens": 4,
          "output_tokens": 3,
          "input_cost": 0.000004,
          "output_cost": 0.000041
        }
      }
    }
  ],
  "total_time_ms": 42.5
}
```

### Metrics Response
```json
{
  "total_requests": 150,
  "successful_requests": 148,
  "failed_requests": 2,
  "avg_response_time_ms": 1250.5,
  "success_rate": 98.67,
  "startup_test_completed": true
}
```

### Cost Summary Response
```json
{
  "total_cost_usd": 0.0456,
  "total_requests": 150,
  "cost_by_model": {
    "gpt-4o-mini": 0.0123,
    "o3": 0.0234,
    "o4-mini": 0.0099
  },
  "token_usage": {
    "total_input_tokens": 1250,
    "total_output_tokens": 3400,
    "total_tokens": 4650
  }
}
```

## 🔧 Model Information

- **Architecture**: DistilBERT + Classification Head
- **Training Accuracy**: 100% (on test set)
- **Model Size**: 255MB (optimized for production)
- **Inference Device**: CPU/GPU (auto-detected)
- **Training Time**: ~3 minutes on Apple Silicon M3 Pro
- **Optimization**: Model quantization and PyTorch compilation for faster inference

## 🎯 Use Cases

This API is perfect for:
- **LLM Gateway Teams**: Testing intelligent routing capabilities
- **Research**: Comparing rule-based vs ML-based routing
- **Prototyping**: Building smart routing features
- **Production**: Real-world LLM routing with cost optimization
- **Evaluation**: Assessing routing accuracy and performance
- **Cost Management**: Track and optimize LLM usage costs

## 📈 Performance

- **Inference Time**: ~15-25ms per prompt
- **Batch Processing**: Efficient handling of multiple prompts
- **Memory Usage**: ~500MB RAM (including model)
- **Concurrent Requests**: Supports multiple simultaneous requests
- **Throughput**: 100+ requests/minute on standard hardware
- **Availability**: 99.9% uptime with proper deployment

## 🔍 How It Works

### DistilBERT Routing Mode
1. **Prompt Analysis**: The fine-tuned DistilBERT model analyzes the semantic content of the prompt
2. **Feature Extraction**: Extracts complex patterns that rule-based systems might miss
3. **Model Selection**: Routes to the most appropriate LLM based on learned patterns
4. **Confidence Scoring**: Provides confidence levels for routing decisions
5. **Real LLM Call**: Generates actual response from selected LLM
6. **Cost Tracking**: Records token usage and costs

### Rule-based Routing Mode
1. **Keyword Analysis**: Analyzes prompt for specific keywords and patterns
2. **Simple Logic**: Uses predefined rules for model selection
3. **Mock Response**: Generates simulated responses for testing
4. **Fast Processing**: Minimal latency for development and testing

## 📊 Monitoring & Observability

### Built-in Monitoring
- **Real-time Metrics**: Request count, response times, success rates
- **Cost Tracking**: Token usage, cost per model, total spending
- **Health Checks**: Automatic system health monitoring
- **Error Tracking**: Failed requests and error patterns

### Dashboard Features
- **Interactive Testing**: Real-time prompt testing interface
- **Performance Visualization**: Charts and graphs for metrics
- **Cost Analysis**: Spending breakdown by model and time
- **Routing Comparison**: Side-by-side comparison of routing modes

## 🚀 Production Deployment

### Azure App Service (Production)
- **Cost**: ~$50-100/month depending on plan
- **Setup**: Deploy with `./deploy-to-azure-app-service.sh`
- **Features**: Auto-scaling, SSL, custom domains, continuous deployment
- **Integration**: Easy integration with other Azure services
- **Monitoring**: Built-in Azure monitoring and logging
- **Security**: Managed SSL certificates and security updates

### Docker (Local Development)
- **Portable**: Run anywhere Docker is supported
- **Lightweight**: Optimized container image
- **Security**: Non-root user, minimal attack surface
- **Health Checks**: Built-in container health monitoring
- **Fast Setup**: Quick local development environment

## 🔒 Security Features

- **Non-root Container**: Runs as non-privileged user
- **Resource Limits**: CPU and memory limits prevent resource exhaustion
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Environment Variables**: Secure configuration management
- **Health Checks**: Automatic failure detection and recovery

## 🛠️ Maintenance & Support

### Regular Maintenance
1. **Monitor Costs**: Review monthly Azure bills and usage
2. **Update Dependencies**: Keep Python packages updated
3. **Model Updates**: Retrain and redeploy models as needed
4. **Security Patches**: Apply security updates regularly
5. **Performance Tuning**: Optimize based on usage patterns

### Troubleshooting
- **App Service Logs**: `az webapp log tail --name llm-gateway-name --resource-group your-resource-group`
- **Health Checks**: `curl -f https://llm-gateway.azurewebsites.net/health`
- **Model Status**: Check `/health` endpoint for model loading status
- **Cost Analysis**: Use `/costs` endpoint for spending analysis
- **Local Docker**: `docker logs <container-id>` for local development issues

## 📚 Additional Resources

- **API Documentation**: Interactive docs available at `/docs` endpoint
- **Dashboard**: Real-time interface available at `/dashboard` endpoint
- **Docker Setup**: Use `Dockerfile` for local development environment


