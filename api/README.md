# LLM Gateway API - Fine-tuned Routing

A simple FastAPI application that demonstrates intelligent LLM routing using a fine-tuned DistilBERT model.

## 🚀 Features

- **Smart Routing**: Uses fine-tuned DistilBERT to route prompts to the best LLM
- **High Accuracy**: 100% accuracy on test set
- **Real-time Inference**: Fast routing decisions
- **Batch Processing**: Route multiple prompts efficiently
- **Confidence Scoring**: Get confidence levels for routing decisions
- **Alternative Models**: See other model options

## 📋 Available Models

| Model | Description | Cost/1K tokens | Response Time | Best For |
|-------|-------------|----------------|---------------|----------|
| `o3` | High-accuracy for complex reasoning | $0.060 | 5000ms | Complex analysis, STEM problems |
| `gpt-4o-mini` | Fast and cost-effective | $0.0001 | 400ms | Simple queries, quick responses |
| `o4-mini` | Balanced performance | $0.015 | 2000ms | Moderate complexity tasks |

## 🛠️ Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API**:
   ```bash
   python main.py
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - Health Check: http://localhost:8000/

## 📡 API Endpoints

### Health Check
```bash
GET /
```
Returns API status and model availability.

### Get Available Models
```bash
GET /models
```
Returns list of available models and their configurations.

### Route Single Prompt
```bash
POST /route
```
```json
{
  "prompt": "What is the weather like today?",
  "user_id": "optional",
  "session_id": "optional"
}
```

### Route Multiple Prompts (Batch)
```bash
POST /route/batch
```
```json
{
  "prompts": [
    "What is 2+2?",
    "Explain quantum computing",
    "Write a haiku"
  ]
}
```

### Get Model Statistics
```bash
GET /stats
```
Returns model performance and capability information.

## 🧪 Testing

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

## 📊 Example Responses

### Single Routing Response
```json
{
  "prompt": "What is the weather like today?",
  "recommended_model": "gpt-4o-mini",
  "confidence": 0.95,
  "inference_time_ms": 15.2,
  "reasoning": "Selected gpt-4o-mini (Fast and cost-effective for simple tasks) with 95.00% confidence",
  "alternative_models": ["o3", "o4-mini"]
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
      "reasoning": "Selected gpt-4o-mini (Fast and cost-effective for simple tasks) with 98.00% confidence",
      "alternative_models": ["o3", "o4-mini"]
    }
  ],
  "total_time_ms": 42.5
}
```

## 🔧 Model Information

- **Architecture**: DistilBERT + Classification Head
- **Training Accuracy**: 100% (on test set)
- **Model Size**: 255MB
- **Inference Device**: CPU/GPU (auto-detected)
- **Training Time**: ~3 minutes on Apple Silicon M3 Pro

## 🎯 Use Cases

This API is perfect for:
- **LLM Gateway Teams**: Testing intelligent routing capabilities
- **Research**: Comparing rule-based vs ML-based routing
- **Prototyping**: Building smart routing features
- **Evaluation**: Assessing routing accuracy and performance

## 📈 Performance

- **Inference Time**: ~15-25ms per prompt
- **Batch Processing**: Efficient handling of multiple prompts
- **Memory Usage**: ~500MB RAM (including model)
- **Concurrent Requests**: Supports multiple simultaneous requests

## 🔍 How It Works

1. **Prompt Analysis**: The fine-tuned DistilBERT model analyzes the semantic content of the prompt
2. **Feature Extraction**: Extracts complex patterns that rule-based systems might miss
3. **Model Selection**: Routes to the most appropriate LLM based on learned patterns
4. **Confidence Scoring**: Provides confidence levels for routing decisions
5. **Reasoning**: Explains why a particular model was selected

## 🚀 Next Steps

For production deployment:
1. Add authentication and rate limiting
2. Implement caching for repeated prompts
3. Add monitoring and logging
4. Scale horizontally for high traffic
5. Add fallback mechanisms for model failures


