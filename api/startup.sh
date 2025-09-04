#!/bin/bash

# LLM Gateway API Startup Script for Azure App Service

echo "🚀 Starting LLM Gateway via GitHub Actions"
echo "========================================="

# Set working directory to where the app is deployed
cd /home/site/wwwroot

# Check the deployment structure
echo "📁 Current directory: $(pwd)"
echo "📂 Files present:"
ls -la

# Install pip first if not available
echo "📦 Setting up Python environment..."
if ! command -v pip &> /dev/null; then
    echo "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies
echo "📦 Installing dependencies..."
if [ -f "api/requirements_api.txt" ]; then
    pip install --user -r api/requirements_api.txt
elif [ -f "requirements_api.txt" ]; then
    pip install --user requirements_api.txt
else
    echo "⚠️ No requirements.txt found, installing basic dependencies..."
    pip install --user fastapi uvicorn requests python-dotenv transformers torch scikit-learn
fi

# Check for ALL DistilBERT model files
echo "🤖 Checking for ALL DistilBERT model files..."
required_files=(
    "models/distilbert_llm_router/model.safetensors"
    "models/distilbert_llm_router/training_history.json"
    "models/distilbert_llm_router/label_encoder.pkl"
    "models/distilbert_llm_router/vocab.txt"
    "models/distilbert_llm_router/special_tokens_map.json"
    "models/distilbert_llm_router/tokenizer_config.json"
    "models/distilbert_llm_router/config.json"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file found: $(ls -lh "$file")"
    else
        echo "❌ $file NOT found!"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo "❌ Missing required files for startup: ${missing_files[*]}"
    echo "Available in models/distilbert_llm_router/: $(ls -la models/distilbert_llm_router/ 2>/dev/null || echo 'not found')"
    echo "⚠️ DistilBERT model loading will fail"
else
    echo "🎉 All DistilBERT model files verified for startup!"
fi

# Show the actual model path that will be used
echo "Model path: $(pwd)/models/distilbert_llm_router"

# Start the API
echo "🌐 Starting FastAPI application..."

# Navigate to API directory if it exists, otherwise assume we're in the right place
if [ -d "api" ]; then
    cd api
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
else
    echo "📁 API directory not found, trying to run from current directory..."
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
fi


