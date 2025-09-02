#!/bin/bash

# LLM Gateway API Startup Script for Azure App Service

echo "🚀 Starting LLM Gateway API on Azure App Service..."

# Set working directory to where the app is deployed
cd /home/site/wwwroot

# Check the deployment structure
echo "📁 Current directory: $(pwd)"
echo "📂 Directory contents:"
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
    echo "⚠️ No requirements file found, installing basic dependencies..."
    pip install --user fastapi uvicorn requests python-dotenv transformers torch scikit-learn
fi

# Check if DistilBERT model exists locally
echo "🤖 Checking for DistilBERT model..."
if [ -d "models/distilbert_llm_router" ] && [ -f "models/distilbert_llm_router/model.safetensors" ]; then
    echo "✅ DistilBERT model found locally!"
else
    echo "📥 DistilBERT model not found locally, will try Azure Blob Storage during startup"
fi

# Download DistilBERT model from Azure Blob Storage
echo "🤖 Checking for DistilBERT model..."
if [ ! -d "models/distilbert_llm_router" ] || [ ! -f "models/distilbert_llm_router/model.safetensors" ]; then
    echo "📥 DistilBERT model not found locally, checking Azure Blob Storage..."
    
    # Install Azure Storage SDK if needed
    pip install --user azure-storage-blob
    
    # Try to download from Azure Blob Storage
    python3 -c "
import os
import sys
sys.path.append('src')

try:
    from azure_model_storage import AzureModelStorage
    
    # Check if connection string is available
    if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
        storage = AzureModelStorage()
        blob_name = 'distilbert_llm_router_v1.zip'
        
        if storage.model_exists(blob_name):
            print('📥 Downloading DistilBERT model from Azure Blob Storage...')
            storage.download_model(blob_name, 'models/distilbert_llm_router')
            print('✅ DistilBERT model downloaded successfully!')
        else:
            print('⚠️ DistilBERT model not found in Azure Blob Storage')
            print('💡 Make sure to upload the model first using: python upload_model_to_azure.py')
    else:
        print('⚠️ AZURE_STORAGE_CONNECTION_STRING not found, skipping download')
        
except Exception as e:
    print(f'❌ Error downloading model: {e}')
    print('💡 This is normal if the model hasn\'t been uploaded yet')
"
    
    # Check if download was successful
    if [ -f "models/distilbert_llm_router/model.safetensors" ]; then
        echo "✅ DistilBERT model ready!"
    else
        echo "⚠️ DistilBERT model not available, will use Lightweight Router"
    fi
else
    echo "✅ DistilBERT model found locally!"
fi

# Start the API
echo "🌐 Starting FastAPI application..."
echo "📚 API docs will be available at: https://llm-gateway-smart-routing-26976.azurewebsites.net/docs"
echo "🔍 Health check: https://llm-gateway-smart-routing-26976.azurewebsites.net/"
echo ""

# Navigate to API directory if it exists, otherwise assume we're in the right place
if [ -d "api" ]; then
    cd api
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
else
    echo "📁 API directory not found, trying to run from current directory..."
    python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
fi


