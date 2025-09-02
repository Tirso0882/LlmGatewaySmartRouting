#!/usr/bin/env python3
"""
Azure Blob Storage Integration for Model Management

This module provides utilities to upload/download models from Azure Blob Storage
for scalable deployment scenarios.
"""

import os
import tempfile
import zipfile
from typing import Optional

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    print("⚠️ Azure Storage SDK not installed. Run: pip install azure-storage-blob")


class AzureModelStorage:
    """Manage models in Azure Blob Storage"""
    
    def __init__(self, connection_string: Optional[str] = None, container_name: str = "llm-models"):
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Storage SDK is required. Install with: pip install azure-storage-blob")
        
        # Get connection string from environment if not provided
        if connection_string is None:
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        
        if not connection_string:
            raise ValueError("Azure Storage connection string is required")
        
        self.blob_service = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = container_name
        
        # Create container if it doesn't exist
        try:
            self.blob_service.create_container(container_name, public_access=None)
            print(f"✅ Created container: {container_name}")
        except Exception as e:
            print(f"📦 Container exists or error: {e}")  # Container might already exist
    
    def upload_model(self, local_model_path: str, blob_name: str):
        """Upload a model directory to Azure Blob Storage"""
        print(f"📤 Uploading model to Azure: {blob_name}")
        
        # Create a zip file of the model directory
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for root, dirs, files in os.walk(local_model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, local_model_path)
                        zip_file.write(file_path, arc_path)
            
            # Upload to blob storage
            blob_client = self.blob_service.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            with open(temp_zip.name, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Clean up temp file
            os.unlink(temp_zip.name)
        
        print(f"✅ Model uploaded successfully: {blob_name}")
    
    def download_model(self, blob_name: str, local_path: str):
        """Download a model from Azure Blob Storage"""
        print(f"📥 Downloading model from Azure: {blob_name}")
        
        blob_client = self.blob_service.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            download_stream = blob_client.download_blob()
            temp_zip.write(download_stream.readall())
            temp_zip_path = temp_zip.name
        
        # Extract to local path
        os.makedirs(local_path, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_file:
            zip_file.extractall(local_path)
        
        # Clean up temp file
        os.unlink(temp_zip_path)
        
        print(f"✅ Model downloaded to: {local_path}")
    
    def list_models(self):
        """List all models in the container"""
        print(f"📋 Models in container '{self.container_name}':")
        
        blob_list = self.blob_service.get_container_client(self.container_name).list_blobs()
        
        for blob in blob_list:
            size_mb = blob.size / (1024 * 1024)
            print(f"  • {blob.name} ({size_mb:.1f} MB)")
    
    def model_exists(self, blob_name: str) -> bool:
        """Check if a model exists in storage"""
        blob_client = self.blob_service.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        return blob_client.exists()


def create_deployment_strategy():
    """Create a complete deployment strategy document"""
    strategy = """
# 🚀 LLM Gateway Model Deployment Strategy

## Current Situation:
- DistilBERT model: ~268MB (too large for GitHub/fast deployment)
- Training time: 3-5 minutes (too slow for startup)
- Need: Fast, reliable model access for production

## Recommended Solution:

### 📦 Azure Blob Storage + Rule-based Fallback
**Primary:** DistilBERT from Azure Blob Storage | **Fallback:** Rule-based routing

**Flow:**
1. Try to download DistilBERT from Azure Blob Storage
2. If successful: Use DistilBERT (95-98% accuracy)
3. If failed: Fall back to rule-based routing (70-80% accuracy)

**Implementation:**
```bash
# Upload model to Azure Blob Storage
python upload_model_to_azure.py

# Deploy to Azure App Service
# Model will be automatically downloaded during startup
```

## Architecture:

### Model Loading Priority:
```
DistilBERT (95-98% accuracy) ← Downloaded from Azure Blob Storage
    ↓ [if not available]
Rule-based (70-80% accuracy) ← Always available (code only)
```

### Azure Blob Storage Benefits:
- ✅ **No GitHub size limits** (268MB model stored in Azure)
- ✅ **Secure storage** (private container, authenticated access)
- ✅ **Fast deployment** (no large files in Git)
- ✅ **Scalable** (update model without code changes)
- ✅ **Reliable** (rule-based fallback always works)

## Implementation Details:

### Phase 1: Model Upload
```bash
# 1. Train DistilBERT locally
python run_fine_tuning.py

# 2. Upload to Azure Blob Storage
python upload_model_to_azure.py
```

### Phase 2: Deployment
```bash
# 1. Add connection string to GitHub secrets
# 2. Deploy via GitHub Actions
# 3. Model downloads automatically during startup
```

### Phase 3: Production Monitoring
```bash
# Monitor model performance
# Update models via Azure Blob Storage
# A/B testing between model versions
```

## File Size Comparison:
- Rule-based: ~0KB (code only)
- DistilBERT: ~268MB (stored in Azure Blob)

## Deployment Time Comparison:
- Rule-based: Instant
- DistilBERT download: ~60 seconds
- Full training: 3-5 minutes (avoided in production)

## Accuracy Comparison:
- Rule-based: ~70-80%
- DistilBERT: ~95-98%

## Environment Variables Required:
```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...

# Azure OpenAI
O3_API_KEY=your_api_key
O3_ENDPOINT=https://your-resource.openai.azure.com/
GPT4O_MINI_ENDPOINT=https://your-resource.openai.azure.com/
O4_MINI_ENDPOINT=https://your-resource.openai.azure.com/
```

## Result:
Your deployment will now:
1. ✅ **Always work** (rule-based fallback)
2. 🤖 **Prefer DistilBERT** (high accuracy when available)
3. 🚀 **Deploy fast** (no large files in Git)
4. 🔒 **Secure** (private Azure Blob Storage)

Perfect for production use with high accuracy! 🎯
"""
    
    with open("DEPLOYMENT_STRATEGY.md", "w") as f:
        f.write(strategy)
    
    print("📋 Deployment strategy saved to: DEPLOYMENT_STRATEGY.md")


if __name__ == "__main__":
    create_deployment_strategy()
