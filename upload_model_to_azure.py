#!/usr/bin/env python3
"""
Upload DistilBERT Model to Azure Blob Storage

This script uploads the trained DistilBERT model to Azure Blob Storage
for deployment access.
"""

import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append('src')

from azure_model_storage import AzureModelStorage


def upload_distilbert_model():
    """Upload the DistilBERT model to Azure Blob Storage"""
    
    print("🚀 Uploading DistilBERT Model to Azure Blob Storage")
    print("=" * 60)
    
    # Check if model exists locally
    model_path = "models/distilbert_llm_router"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("💡 Run 'python run_fine_tuning.py' first to train the model")
        return False
    
    # Check for required files
    required_files = [
        "config.json",
        "model.safetensors", 
        "label_encoder.pkl",
        "tokenizer_config.json",
        "vocab.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    try:
        # Initialize Azure Storage
        storage = AzureModelStorage()
        
        # Upload model
        blob_name = "distilbert_llm_router_v1.zip"
        storage.upload_model(model_path, blob_name)
        
        print("\n✅ Model Upload Successful!")
        print(f"📦 Blob Name: {blob_name}")
        print(f"🔗 Container: {storage.container_name}")
        
        # Verify upload
        if storage.model_exists(blob_name):
            print("✅ Upload verified successfully")
            
            # Show how to access in deployment
            print("\n📋 Deployment Instructions:")
            print("1. Add AZURE_STORAGE_CONNECTION_STRING to your .env")
            print("2. The model will be automatically downloaded during startup")
            print(f"3. Blob URL: https://<storage_account>.blob.core.windows.net/{storage.container_name}/{blob_name}")
            
            return True
        else:
            print("❌ Upload verification failed")
            return False
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def main():
    """Main function"""
    
    # Check environment
    if not os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
        print("❌ AZURE_STORAGE_CONNECTION_STRING not found in environment")
        print("\n💡 Setup Instructions:")
        print("1. Create an Azure Storage Account")
        print("2. Go to 'Access keys' in Azure Portal")
        print("3. Copy Connection string")
        print("4. Add to .env file:")
        print("   AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here")
        return
    
    success = upload_distilbert_model()
    
    if success:
        print("\n🎉 Ready for deployment!")
        print("The model will be downloaded automatically during Azure App Service startup")
    else:
        print("\n❌ Upload failed. Please check the errors above.")

if __name__ == "__main__":
    main()
