#!/bin/bash

# Azure App Service Deployment Script
# This script deploys LLM Gateway to Azure App Service with ACR

set -e

# Configuration variables
RESOURCE_GROUP="smart-router-rg"
LOCATION="eastus"
ACR_NAME="smartrouteracr$(date +%s | tail -c 6)"
APP_SERVICE_PLAN="smart-router-plan"
APP_NAME="smart-router-$(date +%s | tail -c 6)"
IMAGE_NAME="smart-router:latest"

echo "🚀 Setting up Azure App Service for Smart Router..."

# 1. Create Resource Group
echo "📦 Creating resource group: $RESOURCE_GROUP"
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# 2. Create Azure Container Registry
echo "🏗️ Creating Azure Container Registry: $ACR_NAME"
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# 3. Get ACR login server
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
echo "📋 ACR Login Server: $ACR_LOGIN_SERVER"

# 4. Build and push Docker image
echo "🔨 Building and pushing Docker image..."
az acr build \
    --registry $ACR_NAME \
    --image $IMAGE_NAME \
    --file Dockerfile \
    .

# 5. Create App Service Plan
echo "📋 Creating App Service Plan..."
az appservice plan create \
    --name $APP_SERVICE_PLAN \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --is-linux \
    --sku B2  # 2 cores, 3.5GB RAM

# 6. Create Web App
echo "🌐 Creating Web App..."
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan $APP_SERVICE_PLAN \
    --name $APP_NAME \
    --deployment-container-image-name $ACR_LOGIN_SERVER/$IMAGE_NAME

# 7. Configure App Service
echo "⚙️ Configuring App Service..."

# Enable continuous deployment from ACR
az webapp config container set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --docker-custom-image-name $ACR_LOGIN_SERVER/$IMAGE_NAME \
    --docker-registry-server-url https://$ACR_LOGIN_SERVER \
    --docker-registry-server-user $(az acr credential show --name $ACR_NAME --query username --output tsv) \
    --docker-registry-server-password $(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

# Set environment variables
echo "🔐 Setting environment variables..."

# Check if .env file exists
if [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env file..."
    while IFS= read -r line; do
        if [[ ! $line =~ ^[[:space:]]*# ]] && [[ -n $line ]]; then
            key=$(echo $line | cut -d'=' -f1)
            value=$(echo $line | cut -d'=' -f2-)
            echo "   Setting $key"
            az webapp config appsettings set \
                --name $APP_NAME \
                --resource-group $RESOURCE_GROUP \
                --settings "$key=$value"
        fi
    done < .env
else
    echo "⚠️ No .env file found. Setting default environment variables."
fi

# Set required environment variables
az webapp config appsettings set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --settings \
        MODEL_PATH=/app/models/distilbert_llm_router \
        PYTHONPATH=/app \
        WEBSITES_PORT=8000 \
        PORT=8000

# Configure startup command for Azure App Service
echo "⚙️ Configuring startup command..."
az webapp config set \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --startup-file "python -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1"

# 8. Get the public URL
echo "🌐 Getting public URL..."
PUBLIC_URL="https://$APP_NAME.azurewebsites.net"

echo "✅ Deployment completed successfully!"
echo "📋 Deployment Summary:"
echo "   Resource Group: $RESOURCE_GROUP"
echo "   Container Registry: $ACR_NAME"
echo "   App Service Plan: $APP_SERVICE_PLAN"
echo "   Web App: $APP_NAME"
echo "   Public URL: $PUBLIC_URL"
echo "   Health Check: $PUBLIC_URL/health"
echo "   Dashboard: $PUBLIC_URL/dashboard"
echo "   API Docs: $PUBLIC_URL/docs"

# 9. Test the deployment
echo "🧪 Testing deployment..."
sleep 30  # Wait for app to start

if curl -f $PUBLIC_URL/health; then
    echo "✅ Health check passed!"
else
    echo "⚠️ Health check failed. Check app logs:"
    az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP
fi

echo "🎉 Setup complete! LLM Gateway is now running on Azure App Service."
echo "🔗 Access the app at: $PUBLIC_URL"
