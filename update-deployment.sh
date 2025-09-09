#!/bin/bash

# Quick Update Script for LLM Gateway
# This script only rebuilds and pushes the Docker image to existing ACR

set -e

# Configuration variables (use existing resources)
RESOURCE_GROUP="smart-router-rg"
IMAGE_NAME="smart-router:latest"

echo "🚀 Updating LLM Gateway deployment..."

# 1. Get existing ACR name
echo "🔍 Finding existing ACR..."
ACR_NAME=$(az acr list --resource-group $RESOURCE_GROUP --query "[].name" -o tsv)

if [ -z "$ACR_NAME" ]; then
    echo "❌ No ACR found in resource group: $RESOURCE_GROUP"
    echo "Please run ./deploy-to-azure-app-service.sh first to create the initial deployment"
    exit 1
fi

echo "✅ Found ACR: $ACR_NAME"

# 2. Build and push Docker image
echo "🔨 Building and pushing Docker image to ACR..."
az acr build \
    --registry $ACR_NAME \
    --image $IMAGE_NAME \
    --file Dockerfile \
    .

echo "✅ Docker image built and pushed successfully!"

# 3. Get App Service name
echo "🔍 Finding App Service..."
APP_NAME=$(az webapp list --resource-group $RESOURCE_GROUP --query "[].name" -o tsv)

if [ -z "$APP_NAME" ]; then
    echo "❌ No App Service found in resource group: $RESOURCE_GROUP"
    exit 1
fi

echo "✅ Found App Service: $APP_NAME"

# 4. Get the public URL
PUBLIC_URL="https://$APP_NAME.azurewebsites.net"

echo "🎉 Update completed successfully!"
echo "📋 Update Summary:"
echo "   Resource Group: $RESOURCE_GROUP"
echo "   Container Registry: $ACR_NAME"
echo "   App Service: $APP_NAME"
echo "   Public URL: $PUBLIC_URL"
echo "   Health Check: $PUBLIC_URL/health"
echo "   Dashboard: $PUBLIC_URL/dashboard"
echo "   API Docs: $PUBLIC_URL/docs"

# 5. Test the deployment
echo "🧪 Testing deployment..."
sleep 30

if curl -f $PUBLIC_URL/health; then
    echo "✅ Health check passed!"
else
    echo "⚠️ Health check failed. App may still be starting up."
    echo "Check app logs: az webapp log tail --name $APP_NAME --resource-group $RESOURCE_GROUP"
fi

echo "🎉 Update complete! LLM Gateway has been updated."
echo "🔗 Access the app at: $PUBLIC_URL"
