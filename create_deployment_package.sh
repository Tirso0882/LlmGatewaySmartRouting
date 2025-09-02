#!/bin/bash

# 🚀 LLM Gateway Deployment Package Creator
# This script creates a complete deployment package for Azure App Service

# ❌ You DON'T Need It If:
# Using GitHub Actions with direct repository deployment
# Azure pulls code directly from your GitHub repo
# Model is already in Git LFS and accessible
# ✅ You DO Need It If:
# Manual deployment to Azure App Service
# ZIP deployment instead of Git deployment
# Custom startup scripts or configurations
# Testing deployment package locally

set -e  # Exit on any error

echo "🚀 Creating LLM Gateway Deployment Package"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Clean up any existing deployment package
if [ -d "deployment_package" ]; then
    print_status "Cleaning up existing deployment package..."
    rm -rf deployment_package
fi

if [ -f "llm_gateway_deployment.zip" ]; then
    print_status "Removing existing ZIP file..."
    rm -f llm_gateway_deployment.zip
fi

# Create deployment directory structure
print_status "Creating deployment directory structure..."
mkdir -p deployment_package/static
mkdir -p deployment_package/models

print_success "Directory structure created"

# Copy main API file
print_status "Copying main API file..."
if [ -f "api/main.py" ]; then
    cp api/main.py deployment_package/main.py
    print_success "main.py copied"
else
    print_error "api/main.py not found!"
    exit 1
fi

# Copy supporting files with error handling
print_status "Copying supporting files..."

# Mock LLM responses
if [ -f "api/mock_llm_responses.py" ]; then
    cp api/mock_llm_responses.py deployment_package/
    print_success "mock_llm_responses.py copied"
else
    print_warning "api/mock_llm_responses.py not found"
fi

# Real LLM integration
if [ -f "api/real_llm_integration_example.py" ]; then
    cp api/real_llm_integration_example.py deployment_package/
    print_success "real_llm_integration_example.py copied"
else
    print_warning "api/real_llm_integration_example.py not found"
fi

# Azure model storage
if [ -f "src/azure_model_storage.py" ]; then
    cp src/azure_model_storage.py deployment_package/
    print_success "azure_model_storage.py copied"
else
    print_warning "src/azure_model_storage.py not found"
fi

# DistilBERT inference
if [ -f "src/distilbert_inference.py" ]; then
    cp src/distilbert_inference.py deployment_package/
    print_success "distilbert_inference.py copied"
else
    print_warning "src/distilbert_inference.py not found"
fi

# Copy DistilBERT model files (if available)
print_status "Copying DistilBERT model files..."
if [ -d "models/distilbert_llm_router" ]; then
    mkdir -p deployment_package/models/distilbert_llm_router
    
    # Copy all model files
    cp -r models/distilbert_llm_router/* deployment_package/models/distilbert_llm_router/ 2>/dev/null || echo "⚠️ Some model files could not be copied"
    
    # Verify key files were copied
    if [ -f "deployment_package/models/distilbert_llm_router/config.json" ] && [ -f "deployment_package/models/distilbert_llm_router/model.safetensors" ]; then
        print_success "DistilBERT model files copied successfully"
        echo "Model files: $(ls -la deployment_package/models/distilbert_llm_router/)"
    else
        print_warning "Some critical model files missing - check deployment package"
    fi
else
    print_warning "models/distilbert_llm_router directory not found - model will be downloaded during startup"
fi

print_status "DistilBERT model is included in the repository via Git LFS"

# Copy static files
print_status "Copying static files..."

# HTML file
if [ -f "api/static/index.html" ]; then
    cp api/static/index.html deployment_package/static/
    print_success "index.html copied"
else
    print_warning "api/static/index.html not found"
fi

# Logo file
if [ -f "api/static/nexus-ai-logo.png" ]; then
    cp api/static/nexus-ai-logo.png deployment_package/static/
    print_success "nexus-ai-logo.png copied"
else
    print_warning "api/static/nexus-ai-logo.png not found"
fi

# Copy requirements
print_status "Copying requirements file..."
if [ -f "api/requirements_api.txt" ]; then
    cp api/requirements_api.txt deployment_package/requirements.txt
    print_success "requirements.txt copied"
else
    print_error "api/requirements_api.txt not found!"
    exit 1
fi

# Create startup script
print_status "Creating startup script..."
cat > deployment_package/startup.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting LLM Gateway via Deployment Package"
echo "=============================================="
echo "Timestamp: $(date)"
echo "Python version: $(python3 --version)"
echo "Working directory: $(pwd)"
echo "Files present: $(ls -la)"

# Install pip first if not available
echo "📦 Setting up Python environment..."
if ! python3 -m pip --version &> /dev/null; then
    echo "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install dependencies
echo "📦 Installing dependencies..."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user -r requirements.txt

# Check if model is available
echo "🤖 Checking for DistilBERT model..."
if [ -d "models/distilbert_llm_router" ] && [ -f "models/distilbert_llm_router/model.safetensors" ]; then
    echo "✅ DistilBERT model found locally - using pre-loaded model!"
    echo "Model size: $(du -sh models/distilbert_llm_router | cut -f1)"
    echo "Model files: $(ls models/distilbert_llm_router/)"
    echo "No download needed - model included in repository via Git LFS!"
else
    echo "❌ DistilBERT model not found - check if Git LFS is working properly"
fi

# Start the application
echo "🌐 Starting FastAPI application..."
echo "Server will be available at: http://0.0.0.0:8000"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
EOF

# Make startup script executable
chmod +x deployment_package/startup.sh
print_success "startup.sh created and made executable"

# Create .deployment file for Azure
print_status "Creating .deployment file..."
cat > deployment_package/.deployment << 'EOF'
[config]
command = bash startup.sh
EOF
print_success ".deployment file created"

# Create startup command file for Azure App Service
print_status "Creating startup command file..."
cat > deployment_package/startup-command.txt << 'EOF'
bash startup.sh
EOF
print_success "startup-command.txt created"

# Show the deployment structure for debugging
echo ""
print_status "Deployment package structure:"
echo "=================================="
find deployment_package -type f | sort

echo ""
print_status "File sizes:"
echo "============"
du -sh deployment_package/*

# Create ZIP file
print_status "Creating ZIP file..."
cd deployment_package

# Create ZIP with exclusions
zip -r ../llm_gateway_deployment.zip . \
    -x "*.pyc" "*.pyo" "__pycache__/*" "*.DS_Store" "*.log" "*.tmp"

cd ..

# Verify ZIP was created
if [ -f "llm_gateway_deployment.zip" ]; then
    ZIP_SIZE=$(du -h llm_gateway_deployment.zip | cut -f1)
    print_success "Deployment package created: llm_gateway_deployment.zip (Size: $ZIP_SIZE)"
    
    echo ""
    print_status "ZIP contents:"
    echo "=============="
    unzip -l llm_gateway_deployment.zip | head -20
    
    echo ""
    print_success "🎉 Deployment package ready!"
    print_status "Next steps:"
    echo "1. Deploy to Azure: az webapp deployment source config-zip --resource-group llm-gateway-rg --name llm-gateway-1756826876 --src ./llm_gateway_deployment.zip"
    echo "2. Or use GitHub Actions by pushing this ZIP file"
    
else
    print_error "Failed to create ZIP file!"
    exit 1
fi

echo ""
print_success "✅ Deployment package creation completed successfully!"
