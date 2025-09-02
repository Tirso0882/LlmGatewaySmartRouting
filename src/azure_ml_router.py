"""
Azure ML-Integrated Router for LLM Gateway Smart Routing

This module integrates with Azure ML Workspace for:
- Model training and experimentation
- Model deployment and serving
- Experiment tracking with MLflow
- Model versioning and management
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from azureml.core import Environment, Experiment, Model, Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
# from azureml.train.automl import AutoMLConfig  # Commented out due to dependency conflicts
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .models import (RoutingDecision, RoutingPriority, RoutingRequest,
                     RoutingResponse, UserPreferences)
from .prompt_analyzer import PromptAnalysis, PromptAnalyzer


class AzureMLRouter:
    """
    Azure ML-integrated router that uses Azure ML Workspace for training and deployment.
    """
    
    def __init__(self, workspace_name: str = "llm-gateway-ml-workspace", 
                 resource_group: str = "rg-llm-gateway-smart-routing",
                 subscription_id: str = "9f0783db-f1a1-48a9-b104-adf241e7c591"):
        """
        Initialize Azure ML Router with workspace connection.
        
        Args:
            workspace_name: Name of the Azure ML workspace
            resource_group: Resource group containing the workspace
            subscription_id: Azure subscription ID
        """
        self.workspace_name = workspace_name
        self.resource_group = resource_group
        self.subscription_id = subscription_id
        
        # Initialize Azure ML workspace
        try:
            self.workspace = Workspace(
                subscription_id=subscription_id,
                resource_group=resource_group,
                workspace_name=workspace_name
            )
            logging.info(f"Connected to Azure ML workspace: {workspace_name}")
        except Exception as e:
            logging.error(f"Failed to connect to Azure ML workspace: {e}")
            self.workspace = None
        
        # Initialize local components
        self.prompt_analyzer = PromptAnalyzer()
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Azure ML experiment
        self.experiment = None
        if self.workspace:
            self.experiment = Experiment(self.workspace, "llm-routing-experiment")
    
    def create_training_dataset(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Create training dataset and upload to Azure ML workspace.
        
        Args:
            n_samples: Number of training samples to generate
            
        Returns:
            DataFrame with training data
        """
        from .ml_router import create_synthetic_training_data
        
        logging.info(f"Creating training dataset with {n_samples} samples...")
        
        # Generate synthetic data
        training_data = create_synthetic_training_data(n_samples)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        # Upload to Azure ML workspace
        if self.workspace:
            try:
                # Register dataset
                dataset_name = f"llm-routing-data-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                dataset = self.workspace.datasets.register_pandas_dataframe(
                    df, dataset_name, "LLM Routing Training Data"
                )
                logging.info(f"Registered dataset: {dataset_name}")
                
                # Save to blob storage
                datastore = self.workspace.get_default_datastore()
                dataset_path = f"datasets/{dataset_name}.csv"
                df.to_csv(dataset_path, index=False)
                datastore.upload_files(
                    files=[dataset_path],
                    target_path="llm-routing-data",
                    overwrite=True
                )
                logging.info(f"Uploaded dataset to blob storage: {dataset_path}")
                
            except Exception as e:
                logging.warning(f"Failed to upload dataset to Azure ML: {e}")
        
        return df
    
    def train_model_azure_ml(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model using Azure ML for better performance.
        
        Args:
            training_data: Training dataset
            
        Returns:
            Training results and metrics
        """
        if not self.workspace:
            logging.error("Azure ML workspace not available")
            return {}
        
        try:
            # Prepare data
            X = training_data.drop(['prompt', 'selected_model'], axis=1)
            y = training_data['selected_model']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model locally first
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log experiment to Azure ML
            run = self.experiment.start_logging()
            run.log("accuracy", accuracy)
            run.log("n_features", len(X.columns))
            run.log("n_samples", len(training_data))
            run.log("model_type", "RandomForest")
            run.complete()
            
            # Register model
            model_path = "models/llm_routing_model_azure.pkl"
            os.makedirs("models", exist_ok=True)
            joblib.dump(self.model, model_path)
            
            model = Model.register(
                workspace=self.workspace,
                model_path=model_path,
                model_name="llm-routing-model-azure",
                description="LLM Routing Model (Azure ML Training)"
            )
            
            logging.info(f"Azure ML training completed. Accuracy: {accuracy:.3f}")
            
            return {
                'model': model,
                'accuracy': accuracy,
                'run_id': run.id,
                'experiment_name': self.experiment.name
            }
            
        except Exception as e:
            logging.error(f"Azure ML training failed: {e}")
            return {}
    
    def train_model_local(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train model locally and upload to Azure ML workspace.
        
        Args:
            training_data: Training dataset
            
        Returns:
            Training results and metrics
        """
        logging.info("Training model locally...")
        
        # Prepare data
        X = training_data.drop(['prompt', 'selected_model'], axis=1)
        y = training_data['selected_model']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Upload to Azure ML if available
        if self.workspace:
            try:
                # Save model locally
                model_path = "models/llm_routing_model.pkl"
                os.makedirs("models", exist_ok=True)
                joblib.dump(self.model, model_path)
                
                # Register model
                model = Model.register(
                    workspace=self.workspace,
                    model_path=model_path,
                    model_name="llm-routing-model-local",
                    description="LLM Routing Model (Local Training)"
                )
                
                # Log experiment
                run = self.experiment.start_logging()
                run.log("accuracy", accuracy)
                run.log("cv_mean", cv_scores.mean())
                run.log("cv_std", cv_scores.std())
                run.log("n_features", len(self.feature_names))
                run.log("n_samples", len(training_data))
                run.complete()
                
                logging.info(f"Model registered in Azure ML: {model.name}")
                
            except Exception as e:
                logging.warning(f"Failed to upload model to Azure ML: {e}")
        
        return {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_names': self.feature_names,
            'n_samples': len(training_data)
        }
    
    def deploy_model(self, model_name: str = "llm-routing-model") -> Optional[str]:
        """
        Deploy model to Azure Container Instances for real-time inference.
        
        Args:
            model_name: Name of the model to deploy
            
        Returns:
            Service URL if successful, None otherwise
        """
        if not self.workspace:
            logging.error("Azure ML workspace not available")
            return None
        
        try:
            # Get the model
            model = Model(self.workspace, model_name)
            
            # Create scoring script
            scoring_script = """
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model, scaler, label_encoder, feature_names
    model_path = Model.get_model_path('llm-routing-model')
    model = joblib.load(model_path)
    scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
    label_encoder = joblib.load(model_path.replace('.pkl', '_encoder.pkl'))
    feature_names = joblib.load(model_path.replace('.pkl', '_features.pkl'))

def run(raw_data):
    try:
        # Parse input
        data = pd.read_json(raw_data)
        features = data[feature_names].values
        
        # Preprocess
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)
        probability = model.predict_proba(features_scaled)
        
        # Decode
        model_name = label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probability)
        
        return {
            'selected_model': model_name,
            'confidence': float(confidence),
            'probabilities': probability.tolist()
        }
    except Exception as e:
        return {'error': str(e)}
"""
            
            # Save scoring script
            os.makedirs("deployment", exist_ok=True)
            with open("deployment/score.py", "w") as f:
                f.write(scoring_script)
            
            # Create environment
            env = Environment.from_conda_specification(
                name="llm-routing-env",
                file_path="environment.yml"
            )
            
            # Create inference config
            inference_config = InferenceConfig(
                entry_script="deployment/score.py",
                environment=env
            )
            
            # Deploy to ACI
            deployment_config = AciWebservice.deploy_configuration(
                cpu_cores=1,
                memory_gb=1,
                auth_enabled=True
            )
            
            service = Model.deploy(
                self.workspace,
                "llm-routing-service",
                [model],
                inference_config,
                deployment_config
            )
            
            service.wait_for_deployment(show_output=True)
            
            logging.info(f"Model deployed successfully: {service.scoring_uri}")
            return service.scoring_uri
            
        except Exception as e:
            logging.error(f"Model deployment failed: {e}")
            return None
    
    def route_request(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route request using Azure ML deployed model or local fallback.
        
        Args:
            request: Routing request with prompt and preferences
            
        Returns:
            Routing decision with selected model and confidence
        """
        try:
            # Extract features
            features = self._extract_features(request)
            
            # Try Azure ML service first
            if hasattr(self, 'service_url') and self.service_url:
                return self._route_with_azure_service(request, features)
            
            # Fallback to local model
            return self._route_with_local_model(request, features)
            
        except Exception as e:
            logging.error(f"Routing failed: {e}")
            # Fallback to rule-based routing
            return self.prompt_analyzer.route_request(request.prompt, request.user_preferences)
    
    def _extract_features(self, request: RoutingRequest) -> np.ndarray:
        """Extract features from routing request."""
        # Analyze prompt
        analysis = self.prompt_analyzer.analyze_prompt(request.prompt)
        
        # Create feature vector
        features = []
        
        # Text features
        features.extend([
            len(request.prompt.split()),  # word_count
            len(request.prompt),  # char_count
            len(request.prompt.split('.')) - 1,  # sentence_count
            analysis.token_estimate,  # token_estimate
            analysis.complexity.value,  # complexity
        ])
        
        # Domain features (one-hot encoded)
        domain_features = [0] * 6  # GENERAL, CODE, MATH, CREATIVE, ANALYSIS, TRANSLATION
        domain_features[analysis.domain.value] = 1
        features.extend(domain_features)
        
        # Pattern features
        features.extend([
            analysis.has_code,
            analysis.has_math,
            analysis.has_questions,
            analysis.is_urgent,
        ])
        
        # User preference features
        if request.user_preferences:
            features.extend([
                request.user_preferences.priority.value,
                request.user_preferences.max_cost or 0.1,
                request.user_preferences.max_response_time or 5000,
            ])
        else:
            features.extend([0, 0.1, 5000])  # defaults
        
        return np.array(features).reshape(1, -1)
    
    def _route_with_azure_service(self, request: RoutingRequest, features: np.ndarray) -> RoutingDecision:
        """Route using Azure ML deployed service."""
        import json

        import requests

        # Prepare request
        service_request = {
            'data': features.tolist(),
            'feature_names': self.feature_names
        }
        
        # Call service
        response = requests.post(
            self.service_url,
            json=service_request,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            return RoutingDecision(
                selected_model=result['selected_model'],
                confidence=result['confidence'],
                reasoning=f"Azure ML Service (confidence: {result['confidence']:.3f})",
                estimated_cost=self.prompt_analyzer._estimate_cost(
                    len(request.prompt) // 4, result['selected_model']
                ),
                estimated_response_time=1000  # placeholder
            )
        else:
            raise Exception(f"Service call failed: {response.status_code}")
    
    def _route_with_local_model(self, request: RoutingRequest, features: np.ndarray) -> RoutingDecision:
        """Route using local trained model."""
        if self.model is None:
            raise Exception("No trained model available")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Decode prediction
        selected_model = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(probabilities)
        
        return RoutingDecision(
            selected_model=selected_model,
            confidence=confidence,
            reasoning=f"Local ML Model (confidence: {confidence:.3f})",
            estimated_cost=self.prompt_analyzer._estimate_cost(
                len(request.prompt) // 4, selected_model
            ),
            estimated_response_time=1000  # placeholder
        )


def create_environment_yml():
    """Create environment.yml for Azure ML deployment."""
    env_content = """
name: llm-routing-env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.8
  - pip
  - pip:
    - azureml-core
    - scikit-learn==1.7.1
    - pandas==2.3.2
    - numpy==2.3.2
    - joblib==1.5.1
    - requests
"""
    
    with open("environment.yml", "w") as f:
        f.write(env_content)
    
    logging.info("Created environment.yml for Azure ML deployment")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create environment file
    create_environment_yml()
    
    # Initialize Azure ML Router
    router = AzureMLRouter()
    
    # Create and train model
    training_data = router.create_training_dataset(1000)
    results = router.train_model_local(training_data)
    
    print(f"Training completed: {results}")
    
    # Test routing
    request = RoutingRequest(
        prompt="Write a Python function to sort a list",
        user_preferences=UserPreferences(priority=RoutingPriority.QUALITY)
    )
    
    decision = router.route_request(request)
    print(f"Routing decision: {decision}")
