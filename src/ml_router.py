"""
ML-Based Router for LLM Gateway Smart Routing
Uses machine learning to make routing decisions based on prompt features
"""

import logging
import pickle
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .models import (RoutingDecision, RoutingPriority, RoutingRequest,
                     RoutingResponse, UserPreferences)
from .prompt_analyzer import PromptAnalysis, PromptAnalyzer


class MLRouter:
    """Machine learning-based router for intelligent LLM routing decisions"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML Router
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.prompt_analyzer = PromptAnalyzer()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = []
        self.logger = logging.getLogger(__name__)
        
        if model_path:
            self.load_model(model_path)
    
    def extract_features(self, prompt: str, user_preferences: Optional[UserPreferences] = None) -> Dict[str, Any]:
        """
        Extract features from prompt for ML model
        
        Args:
            prompt: Input prompt
            user_preferences: User preferences for routing
            
        Returns:
            Dictionary of features
        """
        # Get prompt analysis
        analysis = self.prompt_analyzer.analyze_prompt(prompt)
        
        # Basic features
        features = {
            'length_tokens': analysis.length_tokens,
            'length_words': analysis.length_words,
            'has_code': int(analysis.has_code),
            'has_math': int(analysis.has_math),
            'has_questions': int(analysis.has_questions),
            'urgency_count': len(analysis.urgency_indicators),
        }
        
        # Complexity features (one-hot encoded)
        complexity_mapping = {'simple': 0, 'moderate': 1, 'complex': 2}
        features['complexity_simple'] = int(analysis.complexity.value == 'simple')
        features['complexity_moderate'] = int(analysis.complexity.value == 'moderate')
        features['complexity_complex'] = int(analysis.complexity.value == 'complex')
        
        # Domain features (one-hot encoded)
        domain_mapping = {
            'general': 0, 'code': 1, 'math': 2, 'creative': 3, 
            'analysis': 4, 'translation': 5
        }
        for domain in domain_mapping.keys():
            features[f'domain_{domain}'] = int(analysis.domain.value == domain)
        
        # User preference features
        if user_preferences:
            features['priority_cost'] = int(user_preferences.priority.value == 'cost')
            features['priority_speed'] = int(user_preferences.priority.value == 'speed')
            features['priority_quality'] = int(user_preferences.priority.value == 'quality')
            features['priority_balanced'] = int(user_preferences.priority.value == 'balanced')
            features['max_cost'] = user_preferences.max_cost or 0.01
            features['max_response_time'] = user_preferences.max_response_time or 5000
        else:
            features['priority_cost'] = 0
            features['priority_speed'] = 0
            features['priority_quality'] = 0
            features['priority_balanced'] = 1
            features['max_cost'] = 0.01
            features['max_response_time'] = 5000
        
        # Text-based features
        features['avg_word_length'] = np.mean([len(word) for word in prompt.split()]) if prompt.split() else 0
        features['sentence_count'] = len([s for s in prompt.split('.') if s.strip()])
        features['special_char_ratio'] = sum(1 for c in prompt if not c.isalnum() and c != ' ') / len(prompt) if prompt else 0
        
        return features
    
    def prepare_training_data(self, training_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for ML model
        
        Args:
            training_data: List of training examples with 'prompt', 'user_preferences', 'selected_model', 'quality_score'
            
        Returns:
            Tuple of (features, labels)
        """
        features_list = []
        labels = []
        
        for example in training_data:
            # Extract features
            features = self.extract_features(
                example['prompt'], 
                example.get('user_preferences')
            )
            
            # Get label (selected model)
            label = example['selected_model']
            
            features_list.append(list(features.values()))
            labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(labels)
        
        # Store feature names
        self.feature_names = list(features.keys())
        
        return X, y
    
    def train_model(self, training_data: List[Dict], test_size: float = 0.2, 
                   random_state: int = 42) -> Dict[str, float]:
        """
        Train the ML routing model
        
        Args:
            training_data: List of training examples
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        self.logger.info(f"Training ML router with {len(training_data)} examples")
        
        # Prepare data
        X, y = self.prepare_training_data(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Train model (Random Forest for interpretability and good performance)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            class_weight='balanced'
        )
        
        # Train
        self.model.fit(X_train_scaled, y_train_encoded)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train_encoded, cv=5)
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        metrics = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        self.logger.info(f"Training completed. Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return metrics
    
    def predict(self, prompt: str, user_preferences: Optional[UserPreferences] = None) -> Tuple[str, float]:
        """
        Predict the best model for a given prompt
        
        Args:
            prompt: Input prompt
            user_preferences: User preferences
            
        Returns:
            Tuple of (selected_model, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Extract features
        features = self.extract_features(prompt, user_preferences)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        prediction = self.model.predict(feature_vector_scaled)[0]
        probabilities = self.model.predict_proba(feature_vector_scaled)[0]
        
        # Decode prediction
        selected_model = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence (probability of predicted class)
        confidence = probabilities[prediction]
        
        return selected_model, confidence
    
    def route_request(self, request: RoutingRequest) -> RoutingResponse:
        """
        Route a request using ML model
        
        Args:
            request: Routing request
            
        Returns:
            Routing response with ML-based decision
        """
        try:
            # Get ML prediction
            selected_model, confidence = self.predict(
                request.prompt, 
                request.user_preferences
            )
            
            # Get model configuration for cost/time estimation
            model_configs = self.prompt_analyzer.model_configs
            if selected_model in model_configs:
                config = model_configs[selected_model]
                estimated_cost = self.prompt_analyzer._estimate_cost(len(request.prompt) // 4, selected_model)
                estimated_time = config.get('avg_response_time', 1500)
            else:
                estimated_cost = 0.001
                estimated_time = 1000
            
            # Create routing decision
            decision = RoutingDecision(
                selected_model=selected_model,
                confidence=confidence,
                reasoning=f"ML-based routing with {confidence:.2f} confidence",
                estimated_cost=estimated_cost,
                estimated_time=estimated_time,
                alternative_models=[],
                fallback_model=None
            )
            
            # Create response
            response = RoutingResponse(
                request=request,
                decision=decision,
                analysis={'ml_confidence': confidence},
                processing_time=0
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error in ML routing: {str(e)}")
            # Fallback to rule-based routing
            return self._fallback_routing(request)
    
    def _fallback_routing(self, request: RoutingRequest) -> RoutingResponse:
        """Fallback to rule-based routing if ML fails"""
        from .routing_engine import RoutingEngine
        
        engine = RoutingEngine()
        return engine.route_request(request)
    
    def save_model(self, model_path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if self.model is None:
            return {}
        
        return dict(zip(self.feature_names, self.model.feature_importances_))
    
    def explain_prediction(self, prompt: str, user_preferences: Optional[UserPreferences] = None) -> Dict[str, Any]:
        """Explain the prediction for a given prompt"""
        if self.model is None:
            return {}
        
        # Extract features
        features = self.extract_features(prompt, user_preferences)
        
        # Get feature importance
        importance = self.get_feature_importance()
        
        # Get prediction
        selected_model, confidence = self.predict(prompt, user_preferences)
        
        # Create explanation
        explanation = {
            'selected_model': selected_model,
            'confidence': confidence,
            'top_features': sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5],
            'feature_values': features
        }
        
        return explanation


def create_synthetic_training_data(n_samples: int = 1000) -> List[Dict]:
    """
    Create synthetic training data for ML model
    
    Args:
        n_samples: Number of training examples to generate
        
    Returns:
        List of training examples
    """
    training_data = []
    
    # Define prompt templates and their expected models
    prompt_templates = [
        # Simple questions -> GPT-3.5 Turbo
        ("What is the weather like today?", "gpt-35-turbo"),
        ("What is the capital of {country}?", "gpt-35-turbo"),
        ("How do you make {food}?", "gpt-35-turbo"),
        ("What is {concept}?", "gpt-35-turbo"),
        
        # Code generation -> GPT-4
        ("Write a Python function to {task}", "gpt-4"),
        ("def {function_name}({params}):", "gpt-4"),
        ("Create a {language} class for {purpose}", "gpt-4"),
        ("Implement {algorithm} in {language}", "gpt-4"),
        
        # Math problems -> GPT-4
        ("Solve the equation: {equation}", "gpt-4"),
        ("Calculate {calculation}", "gpt-4"),
        ("What is the {math_concept} of {expression}?", "gpt-4"),
        
        # Complex analysis -> GPT-4
        ("Analyze the {topic} implications of {subject}", "gpt-4"),
        ("Compare and contrast {concept1} and {concept2}", "gpt-4"),
        ("Evaluate the trade-offs between {option1} and {option2}", "gpt-4"),
        
        # Creative tasks -> GPT-4 Turbo
        ("Write a creative story about {topic}", "gpt-4-turbo"),
        ("Create a poem about {subject}", "gpt-4-turbo"),
        ("Design a {concept} for {purpose}", "gpt-4-turbo"),
    ]
    
    # Sample data for template filling
    countries = ["France", "Japan", "Brazil", "Australia", "Canada"]
    foods = ["pasta", "sushi", "pizza", "curry", "bread"]
    concepts = ["machine learning", "blockchain", "quantum computing", "artificial intelligence"]
    tasks = ["sort a list", "find duplicates", "calculate fibonacci", "parse JSON"]
    languages = ["Python", "JavaScript", "Java", "C++"]
    algorithms = ["binary search", "quicksort", "dijkstra", "k-means"]
    equations = ["2x + 5 = 15", "x^2 + 3x - 4 = 0", "sin(x) = 0.5"]
    calculations = ["the derivative of x^3", "the integral of 2x", "the probability of rolling a 6"]
    topics = ["performance", "security", "scalability", "maintainability"]
    
    for i in range(n_samples):
        # Randomly select template
        template_idx = np.random.randint(0, len(prompt_templates))
        template, expected_model = prompt_templates[template_idx]
        
        # Fill template with random data
        prompt = template.format(
            country=np.random.choice(countries),
            food=np.random.choice(foods),
            concept=np.random.choice(concepts),
            task=np.random.choice(tasks),
            function_name=f"func_{i}",
            params="x, y",
            language=np.random.choice(languages),
            purpose="data processing",
            algorithm=np.random.choice(algorithms),
            equation=np.random.choice(equations),
            calculation=np.random.choice(calculations),
            math_concept="derivative",
            expression="x^2 + 2x + 1",
            topic=np.random.choice(topics),
            subject="algorithm selection",
            concept1="microservices",
            concept2="monolith",
            option1="performance",
            option2="simplicity"
        )
        
        # Random user preferences
        priorities = ['cost', 'speed', 'quality', 'balanced']
        priority = np.random.choice(priorities)
        
        user_preferences = UserPreferences(
            priority=RoutingPriority(priority),
            max_cost=np.random.uniform(0.001, 0.01),
            max_response_time=np.random.randint(1000, 5000)
        )
        
        # Add some noise (not all examples follow the template)
        if np.random.random() < 0.1:  # 10% noise
            expected_model = np.random.choice(['gpt-35-turbo', 'gpt-4', 'gpt-4-turbo'])
        
        # Quality score (simulate feedback)
        quality_score = np.random.normal(0.8, 0.1)  # Mean 0.8, std 0.1
        quality_score = max(0.0, min(1.0, quality_score))
        
        training_data.append({
            'prompt': prompt,
            'user_preferences': user_preferences,
            'selected_model': expected_model,
            'quality_score': quality_score
        })
    
    return training_data


# Example usage
if __name__ == "__main__":
    # Create synthetic training data
    training_data = create_synthetic_training_data(1000)
    
    # Train ML router
    ml_router = MLRouter()
    metrics = ml_router.train_model(training_data)
    
    print(f"Training completed with accuracy: {metrics['accuracy']:.3f}")
    print(f"Cross-validation: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")
    
    # Test prediction
    test_prompt = "Write a Python function to sort a list of numbers"
    selected_model, confidence = ml_router.predict(test_prompt)
    print(f"Test prediction: {selected_model} (confidence: {confidence:.3f})")
    
    # Save model
    ml_router.save_model("ml_router_model.pkl")
