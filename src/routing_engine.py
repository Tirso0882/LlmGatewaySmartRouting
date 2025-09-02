"""
Main Routing Engine for LLM Gateway Smart Routing
Combines prompt analysis with routing logic to make intelligent decisions
"""

import logging
import time
from typing import Dict, List, Optional

from .models import (DEFAULT_MODELS, LLMEndpoint, ModelStatus, RoutingDecision,
                     RoutingRequest, RoutingResponse, UserPreferences,
                     create_user_preferences_from_dict)
from .prompt_analyzer import PromptAnalysis, PromptAnalyzer


class RoutingEngine:
    """Main routing engine that makes intelligent LLM routing decisions"""
    
    def __init__(self, model_configs: Optional[Dict[str, LLMEndpoint]] = None):
        """
        Initialize the routing engine
        
        Args:
            model_configs: Optional custom model configurations
        """
        self.prompt_analyzer = PromptAnalyzer()
        self.model_configs = model_configs or DEFAULT_MODELS
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.routing_history: List[RoutingResponse] = []
        self.model_performance: Dict[str, Dict] = {}
        
        # Initialize performance tracking
        for model_name in self.model_configs.keys():
            self.model_performance[model_name] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0.0,
                'avg_cost': 0.0
            }

    def route_request(self, request: RoutingRequest) -> RoutingResponse:
        """
        Route a request to the most appropriate LLM model
        
        Args:
            request: The routing request containing prompt and preferences
            
        Returns:
            RoutingResponse with the routing decision and analysis
        """
        start_time = time.time()
        
        try:
            # Step 1: Analyze the prompt
            user_prefs_dict = {}
            if request.user_preferences:
                user_prefs_dict = {
                    'priority': request.user_preferences.priority.value,
                    'max_cost': request.user_preferences.max_cost,
                    'max_response_time': request.user_preferences.max_response_time
                }
            
            analysis = self.prompt_analyzer.analyze_prompt(
                request.prompt, user_prefs_dict
            )
            
            # Step 2: Make routing decision
            decision = self._make_routing_decision(request, analysis)
            
            # Step 3: Create response
            processing_time = int((time.time() - start_time) * 1000)  # Convert to milliseconds
            response = RoutingResponse(
                request=request,
                decision=decision,
                analysis=self._analysis_to_dict(analysis),
                processing_time=processing_time
            )
            
            # Step 4: Log and track
            self._log_routing_decision(response)
            self.routing_history.append(response)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error routing request: {str(e)}")
            # Return fallback decision
            return self._create_fallback_response(request, str(e))

    def _make_routing_decision(self, request: RoutingRequest, analysis: PromptAnalysis) -> RoutingDecision:
        """Make the final routing decision based on analysis and constraints"""
        
        # Get available models (filter by user preferences)
        available_models = self._get_available_models(request.user_preferences)
        
        if not available_models:
            # No models available, use fallback
            return self._create_fallback_decision("No models available")
        
        # Get the recommended model from analysis
        recommended_model = analysis.recommended_model
        
        # Check if recommended model is available
        if recommended_model in available_models:
            model_config = self.model_configs[recommended_model]
            confidence = self._calculate_confidence(analysis, model_config)
            
            return RoutingDecision(
                selected_model=recommended_model,
                confidence=confidence,
                reasoning=analysis.reasoning,
                estimated_cost=analysis.estimated_cost,
                estimated_time=model_config.avg_response_time,
                alternative_models=self._get_alternative_models(available_models, recommended_model),
                fallback_model=self._get_fallback_model(available_models, recommended_model)
            )
        else:
            # Recommended model not available, find best alternative
            best_alternative = self._find_best_alternative(available_models, analysis)
            model_config = self.model_configs[best_alternative]
            confidence = self._calculate_confidence(analysis, model_config)
            
            return RoutingDecision(
                selected_model=best_alternative,
                confidence=confidence,
                reasoning=f"Recommended model {recommended_model} unavailable. Using {best_alternative} as alternative.",
                estimated_cost=self._estimate_cost(request.prompt, best_alternative),
                estimated_time=model_config.avg_response_time,
                alternative_models=self._get_alternative_models(available_models, best_alternative),
                fallback_model=self._get_fallback_model(available_models, best_alternative)
            )

    def _get_available_models(self, user_preferences: Optional[UserPreferences]) -> List[str]:
        """Get list of available models based on user preferences"""
        available_models = []
        
        for model_name, model_config in self.model_configs.items():
            # Check if model is available
            if model_config.status != ModelStatus.AVAILABLE:
                continue
                
            # Check if model is excluded by user
            if user_preferences and model_name in user_preferences.excluded_models:
                continue
                
            available_models.append(model_name)
        
        return available_models

    def _calculate_confidence(self, analysis: PromptAnalysis, model_config: LLMEndpoint) -> float:
        """Calculate confidence score for the routing decision"""
        base_confidence = 0.8
        
        # Adjust based on complexity match
        if analysis.complexity.value == 'simple' and model_config.name == 'GPT-3.5 Turbo':
            base_confidence += 0.1
        elif analysis.complexity.value == 'complex' and model_config.name == 'GPT-4':
            base_confidence += 0.1
        elif analysis.complexity.value == 'moderate' and model_config.name == 'GPT-4 Turbo':
            base_confidence += 0.1
        
        # Adjust based on domain match
        if analysis.domain.value == 'code' and 'code_generation' in model_config.capabilities:
            base_confidence += 0.05
        elif analysis.domain.value == 'math' and 'complex_reasoning' in model_config.capabilities:
            base_confidence += 0.05
        
        # Adjust based on model quality
        base_confidence += (model_config.quality_score - 0.8) * 0.2
        
        return min(base_confidence, 1.0)

    def _get_alternative_models(self, available_models: List[str], selected_model: str) -> List[str]:
        """Get alternative models for the selected model"""
        alternatives = [model for model in available_models if model != selected_model]
        return alternatives[:2]  # Return top 2 alternatives

    def _get_fallback_model(self, available_models: List[str], selected_model: str) -> Optional[str]:
        """Get fallback model (usually the most reliable one)"""
        # Prefer GPT-4 Turbo as fallback (good balance)
        if 'gpt-4-turbo' in available_models and 'gpt-4-turbo' != selected_model:
            return 'gpt-4-turbo'
        elif 'gpt-35-turbo' in available_models and 'gpt-35-turbo' != selected_model:
            return 'gpt-35-turbo'
        elif 'gpt-4' in available_models and 'gpt-4' != selected_model:
            return 'gpt-4'
        
        return None

    def _find_best_alternative(self, available_models: List[str], analysis: PromptAnalysis) -> str:
        """Find the best alternative model when recommended model is unavailable"""
        # Simple heuristic: prefer models based on complexity
        if analysis.complexity.value == 'simple':
            # Prefer GPT-3.5 Turbo for simple queries
            if 'gpt-35-turbo' in available_models:
                return 'gpt-35-turbo'
        elif analysis.complexity.value == 'complex':
            # Prefer GPT-4 for complex queries
            if 'gpt-4' in available_models:
                return 'gpt-4'
        
        # Default to GPT-4 Turbo (good balance)
        if 'gpt-4-turbo' in available_models:
            return 'gpt-4-turbo'
        elif 'gpt-35-turbo' in available_models:
            return 'gpt-35-turbo'
        elif 'gpt-4' in available_models:
            return 'gpt-4'
        
        # Return first available model
        return available_models[0]

    def _estimate_cost(self, prompt: str, model_name: str) -> float:
        """Estimate cost for a prompt with a specific model"""
        if model_name not in self.model_configs:
            return 0.0
        
        model_config = self.model_configs[model_name]
        token_count = len(prompt) // 4  # Rough token estimation
        return (token_count / 1000) * model_config.cost_per_1k_tokens

    def _analysis_to_dict(self, analysis: PromptAnalysis) -> Dict:
        """Convert PromptAnalysis to dictionary"""
        return {
            'length_tokens': analysis.length_tokens,
            'length_words': analysis.length_words,
            'complexity': analysis.complexity.value,
            'domain': analysis.domain.value,
            'has_code': analysis.has_code,
            'has_math': analysis.has_math,
            'has_questions': analysis.has_questions,
            'urgency_indicators': analysis.urgency_indicators,
            'estimated_cost': analysis.estimated_cost,
            'recommended_model': analysis.recommended_model,
            'reasoning': analysis.reasoning
        }

    def _create_fallback_decision(self, reason: str) -> RoutingDecision:
        """Create a fallback routing decision"""
        return RoutingDecision(
            selected_model='gpt-35-turbo',  # Default fallback
            confidence=0.5,
            reasoning=f"Fallback decision: {reason}",
            estimated_cost=0.001,
            estimated_time=1000,
            alternative_models=[],
            fallback_model=None
        )

    def _create_fallback_response(self, request: RoutingRequest, error: str) -> RoutingResponse:
        """Create a fallback response when routing fails"""
        decision = self._create_fallback_decision(f"Error: {error}")
        return RoutingResponse(
            request=request,
            decision=decision,
            analysis={'error': error},
            processing_time=0
        )

    def _log_routing_decision(self, response: RoutingResponse):
        """Log routing decision for monitoring"""
        self.logger.info(
            f"Routing decision: {response.decision.selected_model} "
            f"(confidence: {response.decision.confidence:.2f}, "
            f"cost: ${response.decision.estimated_cost:.4f})"
        )

    def update_model_status(self, model_name: str, status: ModelStatus):
        """Update the status of a model"""
        if model_name in self.model_configs:
            self.model_configs[model_name].status = status
            self.logger.info(f"Updated model {model_name} status to {status.value}")

    def get_routing_statistics(self) -> Dict:
        """Get routing statistics and performance metrics"""
        total_requests = len(self.routing_history)
        successful_routes = sum(1 for r in self.routing_history if r.decision.confidence > 0.5)
        
        model_usage = {}
        for response in self.routing_history:
            model = response.decision.selected_model
            model_usage[model] = model_usage.get(model, 0) + 1
        
        avg_processing_time = 0
        if total_requests > 0:
            avg_processing_time = sum(r.processing_time or 0 for r in self.routing_history) / total_requests
        
        return {
            'total_requests': total_requests,
            'successful_routes': successful_routes,
            'success_rate': successful_routes / total_requests if total_requests > 0 else 0,
            'avg_processing_time_ms': avg_processing_time,
            'model_usage': model_usage,
            'available_models': list(self.model_configs.keys())
        }

    def get_model_performance(self, model_name: str) -> Dict:
        """Get performance metrics for a specific model"""
        return self.model_performance.get(model_name, {})

    def reset_statistics(self):
        """Reset all routing statistics"""
        self.routing_history = []
        for model_name in self.model_performance:
            self.model_performance[model_name] = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'avg_response_time': 0.0,
                'avg_cost': 0.0
            }


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create routing engine
    engine = RoutingEngine()
    
    # Test routing requests
    test_requests = [
        RoutingRequest(
            prompt="What is the weather like today?",
            user_preferences=UserPreferences(priority=RoutingPriority.COST)
        ),
        RoutingRequest(
            prompt="Write a Python function to implement binary search",
            user_preferences=UserPreferences(priority=RoutingPriority.QUALITY)
        ),
        RoutingRequest(
            prompt="Solve the equation: 2x^2 + 5x - 3 = 0",
            user_preferences=UserPreferences(priority=RoutingPriority.BALANCED)
        )
    ]
    
    for request in test_requests:
        response = engine.route_request(request)
        print(f"\nPrompt: {request.prompt}")
        print(f"Selected Model: {response.decision.selected_model}")
        print(f"Confidence: {response.decision.confidence:.2f}")
        print(f"Reasoning: {response.decision.reasoning}")
        print(f"Estimated Cost: ${response.decision.estimated_cost:.4f}")
    
    # Print statistics
    stats = engine.get_routing_statistics()
    print(f"\nRouting Statistics: {stats}")

