"""
Shared Data Models for LLM Gateway Smart Routing
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RoutingPriority(Enum):
    """Routing priority options"""
    COST = "cost"
    SPEED = "speed"
    QUALITY = "quality"
    BALANCED = "balanced"


class ModelStatus(Enum):
    """Model availability status"""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"


@dataclass
class UserPreferences:
    """User preferences for routing decisions"""
    priority: RoutingPriority = RoutingPriority.BALANCED
    max_cost: Optional[float] = None
    max_response_time: Optional[int] = None  # milliseconds
    preferred_models: List[str] = field(default_factory=list)
    excluded_models: List[str] = field(default_factory=list)
    quality_threshold: float = 0.7


@dataclass
class RoutingRequest:
    """Incoming routing request"""
    prompt: str
    user_preferences: Optional[UserPreferences] = None
    metadata: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingDecision:
    """Routing decision result"""
    selected_model: str
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_time: int  # milliseconds
    alternative_models: List[str] = field(default_factory=list)
    fallback_model: Optional[str] = None


@dataclass
class LLMEndpoint:
    """LLM endpoint configuration"""
    name: str
    deployment_name: str
    cost_per_1k_tokens: float
    avg_response_time: int  # milliseconds
    quality_score: float
    max_tokens: int
    status: ModelStatus = ModelStatus.AVAILABLE
    use_cases: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class RoutingResponse:
    """Complete routing response"""
    request: RoutingRequest
    decision: RoutingDecision
    analysis: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    processing_time: Optional[int] = None  # milliseconds


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    avg_cost: float = 0.0
    avg_quality_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RoutingMetrics:
    """Routing system metrics"""
    total_requests: int = 0
    successful_routes: int = 0
    failed_routes: int = 0
    avg_processing_time: float = 0.0
    cost_savings: float = 0.0
    model_usage: Dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)


# Default model configurations
DEFAULT_MODELS = {
    'gpt-35-turbo': LLMEndpoint(
        name='GPT-3.5 Turbo',
        deployment_name='gpt-35-turbo',
        cost_per_1k_tokens=0.002,
        avg_response_time=800,
        quality_score=0.75,
        max_tokens=4096,
        use_cases=['simple_qa', 'chat', 'summarization'],
        capabilities=['text_generation', 'chat', 'summarization']
    ),
    'gpt-4': LLMEndpoint(
        name='GPT-4',
        deployment_name='gpt-4',
        cost_per_1k_tokens=0.03,
        avg_response_time=2000,
        quality_score=0.95,
        max_tokens=8192,
        use_cases=['complex_reasoning', 'code_generation', 'analysis'],
        capabilities=['text_generation', 'code_generation', 'complex_reasoning', 'analysis']
    ),
    'gpt-4-turbo': LLMEndpoint(
        name='GPT-4 Turbo',
        deployment_name='gpt-4-turbo',
        cost_per_1k_tokens=0.01,
        avg_response_time=1500,
        quality_score=0.90,
        max_tokens=128000,
        use_cases=['balanced', 'long_context', 'general'],
        capabilities=['text_generation', 'long_context', 'balanced_performance']
    )
}


def create_user_preferences_from_dict(prefs_dict: Dict[str, Any]) -> UserPreferences:
    """Create UserPreferences from dictionary"""
    return UserPreferences(
        priority=RoutingPriority(prefs_dict.get('priority', 'balanced')),
        max_cost=prefs_dict.get('max_cost'),
        max_response_time=prefs_dict.get('max_response_time'),
        preferred_models=prefs_dict.get('preferred_models', []),
        excluded_models=prefs_dict.get('excluded_models', []),
        quality_threshold=prefs_dict.get('quality_threshold', 0.7)
    )


def routing_request_to_dict(request: RoutingRequest) -> Dict[str, Any]:
    """Convert RoutingRequest to dictionary for serialization"""
    return {
        'prompt': request.prompt,
        'user_preferences': {
            'priority': request.user_preferences.priority.value if request.user_preferences else 'balanced',
            'max_cost': request.user_preferences.max_cost if request.user_preferences else None,
            'max_response_time': request.user_preferences.max_response_time if request.user_preferences else None,
            'preferred_models': request.user_preferences.preferred_models if request.user_preferences else [],
            'excluded_models': request.user_preferences.excluded_models if request.user_preferences else [],
            'quality_threshold': request.user_preferences.quality_threshold if request.user_preferences else 0.7
        } if request.user_preferences else None,
        'metadata': request.metadata,
        'session_id': request.session_id,
        'user_id': request.user_id,
        'timestamp': request.timestamp.isoformat()
    }


def routing_response_to_dict(response: RoutingResponse) -> Dict[str, Any]:
    """Convert RoutingResponse to dictionary for serialization"""
    return {
        'request': routing_request_to_dict(response.request),
        'decision': {
            'selected_model': response.decision.selected_model,
            'confidence': response.decision.confidence,
            'reasoning': response.decision.reasoning,
            'estimated_cost': response.decision.estimated_cost,
            'estimated_time': response.decision.estimated_time,
            'alternative_models': response.decision.alternative_models,
            'fallback_model': response.decision.fallback_model
        },
        'analysis': response.analysis,
        'timestamp': response.timestamp.isoformat(),
        'processing_time': response.processing_time
    }

