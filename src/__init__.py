"""
LLM Gateway Smart Routing Package
"""

from .models import (DEFAULT_MODELS, LLMEndpoint, ModelStatus, RoutingDecision,
                     RoutingPriority, RoutingRequest, RoutingResponse,
                     UserPreferences)
from .prompt_analyzer import (PromptAnalysis, PromptAnalyzer, PromptComplexity,
                              PromptDomain)
from .routing_engine import RoutingEngine

__version__ = "1.0.0"
__author__ = "LLM Gateway Team"

__all__ = [
    'PromptAnalyzer',
    'PromptAnalysis', 
    'PromptComplexity',
    'PromptDomain',
    'RoutingEngine',
    'RoutingRequest',
    'RoutingDecision',
    'RoutingResponse',
    'UserPreferences',
    'LLMEndpoint',
    'ModelStatus',
    'RoutingPriority',
    'DEFAULT_MODELS'
]

