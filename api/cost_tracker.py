"""
Cost Tracker for LLM Gateway
Provides cost tracking for token usage across all models
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ModelPricing:
    """Model pricing information per 1M tokens"""
    model_name: str
    input_price_per_1m: float
    output_price_per_1m: float
    cached_input_price_per_1m: Optional[float] = None
    batch_input_price_per_1m: Optional[float] = None
    batch_output_price_per_1m: Optional[float] = None
    provider: str = "Azure OpenAI"
    last_updated: str = "2025-01-27"
    region: str = "Global"


class CostTracker:
    """Tracks token usage and calculates costs in real-time"""
    
    def __init__(self):
        self.model_pricing = self._initialize_pricing()
        self.session_costs = {}  # Track costs per session
        self.total_costs = {
            'total_tokens_input': 0,
            'total_tokens_output': 0,
            'total_cost_usd': 0.0,
            'total_requests': 0
        }
        self.start_time = time.time()
    
    def _initialize_pricing(self) -> Dict[str, ModelPricing]:
        """Initialize current model pricing based on Azure OpenAI rates"""
        return {
            'o3': ModelPricing(
                model_name='o3 2025-04-16 Global',
                input_price_per_1m=2.00,
                output_price_per_1m=8.00,
                cached_input_price_per_1m=0.50,
                batch_input_price_per_1m=1.00,
                batch_output_price_per_1m=4.00,
                region='Global'
            ),
            'o4-mini': ModelPricing(
                model_name='o4-mini 2025-04-16 Global',
                input_price_per_1m=1.10,
                output_price_per_1m=4.40,
                cached_input_price_per_1m=0.28,
                batch_input_price_per_1m=0.55,
                batch_output_price_per_1m=2.20,
                region='Global'
            ),
            'gpt-4o-mini': ModelPricing(
                model_name='GPT-4o-Mini-Realtime-Preview-2024-12-17-Global',
                input_price_per_1m=0.60,
                output_price_per_1m=2.40,
                cached_input_price_per_1m=0.30,
                region='Global'
            )
        }
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for a given text (rough approximation)"""
        # Rough estimation: 1 token ≈ 4 characters for English text
        # This is a simplified approach - in production, use proper tokenization
        return max(1, len(text) // 4)
    
    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int, 
                      use_cached_input: bool = False, use_batch_api: bool = False) -> Dict[str, float]:
        """Calculate cost for a given token usage"""
        if model_name not in self.model_pricing:
            return {
                'input_cost': 0.0,
                'output_cost': 0.0,
                'total_cost': 0.0,
                'cost_per_1k_tokens': 0.0
            }
        
        pricing = self.model_pricing[model_name]
        
        # Determine input price
        if use_batch_api and pricing.batch_input_price_per_1m:
            input_price = pricing.batch_input_price_per_1m
        elif use_cached_input and pricing.cached_input_price_per_1m:
            input_price = pricing.cached_input_price_per_1m
        else:
            input_price = pricing.input_price_per_1m
        
        # Determine output price
        if use_batch_api and pricing.batch_output_price_per_1m:
            output_price = pricing.batch_output_price_per_1m
        else:
            output_price = pricing.output_price_per_1m
        
        # Calculate costs (convert from per 1M to per token)
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost
        
        # Calculate cost per 1K tokens for reference
        total_tokens = input_tokens + output_tokens
        cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0
        
        return {
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'total_cost': round(total_cost, 6),
            'cost_per_1k_tokens': round(cost_per_1k, 6)
        }
    
    def track_request_cost(self, model_name: str, prompt: str, response: str, 
                          session_id: Optional[str] = None, use_cached_input: bool = False,
                          use_batch_api: bool = False) -> Dict[str, any]:
        """Track cost for a single request"""
        input_tokens = self.estimate_token_count(prompt)
        output_tokens = self.estimate_token_count(response)
        
        cost_breakdown = self.calculate_cost(
            model_name, input_tokens, output_tokens, 
            use_cached_input, use_batch_api
        )
        
        # Update total costs
        self.total_costs['total_tokens_input'] += input_tokens
        self.total_costs['total_tokens_output'] += output_tokens
        self.total_costs['total_cost_usd'] += cost_breakdown['total_cost']
        self.total_costs['total_requests'] += 1
        
        # Track session costs if session_id provided
        if session_id:
            if session_id not in self.session_costs:
                self.session_costs[session_id] = {
                    'total_cost': 0.0,
                    'total_tokens_input': 0,
                    'total_tokens_output': 0,
                    'requests': 0,
                    'start_time': time.time()
                }
            
            self.session_costs[session_id]['total_cost'] += cost_breakdown['total_cost']
            self.session_costs[session_id]['total_tokens_input'] += input_tokens
            self.session_costs[session_id]['total_tokens_output'] += output_tokens
            self.session_costs[session_id]['requests'] += 1
        
        return {
            'model_name': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_breakdown': cost_breakdown,
            'timestamp': datetime.now().isoformat(),
            'session_id': session_id
        }
    
    def get_cost_summary(self) -> Dict[str, any]:
        """Get comprehensive cost summary"""
        uptime_hours = (time.time() - self.start_time) / 3600
        
        return {
            'total_usage': {
                'total_tokens_input': self.total_costs['total_tokens_input'],
                'total_tokens_output': self.total_costs['total_tokens_output'],
                'total_tokens': self.total_costs['total_tokens_input'] + self.total_costs['total_tokens_output'],
                'total_cost_usd': round(self.total_costs['total_cost_usd'], 6),
                'total_requests': self.total_costs['total_requests']
            },
            'cost_per_1k_tokens': round(
                (self.total_costs['total_cost_usd'] / 
                 (self.total_costs['total_tokens_input'] + self.total_costs['total_tokens_output']) * 1000)
                if (self.total_costs['total_tokens_input'] + self.total_costs['total_tokens_output']) > 0 else 0, 6
            ),
            'uptime_hours': round(uptime_hours, 2),
            'cost_per_hour': round(self.total_costs['total_cost_usd'] / uptime_hours, 6) if uptime_hours > 0 else 0,
            'session_summary': {
                session_id: {
                    'total_cost': round(data['total_cost'], 6),
                    'total_tokens': data['total_tokens_input'] + data['total_tokens_output'],
                    'requests': data['requests'],
                    'duration_hours': round((time.time() - data['start_time']) / 3600, 2)
                }
                for session_id, data in self.session_costs.items()
            }
        }
    
    def get_model_pricing_table(self) -> List[Dict[str, any]]:
        """Get current model pricing table for UI display"""
        pricing_table = []
        
        for model_key, pricing in self.model_pricing.items():
            pricing_table.append({
                'model_key': model_key,
                'model_name': pricing.model_name,
                'input_price_per_1k': round(pricing.input_price_per_1m / 1000, 4),
                'output_price_per_1k': round(pricing.output_price_per_1m / 1000, 4),
                'cached_input_price_per_1k': round(pricing.cached_input_price_per_1m / 1000, 4) if pricing.cached_input_price_per_1m else None,
                'batch_input_price_per_1k': round(pricing.batch_input_price_per_1m / 1000, 4) if pricing.batch_input_price_per_1m else None,
                'batch_output_price_per_1k': round(pricing.batch_output_price_per_1m / 1000, 4) if pricing.batch_output_price_per_1m else None,
                'provider': pricing.provider,
                'region': pricing.region,
                'last_updated': pricing.last_updated
            })
        
        return pricing_table
    
    def reset_session_costs(self, session_id: str):
        """Reset costs for a specific session"""
        if session_id in self.session_costs:
            del self.session_costs[session_id]
    
    def reset_all_costs(self):
        """Reset all cost tracking"""
        self.session_costs.clear()
        self.total_costs = {
            'total_tokens_input': 0,
            'total_tokens_output': 0,
            'total_cost_usd': 0.0,
            'total_requests': 0
        }
        self.start_time = time.time()


cost_tracker = CostTracker()
