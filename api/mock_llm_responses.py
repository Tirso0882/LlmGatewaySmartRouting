"""
Mock LLM Response Generator
Simulates realistic responses from different LLM models for demonstration purposes
"""

import random
import time
from typing import Dict, List, Tuple


class MockLLMResponseGenerator:
    """Generates realistic mock responses for different LLM models"""
    
    def __init__(self):
        self.model_personalities = {
            'o3': {
                'style': 'detailed_analytical',
                'response_time_range': (4000, 6000),
                'max_tokens': 1000,
                'tone': 'professional_technical',
                'examples': [
                    "Based on my analysis of the provided information, I can offer a comprehensive assessment...",
                    "Let me break this down systematically. First, we need to consider the underlying principles...",
                    "This is an interesting problem that requires careful consideration of multiple factors..."
                ]
            },
            'gpt-4o-mini': {
                'style': 'concise_practical',
                'response_time_range': (300, 500),
                'max_tokens': 500,
                'tone': 'friendly_direct',
                'examples': [
                    "Here's a quick answer to your question...",
                    "The solution is straightforward:",
                    "Based on what you've asked, here's what you need to know..."
                ]
            },
            'o4-mini': {
                'style': 'balanced_informative',
                'response_time_range': (1200, 1800),
                'max_tokens': 800,
                'tone': 'helpful_educational',
                'examples': [
                    "I'll help you understand this topic. Here's what you should know...",
                    "Let me explain this in a clear and helpful way...",
                    "Here's a balanced perspective on your question..."
                ]
            }
        }
        
        self.response_templates = {
            'weather': {
                'o3': "Based on current meteorological data and atmospheric conditions, I can provide a comprehensive weather analysis. The temperature is currently {temp}°F with {condition} conditions. Wind speeds are {wind} mph from the {direction}. This weather pattern is typical for this time of year, influenced by {pressure_system} pressure systems.",
                'gpt-4o-mini': "Current weather: {temp}°F, {condition}. Wind: {wind} mph {direction}. Perfect for {activity}!",
                'o4-mini': "The weather today is {temp}°F with {condition} conditions. Wind is {wind} mph from the {direction}. This should be good for {activity}."
            },
            'math': {
                'o3': "Let me solve this mathematical problem step by step. First, I'll analyze the equation structure: {equation}. Using algebraic principles, I can derive that x = {solution}. This solution is verified by substituting back into the original equation, confirming mathematical consistency.",
                'gpt-4o-mini': "Quick math: {equation} = {solution}",
                'o4-mini': "Here's the solution: {equation} = {solution}. I used {method} to solve this."
            },
            'code': {
                'o3': "I'll provide a comprehensive code solution with detailed explanations. Here's the implementation:\n\n```python\n{code}\n```\n\nThis solution addresses {complexity} aspects of the problem, including {features}. The time complexity is O({complexity}) and space complexity is O({space}).",
                'gpt-4o-mini': "Here's the code:\n```python\n{code}\n```",
                'o4-mini': "Here's a solution:\n```python\n{code}\n```\nThis handles {features} efficiently."
            },
            'general': {
                'o3': "I'll provide a thorough analysis of this topic. {topic} involves several key considerations: {points}. The most important aspect is {main_point}, which influences {implications}. This understanding is crucial for {applications}.",
                'gpt-4o-mini': "Quick answer: {answer}",
                'o4-mini': "Here's what you need to know about {topic}: {answer}. This is important because {reason}."
            }
        }
    
    def generate_response(self, model_name: str, prompt: str) -> Tuple[str, float]:
        """Generate a realistic mock response for the given model and prompt"""
        
        # Simulate response time based on model characteristics
        personality = self.model_personalities[model_name]
        response_time = random.uniform(*personality['response_time_range'])
        
        # Determine response type based on prompt content
        response_type = self._classify_prompt(prompt)
        
        # Generate appropriate response
        if response_type == 'weather':
            response = self._generate_weather_response(model_name)
        elif response_type == 'math':
            response = self._generate_math_response(model_name)
        elif response_type == 'code':
            response = self._generate_code_response(model_name)
        else:
            response = self._generate_general_response(model_name, prompt)
        
        return response, response_time
    
    def _classify_prompt(self, prompt: str) -> str:
        """Classify the type of prompt for appropriate response generation"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'sunny']):
            return 'weather'
        elif any(word in prompt_lower for word in ['solve', 'equation', 'calculate', 'math', '2+2', 'x²']):
            return 'math'
        elif any(word in prompt_lower for word in ['code', 'function', 'python', 'javascript', 'program', 'debug']):
            return 'code'
        else:
            return 'general'
    
    def _generate_weather_response(self, model_name: str) -> str:
        """Generate a weather-related response"""
        template = self.response_templates['weather'][model_name]
        
        weather_data = {
            'temp': random.randint(45, 85),
            'condition': random.choice(['sunny', 'cloudy', 'partly cloudy', 'rainy']),
            'wind': random.randint(5, 25),
            'direction': random.choice(['north', 'south', 'east', 'west', 'northeast', 'northwest']),
            'pressure_system': random.choice(['high', 'low']),
            'activity': random.choice(['outdoor activities', 'staying inside', 'a walk', 'gardening'])
        }
        
        return template.format(**weather_data)
    
    def _generate_math_response(self, model_name: str) -> str:
        """Generate a math-related response"""
        template = self.response_templates['math'][model_name]
        
        # Simple math examples
        math_examples = [
            ('2x + 5 = 13', '4', 'algebraic manipulation'),
            ('x² + 5x - 6 = 0', 'x = 1 or x = -6', 'quadratic formula'),
            ('15 × 7', '105', 'multiplication'),
            ('√144', '12', 'square root calculation')
        ]
        
        equation, solution, method = random.choice(math_examples)
        
        return template.format(equation=equation, solution=solution, method=method)
    
    def _generate_code_response(self, model_name: str) -> str:
        """Generate a code-related response"""
        template = self.response_templates['code'][model_name]
        
        code_examples = [
            ('def sort_list(lst):\n    return sorted(lst)', 'O(n log n)', 'O(n)', 'sorting algorithms'),
            ('def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)', 'O(2^n)', 'O(n)', 'recursive functions'),
            ('def find_max(arr):\n    return max(arr) if arr else None', 'O(n)', 'O(1)', 'array operations')
        ]
        
        code, complexity, space, features = random.choice(code_examples)
        
        return template.format(code=code, complexity=complexity, space=space, features=features)
    
    def _generate_general_response(self, model_name: str, prompt: str) -> str:
        """Generate a general response based on the prompt"""
        template = self.response_templates['general'][model_name]
        
        # Extract topic from prompt
        words = prompt.split()[:3]
        topic = ' '.join(words)
        
        general_responses = {
            'o3': f"I'll provide a comprehensive analysis of '{topic}'. This topic involves several key considerations: historical context, current applications, and future implications. The most important aspect is understanding the underlying principles, which influences decision-making processes. This understanding is crucial for strategic planning and implementation.",
            'gpt-4o-mini': f"Quick answer about {topic}: This is a straightforward topic with clear applications in everyday use.",
            'o4-mini': f"Here's what you need to know about {topic}: It's an important concept with practical applications. This is important because it helps with understanding broader principles."
        }
        
        return general_responses[model_name]
    
    def get_model_characteristics(self, model_name: str) -> Dict:
        """Get the characteristics of a specific model"""
        return self.model_personalities.get(model_name, {})

# Global instance
mock_llm = MockLLMResponseGenerator()
