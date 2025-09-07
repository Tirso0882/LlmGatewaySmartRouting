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
                'description': 'High-accuracy model for complex reasoning',
                'avg_response_time': 5000,
                'complexity_level': 'High',
                'processing_style': 'Deep analysis with multiple perspectives',
                'specialization': 'Complex reasoning and analysis',
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
                'description': 'Fast and cost-effective for simple tasks',
                'avg_response_time': 400,
                'complexity_level': 'Low',
                'processing_style': 'Quick pattern matching',
                'specialization': 'Simple queries and quick responses',
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
                'description': 'Balanced performance for moderate complexity tasks',
                'avg_response_time': 1500,
                'complexity_level': 'Medium',
                'processing_style': 'Balanced analysis',
                'specialization': 'Moderate complexity tasks',
                'examples': [
                    "I'll help you understand this topic. Here's what you should know...",
                    "Let me explain this in a clear and helpful way...",
                    "Here's a balanced perspective on your question..."
                ]
            }
        }
    
    def generate_response(self, model_name: str, prompt: str) -> Tuple[str, float]:
        """Generate a realistic mock response for the given model and prompt"""
        
        # Simulate response time based on model characteristics
        personality = self.model_personalities[model_name]
        response_time = random.uniform(*personality['response_time_range'])
        
        # Determine response type based on prompt content
        response_type = self._classify_prompt(prompt)
        
        if response_type == 'weather':
            response = self._generate_weather_response(model_name)
        elif response_type == 'math':
            response = self._generate_math_response(model_name)
        elif response_type == 'code':
            response = self._generate_code_response(model_name)
        else:
            response = self._generate_general_response(model_name, prompt)
        
        # Apply beautiful formatting
        response = self._format_response_beautifully(response, model_name, prompt)
        
        return response, response_time
    
    def _format_response_beautifully(self, response: str, model_name: str, prompt: str) -> str:
        """Format the response beautifully with clean, professional styling"""
        

        header = f"""
# 🤖 {model_name.upper()} Response

"""
        

        response_type = self._classify_prompt(prompt)
        
        if response_type == 'code':
            formatted_content = self._format_code_response(response)
        elif response_type == 'math':
            formatted_content = self._format_math_response(response)
        elif response_type == 'weather':
            formatted_content = self._format_weather_response(response)
        else:
            formatted_content = self._format_general_response(response)
        

        footer = f"""

"""
        
        return header + formatted_content + footer
    
    def _format_code_response(self, response: str) -> str:
        """Format code responses with clean syntax highlighting"""
        

        if 'comprehensive' in response.lower():
            code_type = "Comprehensive Code Solution"
            features = "Performance optimization and detailed explanations"
        elif 'simple' in response.lower():
            code_type = "Simple Code Solution"
            features = "Clean and straightforward implementation"
        else:
            code_type = "Balanced Code Solution"
            features = "Efficiency and maintainability"
        
        return f"""
## 💻 {code_type}

{response}

### 📋 Code Quality:
- Efficiency: Optimized for performance
- Readability: Clean, well-documented code
- Maintainability: Follows best practices

### 🚀 Implementation Notes:
- Best Practices: Industry-standard coding patterns
- Error Handling: Robust error management
- Documentation: Clear code comments
"""
    
    def _format_math_response(self, response: str) -> str:
        """Format math responses with clear structure"""

        if 'step by step' in response.lower():
            method_type = "Step-by-Step Solution"
            approach = "Systematic analysis and logical reasoning"
        elif 'algebraic' in response.lower():
            method_type = "Algebraic Solution"
            approach = "Algebraic manipulation and equation solving"
        else:
            method_type = "Mathematical Solution"
            approach = "Mathematical principles and techniques"
        
        return f"""
## 🧮 {method_type}

{response}

### 📐 Solution Approach:
1. Problem Analysis: Understanding the equation structure
2. Solution Method: {approach}
3. Verification: Confirming mathematical consistency

### 💡 Key Mathematical Concepts:
- Equation solving techniques
- Mathematical verification
- Logical reasoning
"""
    
    def _format_weather_response(self, response: str) -> str:
        """Format weather responses with visual elements"""

        temp_match = None
        if '°F' in response:
            temp_parts = response.split('°F')[0].split()
            if temp_parts:
                temp_match = temp_parts[-1] + '°F'
        

        conditions = ['sunny', 'cloudy', 'partly cloudy', 'rainy', 'overcast']
        condition_match = None
        for condition in conditions:
            if condition in response.lower():
                condition_match = condition.title()
                break
        
        return f"""
## 🌤️ Weather Information

{response}

### 📊 Current Conditions:
- Temperature: {temp_match if temp_match else 'Variable'}
- Sky Condition: {condition_match if condition_match else 'Mixed'}
- Wind: Included in detailed response above

### 🌍 Weather Context:
- Seasonal Patterns: Typical for this time of year
- Atmospheric Conditions: Pressure system influences
- Activity Recommendations: Weather-appropriate suggestions
"""
    
    def _format_general_response(self, response: str) -> str:
        """Format general responses with clear structure"""

        if 'comprehensive analysis' in response.lower():
            content_type = "Comprehensive Analysis"
            key_points = "Multiple perspectives and detailed insights"
            context = "Historical, current, and future implications"
        elif 'detailed explanation' in response.lower():
            content_type = "Detailed Explanation"
            key_points = "Step-by-step breakdown and core mechanisms"
            context = "Process understanding and practical application"
        elif 'why' in response.lower():
            content_type = "Causal Analysis"
            key_points = "Underlying reasons and implications"
            context = "Decision-making insights and connections"
        else:
            content_type = "Comprehensive Answer"
            key_points = "Core concepts and practical applications"
            context = "Fundamental principles and real-world relevance"
        
        return f"""
## 📚 {content_type}

{response}

### 🎯 Key Insights:
- Core Concept: {response.split("'")[1] if "'" in response else 'Main topic'}
- Key Benefits: {key_points}
- Practical Value: {context}

### 💡 Why This Matters:
- Understanding: Builds foundational knowledge
- Application: Useful across different domains
- Growth: Enables better decision-making
"""
    
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
        
        temp = random.randint(45, 85)
        condition = random.choice(['sunny', 'cloudy', 'partly cloudy', 'rainy', 'overcast'])
        wind = random.randint(5, 25)
        direction = random.choice(['north', 'south', 'east', 'west', 'northeast', 'northwest'])
        
        if model_name == 'o3':
            return f"Based on current meteorological data and atmospheric conditions, I can provide a comprehensive weather analysis. The temperature is currently {temp}°F with {condition} conditions. Wind speeds are {wind} mph from the {direction}. This weather pattern is typical for this time of year, influenced by prevailing pressure systems. The conditions suggest it's suitable for outdoor activities if you're prepared for the current temperature and wind conditions."
        elif model_name == 'gpt-4o-mini':
            return f"Current weather: {temp}°F, {condition}. Wind: {wind} mph {direction}. Perfect for outdoor activities!"
        else:  # o4-mini
            return f"The weather today is {temp}°F with {condition} conditions. Wind is {wind} mph from the {direction}. This should be good for outdoor activities, though you might want to consider the wind factor."
    
    def _generate_math_response(self, model_name: str) -> str:
        """Generate a math-related response"""
        
        if model_name == 'o3':
            return "Let me solve this mathematical problem step by step. First, I'll analyze the equation structure and identify the mathematical principles involved. Using systematic algebraic manipulation, I can derive the solution through logical reasoning. This solution is verified by substituting back into the original equation, confirming mathematical consistency and accuracy."
        elif model_name == 'gpt-4o-mini':
            return "Here's the solution to your mathematical problem. I used the appropriate mathematical method to arrive at the answer efficiently."
        else:  # o4-mini
            return "Here's the solution: I applied the correct mathematical principles to solve this problem. The method involves understanding the equation structure and using appropriate techniques to find the solution."
    
    def _generate_code_response(self, model_name: str) -> str:
        """Generate a code-related response"""
        
        if model_name == 'o3':
            return "I'll provide a comprehensive code solution with detailed explanations. Here's the implementation:\n\n```python\ndef optimized_solution(data):\n    # Pre-process data for efficiency\n    processed = [x for x in data if x is not None]\n    \n    # Apply optimized algorithm\n    result = sorted(processed, key=lambda x: x.value)\n    \n    return result\n```\n\nThis solution addresses performance aspects through efficient data preprocessing and optimized sorting. The time complexity is O(n log n) and space complexity is O(n), making it suitable for production use."
        elif model_name == 'gpt-4o-mini':
            return "Here's the code solution:\n\n```python\ndef simple_solution(data):\n    return sorted(data)\n```\n\nThis provides a clean, straightforward implementation."
        else:  # o4-mini
            return "Here's a solution:\n\n```python\ndef balanced_solution(data):\n    # Handle edge cases\n    if not data:\n        return []\n    \n    # Sort efficiently\n    return sorted(data)\n```\n\nThis handles functionality efficiently while maintaining code clarity."
    
    def _generate_general_response(self, model_name: str, prompt: str) -> str:
        """Generate a general response based on the prompt"""
        
        # Extract meaningful topic from prompt
        prompt_words = prompt.strip().split()
        if len(prompt_words) >= 3:
            topic = ' '.join(prompt_words[:3])
        else:
            topic = prompt.strip()

        if 'meaning' in prompt.lower() or 'explain' in prompt.lower():
            responses = {
                'o3': f"I'll provide a comprehensive analysis of '{topic}'. This concept involves several key considerations: its historical development, current applications across different fields, and future implications. The most important aspect is understanding the underlying principles and how they connect to broader theoretical frameworks. This understanding is crucial for both academic study and practical application in real-world scenarios.",
                'gpt-4o-mini': f"Quick answer about {topic}: This is a fundamental concept that appears in various contexts. It's straightforward to understand and has clear applications in everyday situations.",
                'o4-mini': f"Here's what you need to know about {topic}: It's an important concept with practical applications across different domains. Understanding this helps with grasping broader principles and applying them effectively."
            }
        elif 'how' in prompt.lower() or 'what' in prompt.lower():
            responses = {
                'o3': f"Let me provide a detailed explanation of '{topic}'. This involves understanding the core mechanisms, the step-by-step process, and the underlying principles that make it work. The key is to break down complex concepts into manageable parts and see how they interconnect to form a complete understanding.",
                'gpt-4o-mini': f"Here's a simple explanation of {topic}: It works through a straightforward process that's easy to follow and apply.",
                'o4-mini': f"Let me explain {topic} in a clear way: It involves understanding the basic steps and how they work together to achieve the desired outcome."
            }
        elif 'why' in prompt.lower():
            responses = {
                'o3': f"Let me analyze why '{topic}' matters. This involves examining the underlying reasons, the causal relationships, and the broader implications. Understanding the 'why' helps us make better decisions and see connections that might not be immediately obvious.",
                'gpt-4o-mini': f"Here's why {topic} is important: It has practical benefits and helps solve real problems.",
                'o4-mini': f"Understanding why {topic} matters helps us see its value and apply it more effectively in different situations."
            }
        else:
            responses = {
                'o3': f"I'll provide a comprehensive analysis of '{topic}'. This topic involves several key considerations: its fundamental nature, current applications in various fields, and future implications. The most important aspect is understanding the underlying principles and how they connect to broader theoretical frameworks. This understanding is crucial for both academic study and practical application in real-world scenarios.",
                'gpt-4o-mini': f"Quick answer about {topic}: This is a straightforward topic with clear applications in everyday use.",
                'o4-mini': f"Here's what you need to know about {topic}: It's an important concept with practical applications. This is important because it helps with understanding broader principles."
            }
        
        return responses[model_name]
    
    def get_model_characteristics(self, model_name: str) -> Dict:
        """Get the characteristics of a specific model"""
        return self.model_personalities.get(model_name, {})

mock_llm = MockLLMResponseGenerator()
