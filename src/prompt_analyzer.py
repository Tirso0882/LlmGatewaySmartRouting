"""
Prompt Property Evaluation Module
Analyzes prompt characteristics to make intelligent routing decisions
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


class PromptComplexity(Enum):
    """Enum for prompt complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class PromptDomain(Enum):
    """Enum for prompt domains"""
    GENERAL = "general"
    CODE = "code"
    MATH = "math"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"


@dataclass
class PromptAnalysis:
    """Results of prompt analysis"""
    length_tokens: int
    length_words: int
    complexity: PromptComplexity
    domain: PromptDomain
    has_code: bool
    has_math: bool
    has_questions: bool
    urgency_indicators: List[str]
    estimated_cost: float
    recommended_model: str
    reasoning: str


class PromptAnalyzer:
    """Analyzes prompt properties for smart routing decisions"""
    
    def __init__(self):
        # Code patterns
        self.code_patterns = [
            r'```[\w]*\n',  # Code blocks
            r'function\s+\w+\s*\(',  # Function definitions
            r'def\s+\w+\s*\(',  # Python functions
            r'class\s+\w+',  # Class definitions
            r'import\s+\w+',  # Import statements
            r'const\s+\w+',  # JavaScript constants
            r'let\s+\w+',  # JavaScript let
            r'var\s+\w+',  # JavaScript var
            r'if\s*\(',  # If statements
            r'for\s*\(',  # For loops
            r'while\s*\(',  # While loops
            r'return\s+',  # Return statements
            r'console\.log',  # Console logging
            r'print\s*\(',  # Print statements
            r'write\s+a\s+\w+\s+function',  # Write a function
            r'create\s+a\s+\w+\s+function',  # Create a function
            r'generate\s+a\s+\w+\s+function',  # Generate a function
            r'\bcode\b',  # Code keyword
            r'\bprogramming\b',  # Programming keyword
            r'\balgorithm\b',  # Algorithm keyword
            r'\bapi\b',  # API keyword
            r'\bendpoint\b',  # Endpoint keyword
            r'\bbug\b',  # Bug keyword
            r'\bdebug\b',  # Debug keyword
        ]
        
        # Math patterns
        self.math_patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Basic arithmetic
            r'[a-zA-Z]\s*=\s*\d+',  # Variables with numbers
            r'\bequation\b',  # Equation keywords
            r'\bcalculate\b',  # Calculate keywords
            r'\bsolve\b',  # Solve keywords
            r'\bformula\b',  # Formula keywords
            r'\bpercentage\b',  # Percentage keywords
            r'\bratio\b',  # Ratio keywords
            r'\d+x\b',  # Variables like 2x, 3x
            r'\bx\s*[\+\-\*\/]',  # x +, x -, x *, x /
            r'[\+\-\*\/]\s*x\b',  # + x, - x, * x, / x
            r'\bmath\b',  # Math keyword
            r'\barithmetic\b',  # Arithmetic keyword
            r'\bcalculation\b',  # Calculation keyword
            r'\bcompute\b',  # Compute keyword
            r'\bderivative\b',  # Derivative keyword
            r'\bintegral\b',  # Integral keyword
        ]
        
        # Urgency indicators
        self.urgency_indicators = [
            'urgent', 'asap', 'quick', 'fast', 'immediate',
            'real-time', 'live', 'now', 'instant', 'emergency'
        ]
        
        # Question patterns
        self.question_patterns = [
            r'\?$',  # Ends with question mark
            r'what\s+is',  # What is questions
            r'how\s+to',  # How to questions
            r'why\s+',  # Why questions
            r'when\s+',  # When questions
            r'where\s+',  # Where questions
            r'can\s+you',  # Can you questions
            r'could\s+you',  # Could you questions
        ]
        
        # Model configurations - Latest Azure OpenAI Models
        self.model_configs = {
            'o3': {
                'cost_per_1k_tokens': 0.060,  # Premium pricing for highest accuracy
                'avg_response_time_ms': 5000,  # Slower for advanced reasoning
                'max_tokens': 100000,
                'use_cases': ['complex_reasoning', 'advanced_analysis', 'stem_problems', 'accuracy_priority'],
                'complexity_threshold': PromptComplexity.COMPLEX,
                'specialty': 'Advanced reasoning and logic, excels at complex STEM tasks'
            },
            'gpt-4o-mini': {
                'cost_per_1k_tokens': 0.0001,  # Most cost-effective for speed
                'avg_response_time_ms': 400,    # Fastest response time
                'max_tokens': 16384,
                'use_cases': ['simple_qa', 'chat', 'high_volume', 'speed_priority'],
                'complexity_threshold': PromptComplexity.SIMPLE,
                'specialty': 'Optimized for speed and cost-efficiency for high-volume tasks'
            },
            'o4-mini': {
                'cost_per_1k_tokens': 0.005,  # Balanced cost-performance
                'avg_response_time_ms': 2000,  # Moderate response time
                'max_tokens': 16384,
                'use_cases': ['balanced_tasks', 'cost_efficient', 'general_purpose', 'cost_priority'],
                'complexity_threshold': PromptComplexity.MODERATE,
                'specialty': 'Strong balance between performance and cost for many use cases'
            }
        }

    def analyze_prompt(self, prompt: str, user_preferences: Optional[Dict] = None) -> PromptAnalysis:
        """
        Analyze a prompt and return detailed analysis for routing decisions
        
        Args:
            prompt: The input prompt to analyze
            user_preferences: Optional user preferences (priority, max_cost, etc.)
            
        Returns:
            PromptAnalysis object with all analysis results
        """
        if user_preferences is None:
            user_preferences = {}
            
        # Basic length analysis
        length_tokens = self._estimate_tokens(prompt)
        length_words = len(prompt.split())
        
        # Complexity analysis
        complexity = self._analyze_complexity(prompt, length_tokens)
        
        # Domain analysis
        domain = self._analyze_domain(prompt)
        
        # Feature detection
        has_code = self._detect_code(prompt)
        has_math = self._detect_math(prompt)
        has_questions = self._detect_questions(prompt)
        urgency_indicators = self._detect_urgency(prompt)
        
        # Model selection
        recommended_model, reasoning = self._select_model(
            prompt, complexity, domain, has_code, has_math, 
            length_tokens, user_preferences
        )
        
        # Cost estimation
        estimated_cost = self._estimate_cost(length_tokens, recommended_model)
        
        return PromptAnalysis(
            length_tokens=length_tokens,
            length_words=length_words,
            complexity=complexity,
            domain=domain,
            has_code=has_code,
            has_math=has_math,
            has_questions=has_questions,
            urgency_indicators=urgency_indicators,
            estimated_cost=estimated_cost,
            recommended_model=recommended_model,
            reasoning=reasoning
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 characters)"""
        return len(text) // 4

    def _analyze_complexity(self, prompt: str, token_count: int) -> PromptComplexity:
        """Analyze prompt complexity based on various factors"""
        complexity_score = 0
        
        # Length factor
        if token_count < 50:
            complexity_score += 1
        elif token_count < 200:
            complexity_score += 2
        else:
            complexity_score += 3
            
        # Sentence complexity
        sentences = re.split(r'[.!?]+', prompt)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if avg_sentence_length > 20:
            complexity_score += 2
        elif avg_sentence_length > 15:
            complexity_score += 1
            
        # Vocabulary complexity
        complex_words = ['algorithm', 'implementation', 'optimization', 'architecture', 
                        'framework', 'methodology', 'analysis', 'synthesis', 'hypothesis',
                        'performance', 'implications', 'complexity', 'recommendations',
                        'binary', 'search', 'tree', 'hash', 'table', 'dictionary',
                        'structure', 'time', 'space', 'use cases', 'data structure',
                        'architectural', 'patterns', 'microservices', 'design', 'algorithms',
                        'machine learning', 'summary', 'explain']
        complexity_score += sum(1 for word in complex_words if word.lower() in prompt.lower())
        
        if self._detect_code(prompt):
            complexity_score += 2
            
        if self._detect_math(prompt):
            complexity_score += 2
            
        if complexity_score <= 2:
            return PromptComplexity.SIMPLE
        elif complexity_score <= 5:
            return PromptComplexity.MODERATE
        else:
            return PromptComplexity.COMPLEX

    def _analyze_domain(self, prompt: str) -> PromptDomain:
        """Analyze the domain of the prompt"""
        prompt_lower = prompt.lower()
        
        # Code domain
        if self._detect_code(prompt):
            return PromptDomain.CODE
            
        # Math domain
        if self._detect_math(prompt):
            return PromptDomain.MATH
            
        # Creative domain
        creative_keywords = ['write', 'create', 'story', 'poem', 'creative', 'imagine', 'design']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return PromptDomain.CREATIVE
            
        # Analysis domain
        analysis_keywords = ['analyze', 'compare', 'evaluate', 'assess', 'review', 'examine', 'analysis', 'comprehensive']
        if any(keyword in prompt_lower for keyword in analysis_keywords):
            return PromptDomain.ANALYSIS
            
        # Translation domain
        translation_keywords = ['translate', 'convert', 'interpret', 'meaning']
        if any(keyword in prompt_lower for keyword in translation_keywords):
            return PromptDomain.TRANSLATION
            
        return PromptDomain.GENERAL

    def _detect_code(self, prompt: str) -> bool:
        """Detect if prompt contains code"""
        for pattern in self.code_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def _detect_math(self, prompt: str) -> bool:
        """Detect if prompt contains mathematical content"""
        for pattern in self.math_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def _detect_questions(self, prompt: str) -> bool:
        """Detect if prompt contains questions"""
        for pattern in self.question_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return True
        return False

    def _detect_urgency(self, prompt: str) -> List[str]:
        """Detect urgency indicators in the prompt"""
        found_indicators = []
        prompt_lower = prompt.lower()
        
        for indicator in self.urgency_indicators:
            if indicator in prompt_lower:
                found_indicators.append(indicator)
                
        return found_indicators

    def _select_model(self, prompt: str, complexity: PromptComplexity, domain: PromptDomain,
                     has_code: bool, has_math: bool, token_count: int, 
                     user_preferences: Dict) -> Tuple[str, str]:
        """Select the best model based on analysis and user preferences"""
        
        # Check user preferences first
        priority = user_preferences.get('priority', 'balanced')
        max_cost = user_preferences.get('max_cost', float('inf'))
        
        # Accuracy-optimized routing (prioritize o3 for complex reasoning)
        if priority == 'accuracy' or priority == 'quality':
            if complexity == PromptComplexity.COMPLEX or domain in [PromptDomain.MATH, PromptDomain.ANALYSIS]:
                return 'o3', 'Accuracy-optimized: Advanced reasoning model for complex STEM tasks and deep analysis'
            elif has_math or 'solve' in prompt.lower() or 'analyze' in prompt.lower() or 'analysis' in prompt.lower():
                return 'o3', 'Accuracy-optimized: Advanced reasoning for mathematical and analytical tasks'
            else:
                return 'o4-mini', 'Accuracy-optimized: Balanced performance for moderate complexity tasks'
        
        # Speed-optimized routing (prioritize gpt-4o-mini)
        if priority == 'speed':
            if complexity == PromptComplexity.SIMPLE or 'urgent' in prompt.lower() or token_count < 20:
                return 'gpt-4o-mini', 'Speed-optimized: Fastest response for urgent/simple queries'
            else:
                return 'o4-mini', 'Speed-optimized: Balanced speed for moderate complexity'
        
        # Cost-optimized routing (prioritize o4-mini for balance, gpt-4o-mini for simple)
        if priority == 'cost':
            if complexity == PromptComplexity.SIMPLE and not has_code and not has_math:
                return 'gpt-4o-mini', 'Cost-optimized: Most cost-effective for simple queries'
            else:
                return 'o4-mini', 'Cost-optimized: Strong balance between performance and cost'
        
        # Default balanced routing
        if complexity == PromptComplexity.SIMPLE and not has_code and not has_math:
            return 'gpt-4o-mini', 'Balanced: Simple query, speed and cost efficient'
        elif complexity == PromptComplexity.COMPLEX or domain in [PromptDomain.MATH, PromptDomain.ANALYSIS] or has_math:
            return 'o3', 'Balanced: Complex reasoning task, using advanced reasoning model'
        else:
            return 'o4-mini', 'Balanced: Moderate complexity, optimal performance-cost balance'

    def _estimate_cost(self, token_count: int, model_name: str) -> float:
        """Estimate the cost for the given token count and model"""
        if model_name not in self.model_configs:
            return 0.0
            
        config = self.model_configs[model_name]
        return (token_count / 1000) * config['cost_per_1k_tokens']

    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model"""
        return self.model_configs.get(model_name, {})

    def list_available_models(self) -> List[str]:
        """List all available models"""
        return list(self.model_configs.keys())


if __name__ == "__main__":
    analyzer = PromptAnalyzer()
    
    # Test cases
    test_prompts = [
        "What is the weather like today?",
        "Write a Python function to sort a list of numbers",
        "Solve the equation: 2x + 5 = 15",
        "Analyze the performance implications of using a binary search tree vs a hash table",
        "Create a creative story about a robot learning to paint"
    ]
    
    for prompt in test_prompts:
        analysis = analyzer.analyze_prompt(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Analysis: {analysis}")
        print(f"Recommended Model: {analysis.recommended_model}")
        print(f"Reasoning: {analysis.reasoning}")
        print(f"Estimated Cost: ${analysis.estimated_cost:.4f}")

