"""
Unit tests for Prompt Property Evaluation
"""

import os
import sys
import unittest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import RoutingPriority, RoutingRequest, UserPreferences
from prompt_analyzer import PromptAnalyzer, PromptComplexity, PromptDomain


class TestPromptAnalyzer(unittest.TestCase):
    """Test cases for PromptAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = PromptAnalyzer()
    
    def test_simple_prompt_analysis(self):
        """Test analysis of a simple prompt"""
        prompt = "What is the weather like today?"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertEqual(analysis.complexity, PromptComplexity.SIMPLE)
        self.assertEqual(analysis.domain, PromptDomain.GENERAL)
        self.assertTrue(analysis.has_questions)
        self.assertFalse(analysis.has_code)
        self.assertFalse(analysis.has_math)
        self.assertEqual(analysis.recommended_model, 'gpt-4o-mini')
    
    def test_code_prompt_analysis(self):
        """Test analysis of a code-related prompt"""
        prompt = "Write a Python function to sort a list of numbers"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertIn(analysis.complexity, [PromptComplexity.MODERATE, PromptComplexity.COMPLEX])
        self.assertEqual(analysis.domain, PromptDomain.CODE)
        self.assertTrue(analysis.has_code)
        self.assertIn(analysis.recommended_model, ['o4-mini', 'o3'])
    
    def test_math_prompt_analysis(self):
        """Test analysis of a math-related prompt"""
        prompt = "Solve the equation: 2x + 5 = 15"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertIn(analysis.complexity, [PromptComplexity.MODERATE, PromptComplexity.COMPLEX])
        self.assertEqual(analysis.domain, PromptDomain.MATH)
        self.assertTrue(analysis.has_math)
        self.assertEqual(analysis.recommended_model, 'o3')  # Should route to o3 for math tasks
    
    def test_complex_prompt_analysis(self):
        """Test analysis of a complex prompt"""
        prompt = "Analyze the performance implications of using a binary search tree vs a hash table for implementing a dictionary data structure, considering both time and space complexity, and provide recommendations for different use cases."
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertEqual(analysis.complexity, PromptComplexity.COMPLEX)
        self.assertEqual(analysis.domain, PromptDomain.ANALYSIS)
        self.assertEqual(analysis.recommended_model, 'o3')  # Should route to o3 for complex analysis
    
    def test_cost_optimized_routing(self):
        """Test cost-optimized routing"""
        prompt = "What is the capital of France?"
        user_prefs = {'priority': 'cost'}
        analysis = self.analyzer.analyze_prompt(prompt, user_prefs)
        
        self.assertEqual(analysis.recommended_model, 'gpt-4o-mini')
        self.assertIn('Cost-optimized', analysis.reasoning)
    
    def test_quality_optimized_routing(self):
        """Test quality-optimized routing"""
        prompt = "Write a comprehensive analysis of machine learning algorithms"
        user_prefs = {'priority': 'quality'}
        analysis = self.analyzer.analyze_prompt(prompt, user_prefs)
        
        self.assertEqual(analysis.recommended_model, 'o3')  # Should route to o3 for quality/accuracy
        self.assertIn('Accuracy-optimized', analysis.reasoning)
    
    def test_speed_optimized_routing(self):
        """Test speed-optimized routing"""
        prompt = "What is 2+2?"
        user_prefs = {'priority': 'speed'}
        analysis = self.analyzer.analyze_prompt(prompt, user_prefs)
        
        self.assertEqual(analysis.recommended_model, 'gpt-4o-mini')
        self.assertIn('Speed-optimized', analysis.reasoning)
    
    def test_urgency_detection(self):
        """Test urgency indicator detection"""
        prompt = "I need this urgent analysis asap for my meeting"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertIn('urgent', analysis.urgency_indicators)
        self.assertIn('asap', analysis.urgency_indicators)
    
    def test_token_estimation(self):
        """Test token estimation"""
        prompt = "This is a test prompt with multiple words to estimate tokens"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        # Should have reasonable token count
        self.assertGreater(analysis.length_tokens, 0)
        self.assertLess(analysis.length_tokens, len(prompt))
    
    def test_model_configurations(self):
        """Test model configuration access"""
        models = self.analyzer.list_available_models()
        expected_models = ['o3', 'gpt-4o-mini', 'o4-mini']
        
        for model in expected_models:
            self.assertIn(model, models)
        
        # Test model info
        o3_info = self.analyzer.get_model_info('o3')
        self.assertIn('cost_per_1k_tokens', o3_info)
        self.assertEqual(o3_info['cost_per_1k_tokens'], 0.060)
        
        # Test speed model
        speed_info = self.analyzer.get_model_info('gpt-4o-mini')
        self.assertIn('specialty', speed_info)
        self.assertIn('speed', speed_info['specialty'].lower())
    
    def test_cost_estimation(self):
        """Test cost estimation"""
        prompt = "Test prompt"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertGreaterEqual(analysis.estimated_cost, 0)
        self.assertLess(analysis.estimated_cost, 0.01)  # Should be very small for short prompt
    
    def test_accuracy_priority_routing(self):
        """Test accuracy priority routing to o3"""
        prompt = "Solve this complex physics problem: Calculate the trajectory of a projectile with air resistance"
        user_prefs = {'priority': 'accuracy'}
        analysis = self.analyzer.analyze_prompt(prompt, user_prefs)
        
        self.assertEqual(analysis.recommended_model, 'o3')
        self.assertIn('Accuracy-optimized', analysis.reasoning)
        self.assertTrue(analysis.has_math)
    
    def test_stem_task_routing(self):
        """Test STEM tasks route to o3"""
        prompt = "Derive the formula for calculating quantum entanglement entropy"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertEqual(analysis.recommended_model, 'o3')
        self.assertEqual(analysis.domain, PromptDomain.MATH)
    
    def test_cost_priority_complex_task(self):
        """Test cost priority for complex tasks routes to o4-mini"""
        prompt = "Explain the architectural patterns used in microservices design"
        user_prefs = {'priority': 'cost'}
        analysis = self.analyzer.analyze_prompt(prompt, user_prefs)
        
        self.assertEqual(analysis.recommended_model, 'o4-mini')
        self.assertIn('Cost-optimized', analysis.reasoning)
    
    def test_balanced_moderate_complexity(self):
        """Test balanced routing for moderate complexity"""
        prompt = "Create a summary of machine learning algorithms for beginners"
        analysis = self.analyzer.analyze_prompt(prompt)
        
        self.assertEqual(analysis.recommended_model, 'o4-mini')
        self.assertIn('Balanced', analysis.reasoning)


# class TestRoutingEngine(unittest.TestCase):
#     """Test cases for RoutingEngine"""
#     
#     def setUp(self):
#         """Set up test fixtures"""
#         from routing_engine import RoutingEngine
#         self.engine = RoutingEngine()
#     
#     def test_simple_routing(self):
#         """Test simple routing request"""
#         request = RoutingRequest(
#             prompt="What is the weather like today?",
#             user_preferences=UserPreferences(priority=RoutingPriority.COST)
#         )
#         
#         response = self.engine.route_request(request)
#         
#         self.assertIsNotNone(response)
#         self.assertEqual(response.request.prompt, request.prompt)
#         self.assertIsNotNone(response.decision.selected_model)
#         self.assertGreater(response.decision.confidence, 0)
#         self.assertIsNotNone(response.decision.reasoning)
#     
#     def test_complex_routing(self):
#         """Test complex routing request"""
#         request = RoutingRequest(
#             prompt="Write a Python function to implement a binary search tree with insertion, deletion, and search operations",
#             user_preferences=UserPreferences(priority=RoutingPriority.QUALITY)
#         )
#         
#         response = self.engine.route_request(request)
#         
#         self.assertIsNotNone(response)
#         self.assertIn(response.decision.selected_model, ['gpt-4', 'gpt-4-turbo'])
#         self.assertGreater(response.decision.confidence, 0.7)
#     
#     def test_routing_statistics(self):
#         """Test routing statistics"""
#         # Make a few routing requests
#         requests = [
#             RoutingRequest(prompt="Simple question?"),
#             RoutingRequest(prompt="Complex analysis needed"),
#             RoutingRequest(prompt="Code generation required")
#         ]
#         
#         for req in requests:
#             self.engine.route_request(req)
#         
#         stats = self.engine.get_routing_statistics()
#         
#         self.assertEqual(stats['total_requests'], 3)
#         self.assertGreater(stats['success_rate'], 0)
#         self.assertIn('model_usage', stats)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

