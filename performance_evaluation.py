#!/usr/bin/env python3
"""
Performance Evaluation Script for LLM Gateway
Tests DistilBERT vs Rule-based routing performance without LLM generation time

This script uses the BLOCK_LLM_RESPONSES environment variable to isolate
routing performance from LLM generation time, allowing accurate comparison
of the two routing approaches.

Usage:
    # Block LLM responses for pure routing performance testing
    BLOCK_LLM_RESPONSES=true python performance_evaluation.py
    
    # Enable LLM responses for full end-to-end testing
    BLOCK_LLM_RESPONSES=false python performance_evaluation.py
"""

import json
import os
import statistics
import time
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import requests

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_PROMPTS = [
    "Hello, how are you today?",
    "Can you solve this complex mathematical equation: x² + 5x + 6 = 0?",
    "What's the weather like today?",
    "Write a Python function to sort a list of numbers",
    "Explain the concept of machine learning in simple terms",
    "Calculate the area of a circle with radius 5",
    "What are the benefits of renewable energy?",
    "Debug this JavaScript code: function add(a, b) { return a + b; }",
    "Tell me a joke",
    "How does photosynthesis work?",
    "Create a simple REST API endpoint",
    "What is the capital of France?",
    "Explain quantum computing",
    "Write a SQL query to find all users with age > 25",
    "What are the symptoms of COVID-19?",
    "How do I bake a chocolate cake?",
    "Explain the difference between HTTP and HTTPS",
    "What is the speed of light?",
    "How does a computer work?",
    "Write a regex pattern to match email addresses"
]

class PerformanceEvaluator:
    """Enhanced performance evaluator for LLM Gateway routing"""
    
    def __init__(self, api_base_url: str = API_BASE_URL):
        self.api_base_url = api_base_url
        self.results = {
            'distilbert': [],
            'rule_based': [],
            'metadata': {
                'test_start_time': None,
                'test_end_time': None,
                'llm_responses_blocked': None,
                'total_prompts': len(TEST_PROMPTS),
                'api_base_url': api_base_url
            }
        }
    
    def check_api_status(self) -> Dict:
        """Check if the API is running and get current status"""
        try:
            response = requests.get(f"{self.api_base_url}/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print(f"✅ API is running")
                print(f"📊 LLM Responses Blocked: {status['llm_responses_blocked']}")
                print(f"🤖 Model Loaded: {status['model_loaded']}")
                return status
            else:
                print(f"❌ API returned status code: {response.status_code}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to API: {e}")
            print(f"💡 Make sure the API is running at {self.api_base_url}")
            return {}
    
    def test_routing_performance(self, routing_mode: str, prompt: str) -> Dict:
        """Test routing performance for a single prompt"""
        try:
            start_time = time.time()
            
            payload = {
                "prompt": prompt,
                "routing_mode": routing_mode
            }
            
            response = requests.post(
                f"{self.api_base_url}/route",
                json=payload,
                timeout=30
            )
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'prompt': prompt,
                    'routing_mode': routing_mode,
                    'recommended_model': data['recommended_model'],
                    'confidence': data['confidence'],
                    'inference_time_ms': data['inference_time_ms'],
                    'llm_response_time_ms': data['llm_response_time_ms'],
                    'total_time_ms': total_time,
                    'reasoning': data['reasoning'],
                    'response_length': len(data['llm_response'])
                }
            else:
                return {
                    'success': False,
                    'prompt': prompt,
                    'routing_mode': routing_mode,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'total_time_ms': total_time
                }
                
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'prompt': prompt,
                'routing_mode': routing_mode,
                'error': str(e),
                'total_time_ms': 0
            }
    
    def run_performance_test(self, num_iterations: int = 3) -> Dict:
        """Run comprehensive performance test"""
        print(f"🚀 Starting performance evaluation...")
        print(f"📝 Testing {len(TEST_PROMPTS)} prompts with {num_iterations} iterations each")
        print(f"🔄 Total tests: {len(TEST_PROMPTS) * 2 * num_iterations}")
        
        # Check API status first
        status = self.check_api_status()
        if not status:
            return self.results
        
        self.results['metadata']['llm_responses_blocked'] = status.get('llm_responses_blocked', False)
        self.results['metadata']['test_start_time'] = datetime.now().isoformat()
        
        # Test both routing modes
        for routing_mode in ['distilbert', 'rule_based']:
            print(f"\n🧪 Testing {routing_mode.upper()} routing...")
            
            for iteration in range(num_iterations):
                print(f"  Iteration {iteration + 1}/{num_iterations}")
                
                for i, prompt in enumerate(TEST_PROMPTS):
                    print(f"    Testing prompt {i + 1}/{len(TEST_PROMPTS)}: {prompt[:50]}...")
                    
                    result = self.test_routing_performance(routing_mode, prompt)
                    self.results[routing_mode].append(result)
                    
                    if result['success']:
                        print(f"      ✅ {result['recommended_model']} ({result['confidence']:.2%}) - {result['total_time_ms']:.1f}ms")
                    else:
                        print(f"      ❌ Error: {result.get('error', 'Unknown error')}")
        
        self.results['metadata']['test_end_time'] = datetime.now().isoformat()
        return self.results
    
    def analyze_results(self) -> Dict:
        """Analyze performance results and generate statistics"""
        analysis = {
            'summary': {},
            'detailed_stats': {},
            'comparison': {}
        }
        
        for routing_mode in ['distilbert', 'rule_based']:
            results = [r for r in self.results[routing_mode] if r['success']]
            
            if not results:
                print(f"⚠️ No successful results for {routing_mode}")
                continue
            
            # Calculate statistics
            inference_times = [r['inference_time_ms'] for r in results]
            total_times = [r['total_time_ms'] for r in results]
            confidences = [r['confidence'] for r in results]
            
            stats = {
                'total_tests': len(self.results[routing_mode]),
                'successful_tests': len(results),
                'success_rate': len(results) / len(self.results[routing_mode]) * 100,
                'inference_time': {
                    'mean': statistics.mean(inference_times),
                    'median': statistics.median(inference_times),
                    'std': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                    'min': min(inference_times),
                    'max': max(inference_times)
                },
                'total_time': {
                    'mean': statistics.mean(total_times),
                    'median': statistics.median(total_times),
                    'std': statistics.stdev(total_times) if len(total_times) > 1 else 0,
                    'min': min(total_times),
                    'max': max(total_times)
                },
                'confidence': {
                    'mean': statistics.mean(confidences),
                    'median': statistics.median(confidences),
                    'std': statistics.stdev(confidences) if len(confidences) > 1 else 0,
                    'min': min(confidences),
                    'max': max(confidences)
                }
            }
            
            analysis['detailed_stats'][routing_mode] = stats
            
            # Model selection distribution
            model_counts = {}
            for result in results:
                model = result['recommended_model']
                model_counts[model] = model_counts.get(model, 0) + 1
            
            analysis['detailed_stats'][routing_mode]['model_distribution'] = model_counts
        
        # Comparison analysis
        if 'distilbert' in analysis['detailed_stats'] and 'rule_based' in analysis['detailed_stats']:
            distilbert_stats = analysis['detailed_stats']['distilbert']
            rule_based_stats = analysis['detailed_stats']['rule_based']
            
            analysis['comparison'] = {
                'inference_time_improvement': {
                    'distilbert_vs_rule_based': rule_based_stats['inference_time']['mean'] - distilbert_stats['inference_time']['mean'],
                    'percentage_improvement': ((rule_based_stats['inference_time']['mean'] - distilbert_stats['inference_time']['mean']) / rule_based_stats['inference_time']['mean']) * 100
                },
                'total_time_improvement': {
                    'distilbert_vs_rule_based': rule_based_stats['total_time']['mean'] - distilbert_stats['total_time']['mean'],
                    'percentage_improvement': ((rule_based_stats['total_time']['mean'] - distilbert_stats['total_time']['mean']) / rule_based_stats['total_time']['mean']) * 100
                },
                'confidence_comparison': {
                    'distilbert_mean': distilbert_stats['confidence']['mean'],
                    'rule_based_mean': rule_based_stats['confidence']['mean'],
                    'difference': distilbert_stats['confidence']['mean'] - rule_based_stats['confidence']['mean']
                }
            }
        
        return analysis
    
    def print_summary(self, analysis: Dict):
        """Print performance summary"""
        print("\n" + "="*80)
        print("📊 PERFORMANCE EVALUATION SUMMARY")
        print("="*80)
        
        metadata = self.results['metadata']
        print(f"🕐 Test Duration: {metadata['test_start_time']} to {metadata['test_end_time']}")
        print(f"🚫 LLM Responses Blocked: {metadata['llm_responses_blocked']}")
        print(f"📝 Total Prompts Tested: {metadata['total_prompts']}")
        
        for routing_mode in ['distilbert', 'rule_based']:
            if routing_mode in analysis['detailed_stats']:
                stats = analysis['detailed_stats'][routing_mode]
                print(f"\n🤖 {routing_mode.upper()} ROUTING:")
                print(f"  ✅ Success Rate: {stats['success_rate']:.1f}% ({stats['successful_tests']}/{stats['total_tests']})")
                print(f"  ⚡ Inference Time: {stats['inference_time']['mean']:.2f}ms (avg), {stats['inference_time']['median']:.2f}ms (median)")
                print(f"  🕐 Total Time: {stats['total_time']['mean']:.2f}ms (avg), {stats['total_time']['median']:.2f}ms (median)")
                print(f"  🎯 Confidence: {stats['confidence']['mean']:.2%} (avg), {stats['confidence']['median']:.2%} (median)")
                
                # Model distribution
                print(f"  📊 Model Distribution:")
                for model, count in stats['model_distribution'].items():
                    percentage = (count / stats['successful_tests']) * 100
                    print(f"    - {model}: {count} ({percentage:.1f}%)")
        
        # Comparison
        if analysis['comparison']:
            print(f"\n🆚 COMPARISON (DistilBERT vs Rule-based):")
            comp = analysis['comparison']
            
            if 'inference_time_improvement' in comp:
                improvement = comp['inference_time_improvement']
                if improvement['distilbert_vs_rule_based'] > 0:
                    print(f"  ⚡ DistilBERT is {improvement['distilbert_vs_rule_based']:.2f}ms FASTER for inference")
                    print(f"  📈 Performance improvement: {improvement['percentage_improvement']:.1f}%")
                else:
                    print(f"  ⚡ Rule-based is {abs(improvement['distilbert_vs_rule_based']):.2f}ms FASTER for inference")
                    print(f"  📉 DistilBERT is {abs(improvement['percentage_improvement']):.1f}% slower")
            
            if 'confidence_comparison' in comp:
                conf_comp = comp['confidence_comparison']
                print(f"  🎯 Confidence: DistilBERT {conf_comp['distilbert_mean']:.2%} vs Rule-based {conf_comp['rule_based_mean']:.2%}")
                if conf_comp['difference'] > 0:
                    print(f"  📈 DistilBERT has {conf_comp['difference']:.2%} higher confidence")
                else:
                    print(f"  📉 Rule-based has {abs(conf_comp['difference']):.2%} higher confidence")
    
    def save_results(self, analysis: Dict, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_evaluation_{timestamp}.json"
        
        output_data = {
            'results': self.results,
            'analysis': analysis,
            'metadata': {
                'script_version': '2.0',
                'test_type': 'routing_performance_comparison',
                'llm_responses_blocked': self.results['metadata']['llm_responses_blocked']
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
        return filename
    
    def create_visualizations(self, analysis: Dict, save_plots: bool = True):
        """Create performance visualization plots"""
        if not analysis['detailed_stats']:
            print("⚠️ No data available for visualization")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LLM Gateway Routing Performance Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data
        routing_modes = list(analysis['detailed_stats'].keys())
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # 1. Inference Time Comparison
        ax1 = axes[0, 0]
        inference_means = [analysis['detailed_stats'][mode]['inference_time']['mean'] for mode in routing_modes]
        inference_stds = [analysis['detailed_stats'][mode]['inference_time']['std'] for mode in routing_modes]
        
        bars1 = ax1.bar(routing_modes, inference_means, yerr=inference_stds, 
                       color=colors[:len(routing_modes)], alpha=0.7, capsize=5)
        ax1.set_title('Inference Time Comparison', fontweight='bold')
        ax1.set_ylabel('Time (ms)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, inference_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(inference_stds) * 0.1,
                    f'{mean:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 2. Total Time Comparison
        ax2 = axes[0, 1]
        total_means = [analysis['detailed_stats'][mode]['total_time']['mean'] for mode in routing_modes]
        total_stds = [analysis['detailed_stats'][mode]['total_time']['std'] for mode in routing_modes]
        
        bars2 = ax2.bar(routing_modes, total_means, yerr=total_stds, 
                       color=colors[:len(routing_modes)], alpha=0.7, capsize=5)
        ax2.set_title('Total Response Time Comparison', fontweight='bold')
        ax2.set_ylabel('Time (ms)')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars2, total_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(total_stds) * 0.1,
                    f'{mean:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 3. Confidence Comparison
        ax3 = axes[1, 0]
        confidence_means = [analysis['detailed_stats'][mode]['confidence']['mean'] for mode in routing_modes]
        confidence_stds = [analysis['detailed_stats'][mode]['confidence']['std'] for mode in routing_modes]
        
        bars3 = ax3.bar(routing_modes, confidence_means, yerr=confidence_stds, 
                       color=colors[:len(routing_modes)], alpha=0.7, capsize=5)
        ax3.set_title('Confidence Level Comparison', fontweight='bold')
        ax3.set_ylabel('Confidence')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars3, confidence_means):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(confidence_stds) * 0.1,
                    f'{mean:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Model Distribution
        ax4 = axes[1, 1]
        if len(routing_modes) >= 2:
            # Compare model distributions
            distilbert_dist = analysis['detailed_stats'].get('distilbert', {}).get('model_distribution', {})
            rule_based_dist = analysis['detailed_stats'].get('rule_based', {}).get('model_distribution', {})
            
            models = list(set(list(distilbert_dist.keys()) + list(rule_based_dist.keys())))
            distilbert_counts = [distilbert_dist.get(model, 0) for model in models]
            rule_based_counts = [rule_based_dist.get(model, 0) for model in models]
            
            x = range(len(models))
            width = 0.35
            
            ax4.bar([i - width/2 for i in x], distilbert_counts, width, label='DistilBERT', 
                   color=colors[0], alpha=0.7)
            ax4.bar([i + width/2 for i in x], rule_based_counts, width, label='Rule-based', 
                   color=colors[1], alpha=0.7)
            
            ax4.set_title('Model Selection Distribution', fontweight='bold')
            ax4.set_ylabel('Count')
            ax4.set_xlabel('Models')
            ax4.set_xticks(x)
            ax4.set_xticklabels(models, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor comparison', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Model Selection Distribution', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"performance_comparison_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {plot_filename}")
        
        plt.show()

def main():
    """Main function to run the performance evaluation"""
    print("🚀 LLM Gateway Performance Evaluation")
    print("="*50)
    
    # Check if LLM responses are blocked
    llm_blocked = os.getenv("BLOCK_LLM_RESPONSES", "false").lower() == "true"
    print(f"🚫 LLM Responses Blocked: {llm_blocked}")
    
    if llm_blocked:
        print("✅ Testing pure routing performance (no LLM generation time)")
    else:
        print("⚠️  Testing full end-to-end performance (includes LLM generation time)")
        print("💡 To test routing performance only, set BLOCK_LLM_RESPONSES=true")
    
    evaluator = PerformanceEvaluator()
    results = evaluator.run_performance_test(num_iterations=3)
    analysis = evaluator.analyze_results()
    evaluator.print_summary(analysis)
    results_file = evaluator.save_results(analysis)
    try:
        evaluator.create_visualizations(analysis)
    except Exception as e:
        print(f"⚠️ Could not create visualizations: {e}")
    
    print(f"\n🎉 Performance evaluation completed!")
    print(f"📁 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
