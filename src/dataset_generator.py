"""
Dataset Generator for LLM Routing Fine-tuning

This module creates a synthetic dataset for training a DistilBERT model
to classify prompts and route them to appropriate LLMs.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass
class PromptTemplate:
    """Template for generating synthetic prompts"""
    template: str
    complexity: str
    domain: str
    preferred_model: str
    features: List[str]

class DatasetGenerator:
    """Generate synthetic dataset for LLM routing classification"""
    
    def __init__(self):
        self.prompt_templates = self._create_prompt_templates()
        self.model_mapping = {
            'speed': 'gpt-4o-mini',
            'cost': 'o4-mini', 
            'accuracy': 'o3'
        }
        
    def _create_prompt_templates(self) -> List[PromptTemplate]:
        """Create comprehensive prompt templates for different scenarios"""
        
        templates = [
            # Speed-optimized prompts (gpt-4o-mini) 
            PromptTemplate(
                "What is {topic}?",
                "simple", "general", "gpt-4o-mini", ["question", "simple"]
            ),
            PromptTemplate(
                "Quick answer: {question}",
                "simple", "general", "gpt-4o-mini", ["question", "urgent"]
            ),
            PromptTemplate(
                "Briefly explain {concept}",
                "simple", "general", "gpt-4o-mini", ["question", "simple"]
            ),
            PromptTemplate(
                "What does {term} mean?",
                "simple", "general", "gpt-4o-mini", ["question", "simple"]
            ),
            PromptTemplate(
                "Give me a short summary of {topic}",
                "simple", "general", "gpt-4o-mini", ["question", "simple"]
            ),
            
            # Cost-optimized prompts (o4-mini) 
            PromptTemplate(
                "Write a {language} function to {task}",
                "moderate", "code", "o4-mini", ["code", "programming"]
            ),
            PromptTemplate(
                "Create a simple {type} algorithm for {problem}",
                "moderate", "code", "o4-mini", ["code", "algorithm"]
            ),
            PromptTemplate(
                "Explain how to {task} in {technology}",
                "moderate", "general", "o4-mini", ["explanation", "technical"]
            ),
            PromptTemplate(
                "Help me debug this {language} code: {code_snippet}",
                "moderate", "code", "o4-mini", ["code", "debugging"]
            ),
            PromptTemplate(
                "What's the difference between {concept1} and {concept2}?",
                "moderate", "general", "o4-mini", ["comparison", "question"]
            ),
            
            # Accuracy-optimized prompts (o3)
            PromptTemplate(
                "Solve the differential equation: {equation} with initial conditions {conditions}",
                "complex", "math", "o3", ["math", "complex", "stem"]
            ),
            PromptTemplate(
                "Derive the {formula} from first principles and explain each step",
                "complex", "math", "o3", ["math", "derivation", "complex"]
            ),
            PromptTemplate(
                "Analyze the quantum mechanical {system} and calculate {property}",
                "complex", "math", "o3", ["math", "physics", "complex", "stem"]
            ),
            PromptTemplate(
                "Prove that {mathematical_statement} using {proof_method}",
                "complex", "math", "o3", ["math", "proof", "complex", "stem"]
            ),
            PromptTemplate(
                "Design a comprehensive {architecture} system for {complex_problem} considering {constraints}",
                "complex", "analysis", "o3", ["architecture", "complex", "analysis"]
            ),
            PromptTemplate(
                "Conduct a thorough analysis of {dataset} and identify {patterns} using {statistical_method}",
                "complex", "analysis", "o3", ["analysis", "statistics", "complex"]
            ),
            PromptTemplate(
                "Develop a machine learning model to {ml_task} with {performance_requirements}",
                "complex", "code", "o3", ["code", "ml", "complex"]
            ),
        ]
        
        return templates
    
    def _fill_template(self, template: PromptTemplate) -> Tuple[str, str]:
        """Fill template with realistic values"""
        
        replacements = {
            # General topics
            'topic': ['artificial intelligence', 'climate change', 'blockchain', 'democracy', 
                     'renewable energy', 'cybersecurity', 'space exploration'],
            'question': ['How tall is the Eiffel Tower?', 'What is the capital of Japan?', 
                        'When was Python created?', 'Who invented the telephone?'],
            'concept': ['machine learning', 'quantum computing', 'neural networks', 'algorithms'],
            'term': ['API', 'cryptocurrency', 'GUI', 'database', 'framework'],
            
            # Programming
            'language': ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust'],
            'task': ['sort a list', 'find prime numbers', 'calculate fibonacci', 'reverse a string'],
            'type': ['sorting', 'searching', 'graph traversal', 'dynamic programming'],
            'problem': ['shortest path', 'traveling salesman', 'knapsack problem'],
            'technology': ['React', 'Docker', 'Kubernetes', 'AWS', 'MongoDB'],
            'code_snippet': ['def func(): return None', 'for i in range(10): print(i)', 
                           'class MyClass: pass'],
            
            # Math and science
            'equation': ['dy/dx = x^2 + 2x + 1', 'd²y/dx² + 4y = 0', 'dy/dx + y = e^x'],
            'conditions': ['y(0) = 1', 'y(0) = 2, y\'(0) = 0', 'y(1) = 3'],
            'formula': ['quadratic formula', 'chain rule', 'integration by parts'],
            'system': ['hydrogen atom', 'harmonic oscillator', 'particle in a box'],
            'property': ['energy levels', 'wave function', 'probability density'],
            'mathematical_statement': ['Fermat\'s Last Theorem', 'Pythagorean Theorem', 
                                     'Prime Number Theorem'],
            'proof_method': ['mathematical induction', 'contradiction', 'direct proof'],
            
            # Complex analysis
            'architecture': ['microservices', 'distributed', 'event-driven', 'serverless'],
            'complex_problem': ['fraud detection', 'recommendation engine', 'supply chain optimization'],
            'constraints': ['real-time performance', 'high availability', 'cost efficiency'],
            'dataset': ['customer behavior data', 'financial time series', 'social media data'],
            'patterns': ['anomalies', 'trends', 'correlations', 'clusters'],
            'statistical_method': ['regression analysis', 'time series analysis', 'clustering'],
            'ml_task': ['predict customer churn', 'classify images', 'detect fraud'],
            'performance_requirements': ['99% accuracy', 'sub-second latency', '99.9% uptime'],
            
            # Comparisons
            'concept1': ['React', 'REST', 'SQL', 'supervised learning'],
            'concept2': ['Vue', 'GraphQL', 'NoSQL', 'unsupervised learning'],
        }
        
        filled_template = template.template
        for placeholder, values in replacements.items():
            if f'{{{placeholder}}}' in filled_template:
                filled_template = filled_template.replace(
                    f'{{{placeholder}}}', random.choice(values)
                )
        
        return filled_template, template.preferred_model
    
    def generate_dataset(self, size: int = 1000) -> pd.DataFrame:
        """Generate synthetic dataset with balanced classes"""
        
        data = []
        
        # Ensure balanced classes by splitting samples evenly among models
        samples_per_model = size // 3
        
        for model in ['gpt-4o-mini', 'o4-mini', 'o3']:
            model_templates = [t for t in self.prompt_templates if t.preferred_model == model]
            
            for _ in range(samples_per_model):
                template = random.choice(model_templates)
                prompt, target_model = self._fill_template(template)
                
                # Add some noise and variations
                if random.random() < 0.1:  # 10% noise
                    noise_prefixes = ["Please ", "Can you ", "I need to ", "Help me "]
                    prompt = random.choice(noise_prefixes) + prompt.lower()
                
                data.append({
                    'prompt': prompt,
                    'target_model': target_model,
                    'complexity': template.complexity,
                    'domain': template.domain,
                    'features': ','.join(template.features),
                    'prompt_length': len(prompt.split()),
                    'has_question': '?' in prompt,
                    'has_code_keywords': any(kw in prompt.lower() for kw in 
                                           ['function', 'code', 'debug', 'algorithm', 'class']),
                    'has_math_keywords': any(kw in prompt.lower() for kw in 
                                           ['equation', 'solve', 'calculate', 'derive', 'prove']),
                })
        
        # Add remaining samples to reach exact size
        remaining = size - len(data)
        for _ in range(remaining):
            template = random.choice(self.prompt_templates)
            prompt, target_model = self._fill_template(template)
            
            data.append({
                'prompt': prompt,
                'target_model': target_model,
                'complexity': template.complexity,
                'domain': template.domain,
                'features': ','.join(template.features),
                'prompt_length': len(prompt.split()),
                'has_question': '?' in prompt,
                'has_code_keywords': any(kw in prompt.lower() for kw in 
                                       ['function', 'code', 'debug', 'algorithm', 'class']),
                'has_math_keywords': any(kw in prompt.lower() for kw in 
                                       ['equation', 'solve', 'calculate', 'derive', 'prove']),
            })
        
        df = pd.DataFrame(data)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def _ensure_data_directory(self, filepath: str):
        """Ensure the data directory exists, create if it doesn't"""
        directory = os.path.dirname(filepath)
        
        if directory and not os.path.exists(directory):
            print(f"📁 Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory created successfully")
        elif directory and os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"📁 Saving to current directory")
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to file with automatic directory creation"""
        self._ensure_data_directory(filepath)
        
        df.to_csv(filepath, index=False)
        print(f"💾 Dataset saved to {filepath}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎯 Target distribution:\n{df['target_model'].value_counts()}")

if __name__ == "__main__":
    print("🚀 Starting LLM Routing Dataset Generation")
    print("=" * 50)
    
    current_dir = os.getcwd()
    print(f"📂 Current working directory: {current_dir}")
    
    generator = DatasetGenerator()
    
    print("\n📊 Generating training dataset...")
    train_df = generator.generate_dataset(size=1000)
    
    print("\n📊 Generating test dataset...")
    test_df = generator.generate_dataset(size=200)
    
    print("\n💾 Saving datasets...")
    generator.save_dataset(train_df, "data/llm_routing_train.csv")
    generator.save_dataset(test_df, "data/llm_routing_test.csv")
    
    print("\n🎉 Dataset generation completed!")
    print("=" * 50)
    print(f"📈 Training set: {len(train_df)} samples")
    print(f"📈 Test set: {len(test_df)} samples")
    print(f"📁 Files saved to: {os.path.join(current_dir, 'data')}")
    
    # Verify files were created
    train_file = "data/llm_routing_train.csv"
    test_file = "data/llm_routing_test.csv"
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("✅ All dataset files created successfully!")
        print(f"   - {train_file}: {os.path.getsize(train_file)} bytes")
        print(f"   - {test_file}: {os.path.getsize(test_file)} bytes")
    else:
        print("❌ Error: Some dataset files were not created")
        print(f"   - {train_file}: {'✅' if os.path.exists(train_file) else '❌'}")
        print(f"   - {test_file}: {'✅' if os.path.exists(test_file) else '❌'}")
