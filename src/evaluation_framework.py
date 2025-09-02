"""
Evaluation Framework for LLM Routing Models

This module provides comprehensive evaluation tools to compare
rule-based and fine-tuned approaches for LLM routing.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.dataset_generator import DatasetGenerator
from src.distilbert_finetuner import DistilBERTFineTuner
from src.prompt_analyzer import PromptAnalyzer


@dataclass
class EvaluationMetrics:
    """Structure for storing evaluation metrics"""
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    cohen_kappa: float
    inference_time_ms: float
    confidence_scores: Optional[List[float]] = None

@dataclass
class ModelComparison:
    """Structure for comparing models"""
    model_name: str
    metrics: EvaluationMetrics
    predictions: List[str]
    detailed_report: str
    confusion_matrix: np.ndarray

class EvaluationFramework:
    """Comprehensive evaluation framework for LLM routing models"""
    
    def __init__(self):
        self.results = {}
        self.test_prompts = []
        self.true_labels = []
        
    def evaluate_rule_based_model(self, test_df: pd.DataFrame) -> ModelComparison:
        """Evaluate the rule-based prompt analyzer"""
        
        print("Evaluating Rule-Based Model...")
        
        analyzer = PromptAnalyzer()
        
        predictions = []
        inference_times = []
        confidence_scores = []
        
        for _, row in test_df.iterrows():
            prompt = row['prompt']
            true_label = row['target_model']
            
            # Measure inference time
            start_time = time.time()
            
            # Analyze prompt and get recommendation
            analysis = analyzer.analyze_prompt(prompt)
            recommended_model = analysis.recommended_model
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # Convert to ms
            
            predictions.append(recommended_model)
            inference_times.append(inference_time)
            
            # For rule-based, we can calculate a confidence score based on complexity match
            complexity_match = 1.0 if analysis.complexity.name.lower() in prompt.lower() else 0.8
            confidence_scores.append(complexity_match)
        
        # Calculate metrics
        true_labels = test_df['target_model'].tolist()
        metrics = self._calculate_metrics(true_labels, predictions, inference_times, confidence_scores)
        
        # Generate detailed report
        report = classification_report(true_labels, predictions, target_names=test_df['target_model'].unique())
        cm = confusion_matrix(true_labels, predictions, labels=test_df['target_model'].unique())
        
        return ModelComparison(
            model_name="Rule-Based Analyzer",
            metrics=metrics,
            predictions=predictions,
            detailed_report=report,
            confusion_matrix=cm
        )
    
    def evaluate_fine_tuned_model(self, test_df: pd.DataFrame, 
                                 model_path: str = "models/distilbert_llm_router") -> ModelComparison:
        """Evaluate the fine-tuned DistilBERT model"""
        
        print("Evaluating Fine-Tuned DistilBERT Model...")
        
        fine_tuner = DistilBERTFineTuner()
        fine_tuner.load_model(model_path)
        
        predictions = []
        inference_times = []
        confidence_scores = []
        
        prompts = test_df['prompt'].tolist()
        
        start_time = time.time()
        batch_predictions, batch_confidences = fine_tuner.predict(prompts)
        end_time = time.time()
        
        # Calculate per-prompt inference time (rough estimate)
        total_time = (end_time - start_time) * 1000  # Convert to ms
        avg_inference_time = total_time / len(prompts)
        
        predictions = batch_predictions
        confidence_scores = batch_confidences
        inference_times = [avg_inference_time] * len(prompts)
        
        # Calculate metrics
        true_labels = test_df['target_model'].tolist()
        metrics = self._calculate_metrics(true_labels, predictions, inference_times, confidence_scores)
        
        # Generate detailed report
        unique_labels = sorted(test_df['target_model'].unique())
        report = classification_report(true_labels, predictions, target_names=unique_labels)
        cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
        
        return ModelComparison(
            model_name="Fine-Tuned DistilBERT",
            metrics=metrics,
            predictions=predictions,
            detailed_report=report,
            confusion_matrix=cm
        )
    
    def _calculate_metrics(self, true_labels: List[str], predictions: List[str],
                          inference_times: List[float], confidence_scores: List[float]) -> EvaluationMetrics:
        """Calculate evaluation metrics"""
        
        return EvaluationMetrics(
            accuracy=accuracy_score(true_labels, predictions),
            f1_macro=f1_score(true_labels, predictions, average='macro'),
            f1_weighted=f1_score(true_labels, predictions, average='weighted'),
            precision_macro=precision_score(true_labels, predictions, average='macro'),
            recall_macro=recall_score(true_labels, predictions, average='macro'),
            cohen_kappa=cohen_kappa_score(true_labels, predictions),
            inference_time_ms=np.mean(inference_times),
            confidence_scores=confidence_scores
        )
    
    def compare_models(self, test_df: pd.DataFrame, model_path: str = None) -> Dict:
        """Compare rule-based vs fine-tuned models"""
        
        print("Starting Model Comparison...")
        print("=" * 60)
        
        rule_based_results = self.evaluate_rule_based_model(test_df)
        
        if model_path and os.path.exists(model_path):
            fine_tuned_results = self.evaluate_fine_tuned_model(test_df, model_path)
        else:
            print("Fine-tuned model not found. Training new model...")
            fine_tuned_results = self._train_and_evaluate_model(test_df)
        
        self.results = {
            'rule_based': rule_based_results,
            'fine_tuned': fine_tuned_results,
            'test_dataset_size': len(test_df),
            'evaluation_date': datetime.now().isoformat()
        }
        
        self._print_comparison()
        
        self._create_comparison_plots()
        
        return self.results
    
    def _train_and_evaluate_model(self, test_df: pd.DataFrame) -> ModelComparison:
        """Train a new model and evaluate it (fallback method)"""
        
        
        generator = DatasetGenerator()
        train_df = generator.generate_dataset(size=800)
        
        fine_tuner = DistilBERTFineTuner()
        train_prompts, val_prompts, train_labels, val_labels = fine_tuner.prepare_data(train_df)
        
        train_loader, val_loader = fine_tuner.create_data_loaders(
            train_prompts, val_prompts, train_labels, val_labels, batch_size=8
        )
        
        fine_tuner.initialize_model()
        fine_tuner.train(train_loader, val_loader, epochs=2)
        
        model_path = "models/distilbert_llm_router_temp"
        fine_tuner.save_model(model_path)
        
        return self.evaluate_fine_tuned_model(test_df, model_path)
    
    def _print_comparison(self):
        """Print detailed comparison results"""
        
        rule_metrics = self.results['rule_based'].metrics
        fine_tuned_metrics = self.results['fine_tuned'].metrics
        
        print("\nMODEL COMPARISON RESULTS")
        print("=" * 60)
        
        print(f"{'Metric':<20} {'Rule-Based':<15} {'Fine-Tuned':<15} {'Winner':<10}")
        print("-" * 60)
        
        metrics_to_compare = [
            ('Accuracy', rule_metrics.accuracy, fine_tuned_metrics.accuracy),
            ('F1-Macro', rule_metrics.f1_macro, fine_tuned_metrics.f1_macro),
            ('F1-Weighted', rule_metrics.f1_weighted, fine_tuned_metrics.f1_weighted),
            ('Precision', rule_metrics.precision_macro, fine_tuned_metrics.precision_macro),
            ('Recall', rule_metrics.recall_macro, fine_tuned_metrics.recall_macro),
            ('Cohen Kappa', rule_metrics.cohen_kappa, fine_tuned_metrics.cohen_kappa),
            ('Inference (ms)', rule_metrics.inference_time_ms, fine_tuned_metrics.inference_time_ms),
        ]
        
        for metric_name, rule_val, fine_tuned_val in metrics_to_compare:
            if metric_name == 'Inference (ms)':
                winner = "Rule-Based" if rule_val < fine_tuned_val else "Fine-Tuned"
            else:
                winner = "Rule-Based" if rule_val > fine_tuned_val else "Fine-Tuned"
            
            print(f"{metric_name:<20} {rule_val:<15.4f} {fine_tuned_val:<15.4f} {winner:<10}")
        
        print("\nDETAILED CLASSIFICATION REPORTS")
        print("=" * 60)
        
        print("\nRule-Based Model:")
        print(self.results['rule_based'].detailed_report)
        
        print("\nFine-Tuned Model:")
        print(self.results['fine_tuned'].detailed_report)
    
    def _create_comparison_plots(self):
        """Create comprehensive comparison visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Metrics Comparison Bar Chart
        metrics_names = ['Accuracy', 'F1-Macro', 'F1-Weighted', 'Precision', 'Recall', 'Cohen Kappa']
        rule_values = [
            self.results['rule_based'].metrics.accuracy,
            self.results['rule_based'].metrics.f1_macro,
            self.results['rule_based'].metrics.f1_weighted,
            self.results['rule_based'].metrics.precision_macro,
            self.results['rule_based'].metrics.recall_macro,
            self.results['rule_based'].metrics.cohen_kappa
        ]
        fine_tuned_values = [
            self.results['fine_tuned'].metrics.accuracy,
            self.results['fine_tuned'].metrics.f1_macro,
            self.results['fine_tuned'].metrics.f1_weighted,
            self.results['fine_tuned'].metrics.precision_macro,
            self.results['fine_tuned'].metrics.recall_macro,
            self.results['fine_tuned'].metrics.cohen_kappa
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, rule_values, width, label='Rule-Based', alpha=0.8)
        axes[0, 0].bar(x + width/2, fine_tuned_values, width, label='Fine-Tuned', alpha=0.8)
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Inference Time Comparison
        models = ['Rule-Based', 'Fine-Tuned']
        inference_times = [
            self.results['rule_based'].metrics.inference_time_ms,
            self.results['fine_tuned'].metrics.inference_time_ms
        ]
        
        bars = axes[0, 1].bar(models, inference_times, color=['skyblue', 'lightcoral'])
        axes[0, 1].set_ylabel('Inference Time (ms)')
        axes[0, 1].set_title('Inference Time Comparison')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, inference_times):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{time_val:.2f}ms', ha='center', va='bottom')
        
        # 3. Confusion Matrix - Rule Based
        cm_rule = self.results['rule_based'].confusion_matrix
        labels = ['gpt-4o-mini', 'o4-mini', 'o3']  # Assuming these are the labels
        
        sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=axes[0, 2])
        axes[0, 2].set_title('Rule-Based Confusion Matrix')
        axes[0, 2].set_xlabel('Predicted')
        axes[0, 2].set_ylabel('Actual')
        
        # 4. Confusion Matrix - Fine Tuned
        cm_fine = self.results['fine_tuned'].confusion_matrix
        
        sns.heatmap(cm_fine, annot=True, fmt='d', cmap='Reds',
                   xticklabels=labels, yticklabels=labels, ax=axes[1, 0])
        axes[1, 0].set_title('Fine-Tuned Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 5. Confidence Score Distribution
        if (self.results['rule_based'].metrics.confidence_scores and 
            self.results['fine_tuned'].metrics.confidence_scores):
            
            axes[1, 1].hist(self.results['rule_based'].metrics.confidence_scores, 
                           alpha=0.7, label='Rule-Based', bins=20, density=True)
            axes[1, 1].hist(self.results['fine_tuned'].metrics.confidence_scores, 
                           alpha=0.7, label='Fine-Tuned', bins=20, density=True)
            axes[1, 1].set_xlabel('Confidence Score')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Confidence Score Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model Efficiency (Accuracy vs Speed)
        accuracy_values = [rule_values[0], fine_tuned_values[0]]
        speed_values = [1/t for t in inference_times]  # Convert to throughput
        
        axes[1, 2].scatter(accuracy_values[0], speed_values[0], 
                          s=100, label='Rule-Based', alpha=0.8)
        axes[1, 2].scatter(accuracy_values[1], speed_values[1], 
                          s=100, label='Fine-Tuned', alpha=0.8)
        axes[1, 2].set_xlabel('Accuracy')
        axes[1, 2].set_ylabel('Throughput (predictions/ms)')
        axes[1, 2].set_title('Accuracy vs Speed Trade-off')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, filepath: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        
        serializable_results = {
            'rule_based': {
                'model_name': self.results['rule_based'].model_name,
                'metrics': {
                    'accuracy': self.results['rule_based'].metrics.accuracy,
                    'f1_macro': self.results['rule_based'].metrics.f1_macro,
                    'f1_weighted': self.results['rule_based'].metrics.f1_weighted,
                    'precision_macro': self.results['rule_based'].metrics.precision_macro,
                    'recall_macro': self.results['rule_based'].metrics.recall_macro,
                    'cohen_kappa': self.results['rule_based'].metrics.cohen_kappa,
                    'inference_time_ms': self.results['rule_based'].metrics.inference_time_ms
                },
                'detailed_report': self.results['rule_based'].detailed_report,
                'confusion_matrix': self.results['rule_based'].confusion_matrix.tolist()
            },
            'fine_tuned': {
                'model_name': self.results['fine_tuned'].model_name,
                'metrics': {
                    'accuracy': self.results['fine_tuned'].metrics.accuracy,
                    'f1_macro': self.results['fine_tuned'].metrics.f1_macro,
                    'f1_weighted': self.results['fine_tuned'].metrics.f1_weighted,
                    'precision_macro': self.results['fine_tuned'].metrics.precision_macro,
                    'recall_macro': self.results['fine_tuned'].metrics.recall_macro,
                    'cohen_kappa': self.results['fine_tuned'].metrics.cohen_kappa,
                    'inference_time_ms': self.results['fine_tuned'].metrics.inference_time_ms
                },
                'detailed_report': self.results['fine_tuned'].detailed_report,
                'confusion_matrix': self.results['fine_tuned'].confusion_matrix.tolist()
            },
            'test_dataset_size': self.results['test_dataset_size'],
            'evaluation_date': self.results['evaluation_date']
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {filepath}")


if __name__ == "__main__":
    
    evaluator = EvaluationFramework()
    
    test_df = pd.read_csv("data/llm_routing_test.csv")
    
    results = evaluator.compare_models(test_df)
    
    evaluator.save_results()
    
    print("Evaluation completed!")
