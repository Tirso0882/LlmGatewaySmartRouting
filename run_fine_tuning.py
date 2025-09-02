#!/usr/bin/env python3
"""
Fine-tuning Implementation for LLM Routing (Point X)

This script implements the complete fine-tuning approach using DistilBERT
for intelligent LLM routing classification.

Steps:
1. Generate synthetic dataset
2. Fine-tune DistilBERT model
3. Evaluate and compare with rule-based approach
4. Generate analysis report
"""

import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def main():
    """Main execution function"""
    
    print("🚀 LLM Routing Fine-tuning Implementation (Point X)")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("📊 Step 1: Generating Synthetic Dataset")
    print("-" * 40)
    
    try:
        from src.dataset_generator import DatasetGenerator
        
        generator = DatasetGenerator()
        
        print("Generating training dataset...")
        train_df = generator.generate_dataset(size=1000)
        
        print("Generating test dataset...")
        test_df = generator.generate_dataset(size=200)
        
        generator.save_dataset(train_df, "data/llm_routing_train.csv")
        generator.save_dataset(test_df, "data/llm_routing_test.csv")
        
        print("✅ Dataset generation completed!")
        print()
        
    except Exception as e:
        print(f"❌ Error in dataset generation: {e}")
        return
    
    print("🤖 Step 2: Fine-tuning DistilBERT Model")
    print("-" * 40)
    
    try:
        from src.distilbert_finetuner import DistilBERTFineTuner

        fine_tuner = DistilBERTFineTuner()
        
        print("Preparing data for fine-tuning...")
        train_prompts, val_prompts, train_labels, val_labels = fine_tuner.prepare_data(train_df)
        
        print("Creating data loaders...")
        train_loader, val_loader = fine_tuner.create_data_loaders(
            train_prompts, val_prompts, train_labels, val_labels, batch_size=16
        )
        
        print("Initializing DistilBERT model...")
        fine_tuner.initialize_model()
        
        print("Starting fine-tuning...")
        history = fine_tuner.train(train_loader, val_loader, epochs=3, learning_rate=2e-5)
        
        print("Saving fine-tuned model...")
        fine_tuner.save_model("models/distilbert_llm_router")
        
        print("Generating training plots...")
        fine_tuner.plot_training_history()
        
        print("✅ Fine-tuning completed!")
        print()
        
    except Exception as e:
        print(f"❌ Error in fine-tuning: {e}")
        return
    
    print("🔍 Step 3: Cross-Validation Analysis")
    print("-" * 40)
    
    try:
        print("Performing 5-fold cross-validation...")
        cv_results = fine_tuner.cross_validate(train_df, k_folds=5)
        
        print("✅ Cross-validation completed!")
        print()
        
    except Exception as e:
        print(f"❌ Error in cross-validation: {e}")
        return
    
    print("⚖️ Step 4: Model Comparison (Rule-based vs Fine-tuned)")
    print("-" * 40)
    
    try:
        from src.evaluation_framework import EvaluationFramework

        evaluator = EvaluationFramework()
        
        print("Running model comparison...")
        results = evaluator.compare_models(test_df, "models/distilbert_llm_router")
        
        evaluator.save_results("evaluation_results.json")
        
        print("✅ Model comparison completed!")
        print()
        
    except Exception as e:
        print(f"❌ Error in model comparison: {e}")
        return
    
    print("📋 Step 5: Summary Report")
    print("-" * 40)
    
    try:
        import json
        with open("evaluation_results.json", 'r') as f:
            results = json.load(f)
        
        print("FINE-TUNING IMPLEMENTATION SUMMARY")
        print("=" * 50)
        
        print(f"📊 Dataset Size: {results['test_dataset_size']} test samples")
        print(f"📅 Evaluation Date: {results['evaluation_date']}")
        print()
        
        rule_metrics = results['rule_based']['metrics']
        fine_tuned_metrics = results['fine_tuned']['metrics']
        
        print("PERFORMANCE COMPARISON:")
        print(f"Rule-Based Accuracy: {rule_metrics['accuracy']:.4f}")
        print(f"Fine-Tuned Accuracy: {fine_tuned_metrics['accuracy']:.4f}")
        print(f"Improvement: {((fine_tuned_metrics['accuracy'] - rule_metrics['accuracy']) / rule_metrics['accuracy'] * 100):.2f}%")
        print()
        
        print(f"Rule-Based F1-Macro: {rule_metrics['f1_macro']:.4f}")
        print(f"Fine-Tuned F1-Macro: {fine_tuned_metrics['f1_macro']:.4f}")
        print(f"Improvement: {((fine_tuned_metrics['f1_macro'] - rule_metrics['f1_macro']) / rule_metrics['f1_macro'] * 100):.2f}%")
        print()
        
        print(f"Rule-Based Inference Time: {rule_metrics['inference_time_ms']:.2f}ms")
        print(f"Fine-Tuned Inference Time: {fine_tuned_metrics['inference_time_ms']:.2f}ms")
        print()
        
        print("RECOMMENDATIONS:")
        if fine_tuned_metrics['accuracy'] > rule_metrics['accuracy']:
            print("✅ Fine-tuned model shows better accuracy")
        else:
            print("⚠️ Rule-based model shows better accuracy")
            
        if fine_tuned_metrics['inference_time_ms'] < 100:  # Less than 100ms
            print("✅ Fine-tuned model meets latency requirements")
        else:
            print("⚠️ Fine-tuned model may be too slow for production")
        
        print()
        print("✅ Fine-tuning implementation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in summary generation: {e}")
        return

if __name__ == "__main__":
    main()
