"""
DistilBERT Inference Only - Lightweight version for deployment

This module implements inference-only functionality for DistilBERT-based
LLM routing classification, optimized for production deployment.
"""

import json
import os
import pickle
import warnings
from typing import List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)

warnings.filterwarnings('ignore')


class DistilBERTFineTuner:
    """Lightweight DistilBERT class for inference only - production optimized"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', device: str = None):
        self.model_name = model_name
        
        # Detect best available device for optimal performance
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = 'cuda'  # NVIDIA GPU
        else:
            self.device = 'cpu'  # Fallback to CPU
            
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.label_names = None
        self.num_labels = None
        
        print(f"🤖 DistilBERT initialized on device: {self.device}")
    
    def load_model(self, load_path: str):
        """Load a fine-tuned model - using exact working logic from distilbert_finetuner.py"""
        try:
            # Load model and tokenizer - EXACT same as working version
            self.model = DistilBertForSequenceClassification.from_pretrained(load_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(load_path)
            self.model.to(self.device)
            
            # Load label encoder - EXACT same as working version
            import joblib
            self.label_encoder = joblib.load(os.path.join(load_path, 'label_encoder.pkl'))
            
            # Set label-related attributes
            if hasattr(self.label_encoder, 'classes_'):
                self.label_names = self.label_encoder.classes_
                self.num_labels = len(self.label_names)
            else:
                self.label_names = ['o3', 'o4-mini', 'gpt-4o-mini']
                self.num_labels = 3
            
            print(f"✅ Model loaded successfully from {load_path}")
            print(f"📊 Label mapping: {dict(zip(self.label_names, range(self.num_labels)))}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, prompts: List[str]) -> Tuple[List[str], List[float]]:
        """Make predictions on new prompts - using exact working logic from distilbert_finetuner.py"""
        if not self.model or not self.tokenizer or not self.label_encoder:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        self.model.eval()
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                encoding = self.tokenizer(
                    prompt,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Get prediction and confidence - EXACT same logic as working version
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
                
                predictions.append(predicted_label)
                confidences.append(confidence)
        
        return predictions, confidences


if __name__ == "__main__":
    print("🧪 Testing DistilBERT Inference Module")
    
    # Initialize the model
    fine_tuner = DistilBERTFineTuner()
    
    # Test prompts covering different complexity levels
    test_prompts = [
        "What is the weather like today?",  # Simple question - should route to gpt-4o-mini
        "Write a complex research paper about quantum computing with detailed analysis",  # Complex task - should route to o3
        "Hello, how are you?",  # Simple greeting - should route to gpt-4o-mini
        "Create a comprehensive business strategy for a tech startup including market analysis, financial projections, and competitive landscape",  # Complex business task - should route to o3
        "What's 2+2?",  # Simple math - should route to gpt-4o-mini
        "Develop a machine learning model to predict stock prices using deep learning techniques and explain the mathematical foundations",  # Complex ML task - should route to o3
        "How do I make a sandwich?",  # Simple instruction - should route to gpt-4o-mini
        "Analyze the philosophical implications of artificial general intelligence on human society and ethics",  # Complex philosophical question - should route to o3
    ]
    
    print(f"🤖 Device: {fine_tuner.device}")
    print(f"📝 Testing with {len(test_prompts)} prompts")
    
    # Try to load the model if it exists
    model_path = "../models/distilbert_llm_router"
    if os.path.exists(model_path):
        print(f"📁 Loading model from: {model_path}")
        success = fine_tuner.load_model(model_path)
        
        if success:
            print("\n🚀 Running predictions on test prompts...")
            print("=" * 80)
            
            try:
                predictions, confidences = fine_tuner.predict(test_prompts)
                
                for i, (prompt, prediction, confidence) in enumerate(zip(test_prompts, predictions, confidences), 1):
                    print(f"\n📝 Prompt {i}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
                    print(f"🎯 Predicted Model: {prediction}")
                    print(f"📊 Confidence: {confidence:.2%}")
                    print("-" * 40)
                
                print(f"\n✅ Successfully processed {len(predictions)} predictions!")
                
                # Summary statistics
                model_counts = {}
                for pred in predictions:
                    model_counts[pred] = model_counts.get(pred, 0) + 1
                
                print(f"\n📈 Prediction Summary:")
                for model, count in model_counts.items():
                    print(f"   {model}: {count} predictions")
                
                avg_confidence = sum(confidences) / len(confidences)
                print(f"   Average Confidence: {avg_confidence:.2%}")
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                print("💡 This might be expected if the model files are not properly trained")
        else:
            print("❌ Failed to load model")
    else:
        print(f"⚠️ Model path not found: {model_path}")
        print("💡 Make sure you have trained the model first using the training script")
    
    print("\n✅ DistilBERT Inference module test completed!")
    print("💡 To use in production: fine_tuner.load_model('path/to/model') then fine_tuner.predict(prompts)")
