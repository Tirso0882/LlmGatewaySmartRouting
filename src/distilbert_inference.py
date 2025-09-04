"""
DistilBERT Inference Only - Lightweight version for deployment
"""

import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer)


class DistilBERTFineTuner:
    """Lightweight DistilBERT class for inference only"""
    
    def __init__(self, device: str = None):
        # Detect best available device
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = 'cuda'  # NVIDIA GPU
        else:
            self.device = 'cpu'  # Fallback to CPU
            
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        
        print(f"🤖 Using device: {self.device}")
    
    def load_model(self, load_path: str):
        """Load a fine-tuned model"""
        try:
            self.model = DistilBertForSequenceClassification.from_pretrained(load_path)
            self.tokenizer = DistilBertTokenizer.from_pretrained(load_path)
            self.model.to(self.device)
            
            # Load label encoder
            label_encoder_path = os.path.join(load_path, 'label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                # Try joblib format (backup)
                try:
                    import joblib
                    self.label_encoder = joblib.load(label_encoder_path)
                except:
                    # Fallback label encoder if file is missing
                    print("⚠️ Using fallback label encoder")
                    class FallbackEncoder:
                        def __init__(self):
                            self.classes_ = ['o3', 'o4-mini', 'gpt-4o-mini']
                        def inverse_transform(self, indices):
                            return [self.classes_[i] for i in indices]
                    self.label_encoder = FallbackEncoder()
            
            print(f"✅ Model loaded successfully from {load_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def predict(self, prompts: List[str]) -> Tuple[List[str], List[float]]:
        """Make predictions on new prompts"""
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
                
                # Get prediction and confidence
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_tensor = torch.argmax(probabilities, dim=-1)
                confidence_tensor = torch.max(probabilities)
                
                # Safely extract scalar values with comprehensive validation
                try:
                    # Convert tensors to CPU first to avoid device issues
                    predicted_class_tensor = predicted_class_tensor.cpu()
                    confidence_tensor = confidence_tensor.cpu()
                    
                    # Ensure we're working with scalar tensors - use safer dimension checking
                    if predicted_class_tensor.numel() > 1:
                        predicted_class_tensor = predicted_class_tensor.squeeze()
                    if confidence_tensor.numel() > 1:
                        confidence_tensor = confidence_tensor.squeeze()
                    
                    # Final safety check - ensure tensors are 0-dimensional (scalars)
                    if predicted_class_tensor.dim() > 0:
                        predicted_class_tensor = predicted_class_tensor.flatten()[0]
                    if confidence_tensor.dim() > 0:
                        confidence_tensor = confidence_tensor.flatten()[0]
                    
                    # Extract scalar values safely with type conversion
                    predicted_class_raw = predicted_class_tensor.item()
                    confidence_raw = confidence_tensor.item()
                    
                    # Convert to proper types with validation
                    predicted_class = int(predicted_class_raw)
                    confidence = float(confidence_raw)
                    
                    # Validate predicted_class is within expected range
                    if predicted_class < 0:
                        predicted_class = 0
                    elif predicted_class > 2:  # We have 3 classes (0, 1, 2)
                        predicted_class = 2
                        
                    # Validate confidence is within expected range
                    if confidence < 0.0:
                        confidence = 0.0
                    elif confidence > 1.0:
                        confidence = 1.0
                        
                except Exception as tensor_error:
                    print(f"⚠️ Tensor conversion error: {tensor_error}")
                    print(f"⚠️ Error type: {type(tensor_error).__name__}")
                    # Fallback values
                    predicted_class = 1  # Default to middle class (o4-mini)
                    confidence = 0.5
                
                # Convert to label using safer approach
                try:
                    # Use label encoder to get the actual label
                    if (self.label_encoder and 
                        hasattr(self.label_encoder, 'inverse_transform') and 
                        hasattr(self.label_encoder, 'classes_')):
                        
                        # Ensure predicted_class is within valid range for label encoder
                        num_classes = len(self.label_encoder.classes_)
                        if predicted_class >= num_classes:
                            predicted_class = num_classes - 1
                        elif predicted_class < 0:
                            predicted_class = 0
                            
                        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
                    else:
                        raise ValueError("Label encoder not properly loaded")
                        
                except Exception as e:
                    # Fallback to index-based mapping
                    labels = ['o3', 'o4-mini', 'gpt-4o-mini']  # Updated to match current models
                    try:
                        if 0 <= predicted_class < len(labels):
                            predicted_label = labels[predicted_class]
                        else:
                            predicted_label = 'o4-mini'  # Default fallback
                    except Exception as fallback_error:
                        predicted_label = 'o4-mini'  # Ultimate fallback
                        print(f"⚠️ Fallback conversion failed: {fallback_error}")
                    print(f"⚠️ Label encoder failed, using fallback: {e}")
                    print(f"   Predicted class: {predicted_class}, Fallback label: {predicted_label}")
                
                predictions.append(predicted_label)
                confidences.append(confidence)
        
        return predictions, confidences
