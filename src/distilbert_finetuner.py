"""
DistilBERT Fine-tuning for LLM Routing Classification

This module implements fine-tuning of DistilBERT for intelligent LLM routing
using prompt classification.
"""

import json
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizer, get_linear_schedule_with_warmup)

warnings.filterwarnings('ignore')

class LLMRoutingDataset(Dataset):
    """Dataset class for LLM routing prompts"""
    
    def __init__(self, prompts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.prompts = prompts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = str(self.prompts[idx])
        label = self.labels[idx]
        
        # Tokenize the prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DistilBERTFineTuner:
    """Fine-tune DistilBERT for LLM routing classification"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased', device: str = None):
        self.model_name = model_name
        
        # Detect best available device for MacBook Pro M3
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = 'mps'  # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = 'cuda'  # NVIDIA GPU
        else:
            self.device = 'cpu'  # Fallback to CPU
            
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
        
        print(f"Using device: {self.device}")
        if self.device == 'mps':
            print("🚀 Using Apple Silicon GPU (MPS) for accelerated training!")
        elif self.device == 'cuda':
            print("🚀 Using NVIDIA GPU (CUDA) for accelerated training!")
        else:
            print("⚠️ Using CPU for training (slower)")
    
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        """Prepare data for training"""
        
        # Encode labels
        self.labels = self.label_encoder.fit_transform(df['target_model'])
        self.label_names = self.label_encoder.classes_
        self.num_labels = len(self.label_names)
        
        print(f"Label mapping: {dict(zip(self.label_names, range(self.num_labels)))}")
        print(f"Number of classes: {self.num_labels}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        prompts = df['prompt'].tolist()
        
        train_prompts, val_prompts, train_labels, val_labels = train_test_split(
            prompts, self.labels, test_size=test_size, random_state=42, 
            stratify=self.labels
        )
        
        return train_prompts, val_prompts, train_labels, val_labels
    
    def create_data_loaders(self, train_prompts, val_prompts, train_labels, val_labels, 
                          batch_size: int = 16) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch data loaders"""
        
        train_dataset = LLMRoutingDataset(
            train_prompts, train_labels, self.tokenizer
        )
        
        val_dataset = LLMRoutingDataset(
            val_prompts, val_labels, self.tokenizer
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        return train_loader, val_loader
    
    def initialize_model(self):
        """Initialize DistilBERT model for classification"""
        
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        self.model.to(self.device)
        
        print(f"Model initialized with {self.num_labels} classes")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_epoch(self, train_loader, optimizer, scheduler) -> Tuple[float, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            current_accuracy = correct_predictions / total_predictions
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'accuracy': f'{current_accuracy:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader) -> Tuple[float, float, float]:
        """Evaluate model on validation set"""
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='macro')
        
        return avg_loss, accuracy, f1
    
    def train(self, train_loader, val_loader, epochs: int = 3, 
              learning_rate: float = 2e-5) -> Dict:
        """Main training loop"""
        
        # Initialize optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Total training steps: {total_steps}")
        
        best_f1 = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Evaluate
            val_loss, val_accuracy, val_f1 = self.evaluate(val_loader)
            
            # Save metrics
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_accuracy)
            self.training_history['val_f1'].append(val_f1)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
                print(f"New best F1 score: {best_f1:.4f}")
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\nTraining completed. Best F1 score: {best_f1:.4f}")
        
        return self.training_history
    
    def predict(self, prompts: List[str]) -> Tuple[List[str], List[float]]:
        """Make predictions on new prompts"""
        
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
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                
                predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
                
                predictions.append(predicted_label)
                confidences.append(confidence)
        
        return predictions, confidences
    
    def cross_validate(self, df: pd.DataFrame, k_folds: int = 5) -> Dict:
        """Perform k-fold cross-validation"""
        
        print(f"\nPerforming {k_folds}-fold cross-validation...")
        
        # Prepare labels
        labels = self.label_encoder.fit_transform(df['target_model'])
        prompts = df['prompt'].tolist()
        self.num_labels = len(self.label_encoder.classes_)
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        cv_results = {
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(prompts, labels)):
            print(f"\nFold {fold + 1}/{k_folds}")
            
            # Split data
            train_prompts = [prompts[i] for i in train_idx]
            val_prompts = [prompts[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(
                train_prompts, val_prompts, train_labels, val_labels, batch_size=16
            )
            
            # Initialize fresh model
            self.initialize_model()
            
            # Train
            self.train(train_loader, val_loader, epochs=2, learning_rate=2e-5)
            
            # Evaluate
            predictions, _ = self.predict(val_prompts)
            predicted_labels = self.label_encoder.transform(predictions)
            
            # Calculate metrics
            accuracy = accuracy_score(val_labels, predicted_labels)
            f1_macro = f1_score(val_labels, predicted_labels, average='macro')
            f1_weighted = f1_score(val_labels, predicted_labels, average='weighted')
            
            cv_results['accuracy'].append(accuracy)
            cv_results['f1_macro'].append(f1_macro)
            cv_results['f1_weighted'].append(f1_weighted)
            
            print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
        
        # Calculate final metrics
        final_results = {
            'accuracy_mean': np.mean(cv_results['accuracy']),
            'accuracy_std': np.std(cv_results['accuracy']),
            'f1_macro_mean': np.mean(cv_results['f1_macro']),
            'f1_macro_std': np.std(cv_results['f1_macro']),
            'f1_weighted_mean': np.mean(cv_results['f1_weighted']),
            'f1_weighted_std': np.std(cv_results['f1_weighted'])
        }
        
        print(f"\nCross-Validation Results:")
        print(f"Accuracy: {final_results['accuracy_mean']:.4f} ± {final_results['accuracy_std']:.4f}")
        print(f"F1-Macro: {final_results['f1_macro_mean']:.4f} ± {final_results['f1_macro_std']:.4f}")
        print(f"F1-Weighted: {final_results['f1_weighted_mean']:.4f} ± {final_results['f1_weighted_std']:.4f}")
        
        return final_results
    
    def save_model(self, save_path: str):
        """Save the fine-tuned model"""
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, os.path.join(save_path, 'label_encoder.pkl'))
        
        # Save training history
        with open(os.path.join(save_path, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a fine-tuned model"""
        
        self.model = DistilBertForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        
        # Load label encoder
        import joblib
        self.label_encoder = joblib.load(os.path.join(load_path, 'label_encoder.pkl'))
        
        # Load training history
        history_path = os.path.join(load_path, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        print(f"Model loaded from {load_path}")
    
    def plot_training_history(self):
        """Plot training metrics"""
        
        if not self.training_history['train_loss']:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(epochs, self.training_history['train_accuracy'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.training_history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # F1 Score
        axes[1, 0].plot(epochs, self.training_history['val_f1'], 'g-', label='Validation F1')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        
        # Model comparison (placeholder)
        axes[1, 1].text(0.5, 0.5, 'Model Comparison\n(Implementation specific)', 
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Model Performance')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Example usage
    fine_tuner = DistilBERTFineTuner()
    
    # Load data
    df = pd.read_csv("data/llm_routing_train.csv")
    
    # Prepare data
    train_prompts, val_prompts, train_labels, val_labels = fine_tuner.prepare_data(df)
    
    # Create data loaders
    train_loader, val_loader = fine_tuner.create_data_loaders(
        train_prompts, val_prompts, train_labels, val_labels
    )
    
    # Initialize and train model
    fine_tuner.initialize_model()
    history = fine_tuner.train(train_loader, val_loader, epochs=3)
    
    # Save model
    fine_tuner.save_model("models/distilbert_llm_router")
    
    # Plot results
    fine_tuner.plot_training_history()
    
    print("Fine-tuning completed!")
