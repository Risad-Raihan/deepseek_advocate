"""
Fine-tuning Engine - Phase 3
LoRA-based Parameter-Efficient Fine-tuning for Bengali Legal Domain
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import gc
from datetime import datetime
import warnings

# Import required libraries
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
        Trainer, DataCollatorForLanguageModeling, EarlyStoppingCallback
    )
    from peft import (
        LoraConfig, get_peft_model, TaskType, PeftModel,
        prepare_model_for_int8_training, prepare_model_for_kbit_training
    )
    from datasets import Dataset
    import wandb
    from accelerate import Accelerator
    import bitsandbytes as bnb
except ImportError as e:
    warnings.warn(f"Required libraries not installed: {e}")

class LegalFineTuningEngine:
    """Advanced fine-tuning engine for Bengali legal domain using LoRA"""
    
    def __init__(self, config, training_config=None):
        """
        Initialize fine-tuning engine
        
        Args:
            config: Main configuration object
            training_config: Training-specific configuration
        """
        self.config = config
        self.training_config = training_config
        
        self.setup_logging()
        self.setup_device()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.trainer = None
        self.accelerator = None
        
        # Training state
        self.training_state = {
            'is_initialized': False,
            'is_training': False,
            'current_epoch': 0,
            'total_steps': 0,
            'best_metric': float('inf'),
            'training_start_time': None
        }
        
        # Memory management
        self.memory_stats = {
            'initial_memory': 0,
            'peak_memory': 0,
            'current_memory': 0
        }
        
    def setup_logging(self):
        """Setup logging for fine-tuning engine"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def setup_device(self):
        """Setup device and memory management"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            self.memory_stats['initial_memory'] = torch.cuda.memory_allocated()
        
        # Enable optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
    
    def initialize_model_and_tokenizer(self) -> bool:
        """Initialize base model and tokenizer"""
        try:
            self.logger.info("Initializing model and tokenizer...")
            
            model_config = self.config.MODEL_CONFIG
            model_name = model_config["base_model"]
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=model_config.get("trust_remote_code", True),
                padding_side=model_config.get("tokenizer_padding_side", "right"),
                truncation_side=model_config.get("tokenizer_truncation_side", "right")
            )
            
            # Add special tokens if needed
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Configure tokenizer for Bengali
            self._configure_bengali_tokenizer()
            
            # Initialize model with quantization
            model_kwargs = {
                "trust_remote_code": model_config.get("trust_remote_code", True),
                "torch_dtype": getattr(torch, model_config.get("torch_dtype", "float16")),
                "device_map": model_config.get("device_map", "auto"),
                "low_cpu_mem_usage": model_config.get("low_cpu_mem_usage", True),
            }
            
            # Add quantization settings
            if model_config.get("load_in_4bit", True):
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                })
            elif model_config.get("load_in_8bit", False):
                model_kwargs["load_in_8bit"] = True
            
            # Load model
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to load primary model {model_name}: {e}")
                backup_model = model_config.get("backup_model", "microsoft/DialoGPT-medium")
                self.logger.info(f"Trying backup model: {backup_model}")
                self.model = AutoModelForCausalLM.from_pretrained(backup_model, **model_kwargs)
            
            # Prepare model for training
            if model_config.get("load_in_4bit") or model_config.get("load_in_8bit"):
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=self.training_config.TRAINING_CONFIG.get("gradient_checkpointing", True)
                )
            
            # Resize token embeddings if tokenizer was modified
            if len(self.tokenizer) != self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.logger.info(f"Resized token embeddings to {len(self.tokenizer)}")
            
            self.training_state['is_initialized'] = True
            self.logger.info("Model and tokenizer initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing model and tokenizer: {e}")
            return False
    
    def _configure_bengali_tokenizer(self):
        """Configure tokenizer for Bengali text processing"""
        # Add Bengali-specific tokens if needed
        bengali_tokens = [
            "।", "॥", "৳", "টাকা", "ধারা", "অনুচ্ছেদ", "আইন", "বিধি",
            "আদালত", "বিচারক", "মামলা", "রায়", "আদেশ", "ডিক্রি"
        ]
        
        # Check which tokens are not in vocabulary
        new_tokens = []
        for token in bengali_tokens:
            if token not in self.tokenizer.get_vocab():
                new_tokens.append(token)
        
        if new_tokens:
            self.tokenizer.add_tokens(new_tokens)
            self.logger.info(f"Added {len(new_tokens)} Bengali legal tokens to tokenizer")
    
    def setup_lora_model(self) -> bool:
        """Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning"""
        try:
            if not self.training_state['is_initialized']:
                raise ValueError("Model must be initialized before setting up LoRA")
            
            self.logger.info("Setting up LoRA configuration...")
            
            lora_config = self.config.LORA_CONFIG
            
            # Create LoRA configuration
            peft_config = LoraConfig(
                r=lora_config["r"],
                lora_alpha=lora_config["lora_alpha"],
                lora_dropout=lora_config["lora_dropout"],
                bias=lora_config["bias"],
                task_type=TaskType.CAUSAL_LM,
                target_modules=lora_config["target_modules"],
                inference_mode=lora_config.get("inference_mode", False),
                modules_to_save=lora_config.get("modules_to_save", None),
            )
            
            # Apply LoRA to model
            self.peft_model = get_peft_model(self.model, peft_config)
            
            # Print trainable parameters
            self.peft_model.print_trainable_parameters()
            
            # Log LoRA configuration
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            
            self.logger.info(f"LoRA setup complete:")
            self.logger.info(f"- Trainable parameters: {trainable_params:,}")
            self.logger.info(f"- Total parameters: {total_params:,}")
            self.logger.info(f"- Trainable ratio: {100 * trainable_params / total_params:.2f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up LoRA: {e}")
            return False
    
    def prepare_dataset(self, qa_pairs: List[Dict]) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare training, validation, and test datasets
        
        Args:
            qa_pairs: List of Q&A pairs
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        try:
            self.logger.info(f"Preparing dataset from {len(qa_pairs)} Q&A pairs...")
            
            data_config = self.config.DATA_CONFIG
            
            # Filter and validate data
            filtered_pairs = self._filter_and_validate_data(qa_pairs)
            self.logger.info(f"Filtered to {len(filtered_pairs)} valid Q&A pairs")
            
            # Format data for training
            formatted_data = []
            for pair in filtered_pairs:
                formatted_text = self._format_training_example(pair)
                if formatted_text:
                    # Tokenize and check length
                    tokens = self.tokenizer(
                        formatted_text,
                        truncation=True,
                        max_length=data_config["max_seq_length"],
                        return_length=True
                    )
                    
                    formatted_data.append({
                        'text': formatted_text,
                        'input_ids': tokens['input_ids'],
                        'attention_mask': tokens['attention_mask'],
                        'length': tokens['length'][0],
                        'domain': pair.get('domain', 'general'),
                        'question_type': pair.get('question_type', 'unknown')
                    })
            
            self.logger.info(f"Formatted {len(formatted_data)} training examples")
            
            # Split data
            train_data, val_data, test_data = self._split_dataset(formatted_data)
            
            # Create datasets
            train_dataset = Dataset.from_list(train_data)
            val_dataset = Dataset.from_list(val_data)
            test_dataset = Dataset.from_list(test_data)
            
            # Log dataset statistics
            self._log_dataset_statistics(train_dataset, val_dataset, test_dataset)
            
            return train_dataset, val_dataset, test_dataset
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset: {e}")
            raise
    
    def _filter_and_validate_data(self, qa_pairs: List[Dict]) -> List[Dict]:
        """Filter and validate Q&A pairs"""
        data_config = self.config.DATA_CONFIG
        filtered_pairs = []
        
        for pair in qa_pairs:
            # Check required fields
            if not all(key in pair for key in ['instruction', 'response']):
                continue
            
            instruction = pair['instruction'].strip()
            response = pair['response'].strip()
            
            # Length filters
            if len(instruction.split()) < data_config.get("min_instruction_length", 10):
                continue
            if len(instruction.split()) > data_config.get("max_instruction_length", 500):
                continue
            if len(response.split()) < data_config.get("min_response_length", 20):
                continue
            if len(response.split()) > data_config.get("max_response_length", 1500):
                continue
            
            # Quality filter
            quality_score = pair.get('quality_score', 0)
            if quality_score < data_config.get("quality_threshold", 0.7):
                continue
            
            # Bengali text validation
            if not self._validate_bengali_text(instruction) or not self._validate_bengali_text(response):
                continue
            
            filtered_pairs.append(pair)
        
        return filtered_pairs
    
    def _validate_bengali_text(self, text: str) -> bool:
        """Validate Bengali text quality"""
        # Check for minimum Bengali character ratio
        bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
        total_chars = len([char for char in text if char.isalnum()])
        
        if total_chars == 0:
            return False
        
        bengali_ratio = bengali_chars / total_chars
        return bengali_ratio > 0.6  # At least 60% Bengali characters
    
    def _format_training_example(self, qa_pair: Dict) -> str:
        """Format a Q&A pair for training"""
        data_config = self.config.DATA_CONFIG
        template = data_config["prompt_template"]
        
        return template.format(
            system=qa_pair.get('system', ''),
            instruction=qa_pair.get('instruction', ''),
            context=qa_pair.get('context', ''),
            response=qa_pair.get('response', '')
        )
    
    def _split_dataset(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset into train, validation, and test sets"""
        import random
        
        data_config = self.config.DATA_CONFIG
        
        # Shuffle data
        if data_config.get("shuffle_data", True):
            random.shuffle(data)
        
        # Calculate split sizes
        total_size = len(data)
        train_size = int(total_size * data_config.get("train_split_ratio", 0.8))
        val_size = int(total_size * data_config.get("validation_split_ratio", 0.1))
        
        # Split data
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def _log_dataset_statistics(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
        """Log dataset statistics"""
        self.logger.info("Dataset Statistics:")
        self.logger.info(f"- Training samples: {len(train_dataset)}")
        self.logger.info(f"- Validation samples: {len(val_dataset)}")
        self.logger.info(f"- Test samples: {len(test_dataset)}")
        
        # Domain distribution
        domains = {}
        for dataset, name in [(train_dataset, "train"), (val_dataset, "val"), (test_dataset, "test")]:
            domain_counts = {}
            for example in dataset:
                domain = example.get('domain', 'unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            domains[name] = domain_counts
            
        self.logger.info(f"Domain distribution: {domains}")
    
    def setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset) -> bool:
        """Setup Hugging Face Trainer"""
        try:
            if not self.peft_model:
                raise ValueError("LoRA model must be set up before trainer")
            
            self.logger.info("Setting up trainer...")
            
            # Get training configuration
            training_config = self.training_config.TRAINING_CONFIG
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(self.config.get_output_dir()),
                logging_dir=str(self.config.get_logging_dir()),
                **training_config
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,  # Causal LM, not masked LM
                pad_to_multiple_of=8,  # For efficiency
            )
            
            # Callbacks
            callbacks = []
            if training_config.get("early_stopping_patience"):
                callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=training_config["early_stopping_patience"],
                        early_stopping_threshold=training_config.get("early_stopping_threshold", 0.001)
                    )
                )
            
            # Initialize trainer
            self.trainer = Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=callbacks,
                compute_metrics=self._compute_metrics,
            )
            
            self.logger.info("Trainer setup complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up trainer: {e}")
            return False
    
    def _compute_metrics(self, eval_preds):
        """Compute evaluation metrics"""
        predictions, labels = eval_preds
        
        # Basic perplexity calculation
        import numpy as np
        
        # Flatten predictions and labels
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)
        
        # Filter out padding tokens
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Calculate perplexity
        loss = nn.CrossEntropyLoss()(torch.from_numpy(predictions), torch.from_numpy(labels))
        perplexity = torch.exp(loss).item()
        
        return {
            "perplexity": perplexity,
            "eval_loss": loss.item()
        }
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> Dict[str, Any]:
        """Execute the fine-tuning process"""
        try:
            if not self.trainer:
                raise ValueError("Trainer must be set up before training")
            
            self.logger.info("Starting fine-tuning process...")
            self.training_state['is_training'] = True
            self.training_state['training_start_time'] = datetime.now()
            
            # Initialize monitoring
            self._initialize_monitoring()
            
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Start training
            train_result = self.trainer.train()
            
            # Training completed
            self.training_state['is_training'] = False
            training_duration = datetime.now() - self.training_state['training_start_time']
            
            self.logger.info(f"Training completed in {training_duration}")
            self.logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            
            # Save training results
            training_results = {
                'training_loss': train_result.training_loss,
                'training_duration': str(training_duration),
                'total_steps': self.trainer.state.global_step,
                'epochs_completed': self.trainer.state.epoch,
                'best_metric': self.trainer.state.best_metric,
                'log_history': self.trainer.state.log_history
            }
            
            # Save model
            self._save_model()
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            self.training_state['is_training'] = False
            raise
    
    def _initialize_monitoring(self):
        """Initialize training monitoring"""
        monitoring_config = self.config.MONITORING_CONFIG
        
        # Initialize Weights & Biases if enabled
        if monitoring_config.get("use_wandb", False):
            try:
                wandb.init(
                    project=monitoring_config.get("wandb_project", "bengali-legal-advocate"),
                    name=monitoring_config.get("wandb_run_name") or self.config.get_training_run_name(),
                    tags=monitoring_config.get("wandb_tags", []),
                    config=self.config.get_full_config()
                )
                self.logger.info("Weights & Biases monitoring initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize W&B: {e}")
    
    def _save_model(self):
        """Save the fine-tuned model"""
        try:
            self.logger.info("Saving fine-tuned model...")
            
            # Save LoRA adapters
            output_dir = self.config.get_output_dir()
            self.peft_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training configuration
            config_path = output_dir / "training_config.json"
            self.config.save_config(config_path)
            
            # Create model card
            self._create_model_card(output_dir)
            
            self.logger.info(f"Model saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    def _create_model_card(self, output_dir: Path):
        """Create a model card with training information"""
        model_card_content = f"""---
language: bn
license: apache-2.0
tags:
- legal
- bengali
- fine-tuned
- lora
- question-answering
---

# Bengali Legal Advocate - Fine-tuned Model

This model is a fine-tuned version of {self.config.MODEL_CONFIG['base_model']} for Bengali legal question-answering.

## Model Details

- **Base Model**: {self.config.MODEL_CONFIG['base_model']}
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Language**: Bengali (বাংলা)
- **Domain**: Legal
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Training Configuration

- **LoRA Rank**: {self.config.LORA_CONFIG['r']}
- **LoRA Alpha**: {self.config.LORA_CONFIG['lora_alpha']}
- **Learning Rate**: {self.config.TRAINING_CONFIG['learning_rate']}
- **Batch Size**: {self.config.TRAINING_CONFIG['per_device_train_batch_size']}
- **Epochs**: {self.config.TRAINING_CONFIG['num_train_epochs']}

## Usage

This model is designed to answer questions about Bangladeshi law in Bengali. It should be used for informational purposes only and does not replace professional legal advice.

## Limitations

- The model is trained on specific legal documents and may not cover all aspects of Bangladeshi law
- Responses should be verified with qualified legal professionals
- The model may occasionally generate inaccurate or incomplete information

## Training Data

The model was fine-tuned on a dataset of Bengali legal Q&A pairs derived from official legal documents including:
- Constitution of Bangladesh
- Family laws and ordinances
- Property laws
- Court procedures

## Disclaimer

This model provides general legal information only and should not be considered as professional legal advice.
"""
        
        model_card_path = output_dir / "README.md"
        with open(model_card_path, 'w', encoding='utf-8') as f:
            f.write(model_card_content)
    
    def evaluate_model(self, test_dataset: Dataset) -> Dict[str, Any]:
        """Evaluate the fine-tuned model"""
        try:
            if not self.trainer:
                raise ValueError("Trainer must be initialized for evaluation")
            
            self.logger.info("Evaluating fine-tuned model...")
            
            # Run evaluation
            eval_results = self.trainer.evaluate(eval_dataset=test_dataset)
            
            self.logger.info(f"Evaluation results: {eval_results}")
            
            return eval_results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def load_trained_model(self, model_path: Path) -> bool:
        """Load a previously trained model"""
        try:
            self.logger.info(f"Loading trained model from {model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load base model
            model_config = self.config.MODEL_CONFIG
            base_model_name = model_config["base_model"]
            
            # Load base model with same configuration used for training
            model_kwargs = {
                "trust_remote_code": model_config.get("trust_remote_code", True),
                "torch_dtype": getattr(torch, model_config.get("torch_dtype", "float16")),
                "device_map": model_config.get("device_map", "auto"),
                "low_cpu_mem_usage": model_config.get("low_cpu_mem_usage", True),
            }
            
            if model_config.get("load_in_4bit", True):
                model_kwargs.update({
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "bnb_4bit_use_double_quant": True,
                })
            
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
            
            # Load LoRA adapters
            self.peft_model = PeftModel.from_pretrained(base_model, model_path)
            
            self.logger.info("Trained model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading trained model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using the fine-tuned model"""
        try:
            if not self.peft_model or not self.tokenizer:
                raise ValueError("Model and tokenizer must be loaded")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.DATA_CONFIG["max_seq_length"] - max_length
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.peft_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            generated_text = full_response[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return "দুঃখিত, উত্তর তৈরি করতে সমস্যা হয়েছে।"
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        stats = {
            'training_state': self.training_state.copy(),
            'memory_stats': self.memory_stats.copy(),
            'model_info': {},
            'hardware_info': {}
        }
        
        # Model information
        if self.peft_model:
            trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.peft_model.parameters())
            
            stats['model_info'] = {
                'trainable_parameters': trainable_params,
                'total_parameters': total_params,
                'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # Rough estimate for float32
            }
        
        # Hardware information
        if torch.cuda.is_available():
            stats['hardware_info'] = {
                'device': str(self.device),
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
                'gpu_memory_cached': torch.cuda.memory_reserved()
            }
        
        return stats
    
    def cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        # Close wandb if active
        try:
            wandb.finish()
        except:
            pass
        
        self.logger.info("Resources cleaned up")