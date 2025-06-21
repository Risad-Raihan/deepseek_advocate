"""
Training Configuration for Phase 3 - Bengali Legal Advocate Fine-tuning
Centralized configuration for LoRA fine-tuning and training pipeline
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional

class Phase3TrainingConfig:
    """Configuration class for Phase 3 training pipeline"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    TRAINING_DATA_DIR = BASE_DIR / "training_data"
    LOGS_DIR = BASE_DIR / "logs"
    CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
    
    # Model Configuration
    MODEL_CONFIG = {
        # Primary model for fine-tuning
        "base_model": "DeepSeek-R1-Distill-Qwen-7B",
        "backup_model": "microsoft/DialoGPT-medium",
        "model_max_length": 2048,
        "trust_remote_code": True,
        
        # Tokenizer settings
        "tokenizer_padding_side": "right",
        "tokenizer_truncation_side": "right",
        "add_eos_token": True,
        "add_bos_token": False,
        
        # Model loading settings
        "load_in_8bit": False,
        "load_in_4bit": True,
        "device_map": "auto",
        "torch_dtype": "float16",
        "low_cpu_mem_usage": True,
    }
    
    # LoRA Configuration
    LORA_CONFIG = {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            "lm_head"
        ],
        "inference_mode": False,
        "modules_to_save": ["embed_tokens", "lm_head"],
        "use_rslora": True,  # Use Rank-Stabilized LoRA
        "use_dora": False,   # Disable DoRA for now
    }
    
    # Training Configuration
    TRAINING_CONFIG = {
        # Core training settings
        "learning_rate": 2e-4,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        
        # Batch and gradient settings
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
        
        # Training duration
        "num_train_epochs": 3,
        "max_steps": -1,  # Use epochs instead
        "save_steps": 500,
        "eval_steps": 500,
        "logging_steps": 50,
        
        # Optimization
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        
        # Memory optimization
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
        "group_by_length": True,
        "length_column_name": "length",
        
        # Evaluation and saving
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        
        # Reproducibility
        "seed": 42,
        "data_seed": 42,
        
        # Reporting
        "report_to": ["tensorboard"],
        "run_name": None,  # Will be set dynamically
        "output_dir": None,  # Will be set dynamically
        "logging_dir": None,  # Will be set dynamically
        
        # Early stopping
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
        
        # Mixed precision training
        "fp16_full_eval": False,
        "tf32": True,  # Enable TensorFloat-32 for A100
        
        # Distributed training
        "ddp_backend": "nccl",
        "ddp_find_unused_parameters": False,
        "ddp_bucket_cap_mb": 25,
        
        # Prediction settings
        "prediction_loss_only": False,
        "include_inputs_for_metrics": False,
    }
    
    # Data Configuration
    DATA_CONFIG = {
        "max_seq_length": 2048,
        "response_template": "\n### Response:",
        "instruction_template": "\n### Instruction:",
        "train_split_ratio": 0.8,
        "validation_split_ratio": 0.1,
        "test_split_ratio": 0.1,
        "shuffle_data": True,
        "num_proc": 4,
        
        # Data filtering
        "min_instruction_length": 10,
        "max_instruction_length": 500,
        "min_response_length": 20,
        "max_response_length": 1500,
        
        # Bengali specific settings
        "normalize_unicode": True,
        "fix_bengali_spacing": True,
        "remove_duplicates": True,
        "quality_threshold": 0.7,
        
        # Prompt template
        "prompt_template": """### System:
{system}

### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}""",
        
        "inference_template": """### System:
{system}

### Instruction:
{instruction}

### Context:
{context}

### Response:
""",
    }
    
    # Q&A Generation Configuration
    QA_GENERATION_CONFIG = {
        "num_qa_pairs": 10000,
        "quality_threshold": 0.8,
        "batch_size": 100,
        "max_retries": 3,
        "diversity_threshold": 0.7,
        
        # Domain distribution
        "domain_distribution": {
            "family_law": 0.3,
            "property_law": 0.25,
            "constitutional_law": 0.2,
            "procedural_law": 0.15,
            "general": 0.1
        },
        
        # Question type distribution
        "question_type_distribution": {
            "factual": 0.3,
            "procedural": 0.25,
            "rights_duties": 0.2,
            "consequences": 0.1,
            "comparative": 0.1,
            "case_based": 0.05
        },
        
        # Export settings
        "export_formats": ["json", "jsonl"],
        "export_splits": True,
        "create_backup": True,
    }
    
    # Evaluation Configuration
    EVALUATION_CONFIG = {
        "metrics": [
            "perplexity",
            "bleu",
            "rouge",
            "legal_accuracy",
            "citation_accuracy",
            "bengali_fluency"
        ],
        
        "eval_batch_size": 8,
        "eval_accumulation_steps": 1,
        "eval_max_samples": 1000,
        
        # BLEU settings
        "bleu_smoothing": True,
        "bleu_weights": [0.25, 0.25, 0.25, 0.25],
        
        # ROUGE settings
        "rouge_types": ["rouge1", "rouge2", "rougeL"],
        "rouge_use_stemmer": False,
        
        # Legal-specific evaluation
        "legal_term_weight": 0.3,
        "citation_weight": 0.2,
        "fluency_weight": 0.2,
        "accuracy_weight": 0.3,
        
        # Evaluation frequency
        "eval_during_training": True,
        "eval_at_end": True,
        "save_eval_results": True,
    }
    
    # Monitoring and Logging Configuration
    MONITORING_CONFIG = {
        "use_wandb": True,
        "wandb_project": "bengali-legal-advocate-phase3",
        "wandb_entity": None,  # Set if using team account
        "wandb_run_name": None,  # Will be set dynamically
        "wandb_tags": ["legal", "bengali", "fine-tuning", "lora"],
        
        "use_tensorboard": True,
        "tensorboard_log_dir": "logs/tensorboard",
        
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        
        # Performance monitoring
        "monitor_gpu_usage": True,
        "monitor_memory_usage": True,
        "log_model_parameters": True,
        "log_gradients": False,  # Can be expensive
        
        # Checkpoint monitoring
        "monitor_checkpoints": True,
        "cleanup_old_checkpoints": True,
        "max_checkpoints_to_keep": 5,
    }
    
    # Hardware Configuration
    HARDWARE_CONFIG = {
        "cuda_visible_devices": None,  # Use all available GPUs
        "num_gpus": -1,  # Auto-detect
        "gpu_memory_limit": None,  # No limit
        "cpu_threads": 8,
        
        # Memory management
        "max_memory_usage": 0.9,  # 90% of available memory
        "empty_cache_steps": 100,
        "force_gc_steps": 500,
        
        # Optimization settings
        "use_amp": True,  # Automatic Mixed Precision
        "compile_model": False,  # PyTorch 2.0 compilation (experimental)
        "channels_last": False,  # Memory format optimization
        
        # Distributed settings
        "backend": "nccl",
        "init_method": "env://",
        "world_size": -1,  # Auto-detect
        "rank": -1,  # Auto-detect
    }
    
    # Model Registry Configuration
    MODEL_REGISTRY_CONFIG = {
        "registry_path": MODELS_DIR / "registry",
        "versioning_scheme": "semantic",  # semantic, timestamp, incremental
        "auto_version": True,
        "save_model_card": True,
        "save_training_config": True,
        "save_evaluation_results": True,
        
        # Model metadata
        "model_metadata": {
            "framework": "transformers",
            "task": "text-generation",
            "language": "bengali",
            "domain": "legal",
            "license": "apache-2.0",
            "tags": ["legal", "bengali", "fine-tuned", "lora"],
        },
        
        # Model validation
        "validate_before_save": True,
        "test_inference": True,
        "check_model_size": True,
        "max_model_size_gb": 20,
    }
    
    # Environment Configuration
    ENV_CONFIG = {
        "environment": "development",  # development, staging, production
        "debug_mode": False,
        "verbose_logging": True,
        "save_intermediate_results": True,
        
        # Paths
        "temp_dir": "/tmp/legal_advocate_training",
        "cache_dir": str(BASE_DIR / "cache"),
        "backup_dir": str(BASE_DIR / "backups"),
        
        # Resource limits
        "max_disk_usage_gb": 100,
        "max_training_time_hours": 24,
        "memory_limit_gb": 32,
        
        # Safety settings
        "enable_safety_checks": True,
        "backup_before_training": True,
        "validate_data_integrity": True,
        "check_disk_space": True,
    }
    
    @classmethod
    def get_training_run_name(cls, model_name: str = None) -> str:
        """Generate a unique training run name"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_name or cls.MODEL_CONFIG["base_model"].split("/")[-1]
        return f"legal_advocate_{model_name}_{timestamp}"
    
    @classmethod
    def get_output_dir(cls, run_name: str = None) -> Path:
        """Get output directory for training run"""
        run_name = run_name or cls.get_training_run_name()
        return cls.CHECKPOINTS_DIR / run_name
    
    @classmethod
    def get_logging_dir(cls, run_name: str = None) -> Path:
        """Get logging directory for training run"""
        run_name = run_name or cls.get_training_run_name()
        return cls.LOGS_DIR / run_name
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.MODELS_DIR,
            cls.TRAINING_DATA_DIR,
            cls.LOGS_DIR,
            cls.CHECKPOINTS_DIR,
            cls.MODEL_REGISTRY_CONFIG["registry_path"],
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration settings"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required directories
        try:
            cls.ensure_directories()
        except Exception as e:
            validation_results["errors"].append(f"Failed to create directories: {e}")
            validation_results["valid"] = False
        
        # Validate model configuration
        if not cls.MODEL_CONFIG["base_model"]:
            validation_results["errors"].append("Base model not specified")
            validation_results["valid"] = False
        
        # Validate training configuration
        if cls.TRAINING_CONFIG["per_device_train_batch_size"] < 1:
            validation_results["errors"].append("Invalid batch size")
            validation_results["valid"] = False
        
        if cls.TRAINING_CONFIG["learning_rate"] <= 0:
            validation_results["errors"].append("Invalid learning rate")
            validation_results["valid"] = False
        
        # Validate LoRA configuration
        if cls.LORA_CONFIG["r"] < 1:
            validation_results["errors"].append("Invalid LoRA rank")
            validation_results["valid"] = False
        
        if cls.LORA_CONFIG["lora_alpha"] < 1:
            validation_results["errors"].append("Invalid LoRA alpha")
            validation_results["valid"] = False
        
        # Check hardware requirements
        import torch
        if not torch.cuda.is_available():
            validation_results["warnings"].append("CUDA not available, training will be slow")
        
        # Check disk space
        import shutil
        free_space_gb = shutil.disk_usage(cls.BASE_DIR).free / (1024**3)
        if free_space_gb < cls.ENV_CONFIG["max_disk_usage_gb"]:
            validation_results["warnings"].append(f"Low disk space: {free_space_gb:.1f}GB available")
        
        return validation_results
    
    @classmethod
    def get_full_config(cls) -> Dict[str, Any]:
        """Get complete configuration dictionary"""
        return {
            "model_config": cls.MODEL_CONFIG,
            "lora_config": cls.LORA_CONFIG,
            "training_config": cls.TRAINING_CONFIG,
            "data_config": cls.DATA_CONFIG,
            "qa_generation_config": cls.QA_GENERATION_CONFIG,
            "evaluation_config": cls.EVALUATION_CONFIG,
            "monitoring_config": cls.MONITORING_CONFIG,
            "hardware_config": cls.HARDWARE_CONFIG,
            "model_registry_config": cls.MODEL_REGISTRY_CONFIG,
            "env_config": cls.ENV_CONFIG,
        }
    
    @classmethod
    def override_config(cls, overrides: Dict[str, Any]):
        """Override configuration values"""
        for config_section, values in overrides.items():
            if hasattr(cls, config_section.upper()):
                config_dict = getattr(cls, config_section.upper())
                if isinstance(config_dict, dict):
                    config_dict.update(values)
                else:
                    setattr(cls, config_section.upper(), values)
    
    @classmethod
    def save_config(cls, filepath: Path):
        """Save configuration to file"""
        import json
        config = cls.get_full_config()
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        config = convert_paths(config)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, filepath: Path):
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        cls.override_config(config)

# Bengali Legal Domain Specific Settings
BENGALI_LEGAL_CONFIG = {
    "legal_terms": [
        "আইন", "ধারা", "অনুচ্ছেদ", "বিধি", "নিয়ম", "প্রবিধান",
        "আদালত", "বিচার", "মামলা", "রায়", "আদেশ", "ডিক্রি",
        "অধিকার", "কর্তব্য", "দায়িত্ব", "ক্ষমতা", "স্বাধীনতা",
        "সংবিধান", "মৌলিক অধিকার", "নাগরিক", "রাষ্ট্র"
    ],
    
    "legal_domains": {
        "family_law": {
            "weight": 0.3,
            "keywords": ["তালাক", "বিবাহ", "খোরপোশ", "দেনমোহর", "পারিবারিক"],
            "complexity": "medium"
        },
        "property_law": {
            "weight": 0.25,
            "keywords": ["সম্পত্তি", "জমি", "দলিল", "মালিকানা", "রেজিস্ট্রেশন"],
            "complexity": "high"
        },
        "constitutional_law": {
            "weight": 0.2,
            "keywords": ["সংবিধান", "মৌলিক অধিকার", "নাগরিক", "রাষ্ট্র"],
            "complexity": "high"
        },
        "procedural_law": {
            "weight": 0.15,
            "keywords": ["আদালত", "মামলা", "বিচার", "প্রক্রিয়া", "আপিল"],
            "complexity": "medium"
        },
        "general": {
            "weight": 0.1,
            "keywords": ["আইন", "বিধি", "নিয়ম"],
            "complexity": "low"
        }
    },
    
    "text_processing": {
        "normalize_unicode": True,
        "fix_ocr_errors": True,
        "standardize_punctuation": True,
        "remove_extra_spaces": True,
        "handle_english_numbers": True,
        "preserve_legal_citations": True
    },
    
    "quality_metrics": {
        "legal_term_density": 0.05,  # Minimum 5% legal terms
        "citation_accuracy": 0.8,     # 80% citation accuracy
        "bengali_fluency": 0.85,      # 85% Bengali fluency
        "coherence": 0.75,            # 75% coherence score
        "completeness": 0.8           # 80% completeness
    }
}

# Training Presets
TRAINING_PRESETS = {
    "quick_test": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "eval_steps": 100,
        "save_steps": 100,
        "logging_steps": 25,
        "max_steps": 200,
        "warmup_steps": 20
    },
    
    "development": {
        "num_train_epochs": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "eval_steps": 250,
        "save_steps": 250,
        "logging_steps": 50,
        "warmup_steps": 50
    },
    
    "production": {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "eval_steps": 500,
        "save_steps": 500,
        "logging_steps": 100,
        "warmup_steps": 100
    },
    
    "high_quality": {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "eval_steps": 200,
        "save_steps": 200,
        "logging_steps": 50,
        "warmup_steps": 200,
        "learning_rate": 1e-4,
        "weight_decay": 0.05
    }
}