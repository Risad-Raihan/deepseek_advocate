"""
Model Configuration for Bengali Legal Advocate
Contains all model settings, paths, and hyperparameters
"""

import os
from pathlib import Path
from typing import Dict, List, Any

class ModelConfig:
    """Configuration class for Bengali Legal Advocate models"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR.parent / "data"
    VECTOR_DB_DIR = BASE_DIR / "vector_db"
    MODELS_DIR = BASE_DIR / "models"
    TRAINING_DATA_DIR = BASE_DIR / "training_data"
    
    # Embedding model configuration
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM = 384
    
    # Base language model configuration
    BASE_MODEL_NAME = "microsoft/DialoGPT-medium"  # Fallback if DeepSeek not available
    DEEPSEEK_MODEL = "deepseek-ai/deepseek-coder-6.7b-base"  # Preferred model
    
    # Model paths
    FINE_TUNED_MODEL_PATH = MODELS_DIR / "bengali_legal_expert"
    TOKENIZER_PATH = MODELS_DIR / "tokenizer"
    
    # Vector database configuration
    VECTOR_DB_CONFIG = {
        "index_type": "faiss",
        "embedding_model": EMBEDDING_MODEL,
        "similarity_metric": "cosine",
        "index_levels": ["document", "section", "paragraph", "entity"],
        "hybrid_search_alpha": 0.7,  # Weight for dense vs BM25 search
        "max_context_length": 2048
    }
    
    # Fine-tuning configuration
    FINE_TUNING_CONFIG = {
        "method": "lora",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": [
            "q_proj", "v_proj", "k_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "save_steps": 500,
        "eval_steps": 500,
        "logging_steps": 100,
        "fp16": True,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 3
    }
    
    # Data processing configuration
    DATA_CONFIG = {
        "max_input_length": 1024,
        "max_target_length": 512,
        "doc_stride": 128,
        "min_paragraph_length": 50,
        "max_context_paragraphs": 5,
        "legal_domains": [
            "constitution", "family_law", "property_law", 
            "criminal_law", "civil_law", "procedural_law"
        ]
    }
    
    # Bengali text processing configuration
    BENGALI_CONFIG = {
        "normalize_unicode": True,
        "fix_ocr_errors": True,
        "standardize_punctuation": True,
        "legal_entity_types": [
            "laws", "sections", "articles", "ordinances",
            "court_names", "legal_terms", "case_references"
        ],
        "citation_patterns": {
            "section": r'ধারা\s*(\d+(?:\([ক-৯]+\))?)',
            "article": r'অনুচ্ছেদ\s*(\d+(?:\([ক-৯]+\))?)',
            "law_year": r'(\d{4})\s*সালের\s*(.+?)\s*আইন',
            "ordinance": r'(\d{4})\s*সালের\s*(.+?)\s*অধ্যাদেশ'
        }
    }
    
    # RAG configuration
    RAG_CONFIG = {
        "retrieval_top_k": 10,
        "context_window": 2048,
        "min_relevance_score": 0.3,
        "max_context_documents": 5,
        "enable_multi_hop": True,
        "cross_reference_depth": 2,
        "legal_hierarchy_weight": 0.2,
        "recency_weight": 0.1
    }
    
    # LM Studio Configuration (Local DeepSeek)
    LM_STUDIO_CONFIG = {
        'base_url': 'http://localhost:1234/v1',
        'model_name': 'deepseek',  # Will be detected automatically from LM Studio
        'api_timeout': 120,
        'temperature': 0.3,  # Low temperature for legal accuracy
        'top_p': 0.9,
        'max_tokens': 2048,
        'stream': False
    }
    
    # Response Generation Configuration
    RESPONSE_CONFIG = {
        'max_response_length': 1500,
        'include_citations': True,
        'include_legal_disclaimer': True,
        'response_language': 'bengali',
        'fallback_to_template': True,
        'quality_threshold': 0.7
    }
    
    # Phase 2 Specific Configurations
    PHASE2_CONFIG = {
        'retrieval_strategies': {
            'direct_legal_retrieval': {
                'enabled': True,
                'top_k': 8,
                'alpha': 0.8
            },
            'conceptual_retrieval': {
                'enabled': True,
                'top_k': 10,
                'alpha': 0.7
            },
            'multi_hop_retrieval': {
                'enabled': True,
                'top_k': 10,
                'max_hops': 3
            },
            'precedence_retrieval': {
                'enabled': True,
                'top_k': 8,
                'procedural_boost': 1.2
            }
        },
        
        'context_building': {
            'max_context_length': 2048,
            'hierarchy_weighting': True,
            'cross_reference_linking': True,
            'diversity_threshold': 0.7
        },
        
        'query_processing': {
            'complexity_analysis': True,
            'multi_part_detection': True,
            'entity_expansion': True,
            'intent_classification': True
        }
    }
    
    # Legal Domain Configurations
    LEGAL_DOMAIN_CONFIG = {
        'family_law': {
            'keywords': ['তালাক', 'বিবাহ', 'খোরপোশ', 'দেনমোহর', 'পারিবারিক', 'উত্তরাধিকার'],
            'boost_factor': 1.2,
            'specific_retrievals': ['direct_legal_retrieval', 'precedence_retrieval']
        },
        'property_law': {
            'keywords': ['সম্পত্তি', 'জমি', 'দলিল', 'মালিকানা', 'রেজিস্ট্রেশন'],
            'boost_factor': 1.15,
            'specific_retrievals': ['direct_legal_retrieval', 'multi_hop_retrieval']
        },
        'constitutional_law': {
            'keywords': ['সংবিধান', 'মৌলিক অধিকার', 'নাগরিক', 'রাষ্ট্র'],
            'boost_factor': 1.3,
            'specific_retrievals': ['direct_legal_retrieval', 'conceptual_retrieval']
        },
        'rent_control': {
            'keywords': ['ভাড়া', 'ইজারা', 'বাড়িওয়ালা', 'ভাড়াটিয়া'],
            'boost_factor': 1.1,
            'specific_retrievals': ['direct_legal_retrieval', 'precedence_retrieval']
        },
        'court_procedure': {
            'keywords': ['আদালত', 'মামলা', 'বিচার', 'প্রক্রিয়া', 'আপিল'],
            'boost_factor': 1.25,
            'specific_retrievals': ['precedence_retrieval', 'multi_hop_retrieval']
        }
    }
    
    # Testing Configuration for Phase 2
    TESTING_CONFIG = {
        'test_queries': [
            {
                'query': 'তালাকের পর খোরপোশের নিয়ম কি?',
                'expected_domain': 'family_law',
                'expected_complexity': 'medium'
            },
            {
                'query': 'সংবিধানের কোন অনুচ্ছেদে ধর্মের স্বাধীনতার কথা বলা হয়েছে?',
                'expected_domain': 'constitutional_law',
                'expected_complexity': 'low'
            },
            {
                'query': 'জমি কিনতে কি কি কাগজপত্র লাগে এবং রেজিস্ট্রেশন প্রক্রিয়া কি?',
                'expected_domain': 'property_law',
                'expected_complexity': 'high'
            }
        ],
        'performance_metrics': {
            'response_time_threshold': 30,  # seconds
            'accuracy_threshold': 0.8,
            'completeness_threshold': 0.7
        }
    }
    
    # System prompts for different legal domains
    SYSTEM_PROMPTS = {
        "general": "আপনি একজন দক্ষ বাংলাদেশী আইনজীবী। বাংলাদেশের আইন সম্পর্কে আপনার গভীর জ্ঞান রয়েছে এবং আপনি সঠিক আইনি পরামর্শ প্রদান করেন।",
        
        "family_law": "আপনি একজন পারিবারিক আইনের বিশেষজ্ঞ আইনজীবী। বাংলাদেশের মুসলিম পারিবারিক আইন, তালাক, খোরপোশ, এবং পারিবারিক বিষয়ে আপনার বিশেষত্ব রয়েছে।",
        
        "property_law": "আপনি একজন সম্পত্তি আইনের বিশেষজ্ঞ। জমি, বাড়ি, উত্তরাধিকার, এবং সম্পত্তি সংক্রান্ত আইনি বিষয়ে আপনার গভীর জ্ঞান রয়েছে।",
        
        "constitutional_law": "আপনি একজন সাংবিধানিক আইনের বিশেষজ্ঞ। বাংলাদেশের সংবিধান, মৌলিক অধিকার, এবং সাংবিধানিক বিষয়ে আপনার বিশেষত্ব রয়েছে।",
        
        "procedural_law": "আপনি একজন আদালতি প্রক্রিয়া ও পদ্ধতির বিশেষজ্ঞ। মামলা দায়ের, আদালতের নিয়ম, এবং আইনি প্রক্রিয়া সম্পর্কে আপনার গভীর জ্ঞান রয়েছে।"
    }
    
    # Evaluation configuration
    EVALUATION_CONFIG = {
        "metrics": ["bleu", "rouge", "legal_accuracy", "citation_accuracy"],
        "test_set_size": 0.1,
        "validation_set_size": 0.1,
        "cross_validation_folds": 5,
        "human_evaluation_samples": 100,
        "legal_expert_review": True
    }
    
    # API configuration
    API_CONFIG = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 4,
        "timeout": 120,
        "max_request_size": 10 * 1024 * 1024,  # 10MB
        "rate_limit": "100/hour",
        "cors_origins": ["*"],
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }
    
    # Logging configuration
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            }
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "standard",
                "class": "logging.StreamHandler"
            },
            "file": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": "legal_advocate.log",
                "mode": "a"
            }
        },
        "loggers": {
            "": {
                "handlers": ["default", "file"],
                "level": "INFO",
                "propagate": False
            }
        }
    }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.VECTOR_DB_DIR,
            cls.MODELS_DIR,
            cls.TRAINING_DATA_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_type: str) -> Path:
        """Get path for specific model type"""
        model_paths = {
            "base": cls.FINE_TUNED_MODEL_PATH,
            "tokenizer": cls.TOKENIZER_PATH,
            "embeddings": cls.VECTOR_DB_DIR / "embeddings.faiss",
            "bm25": cls.VECTOR_DB_DIR / "bm25_indexes.pkl"
        }
        return model_paths.get(model_type, cls.MODELS_DIR)
    
    @classmethod
    def get_prompt_template(cls, domain: str = "general") -> str:
        """Get system prompt for specific legal domain"""
        return cls.SYSTEM_PROMPTS.get(domain, cls.SYSTEM_PROMPTS["general"])
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check if required directories exist
        required_dirs = [cls.DATA_DIR, cls.VECTOR_DB_DIR, cls.MODELS_DIR]
        for directory in required_dirs:
            if not directory.exists():
                validation_results["warnings"].append(f"Directory {directory} does not exist")
        
        # Validate model parameters
        if cls.FINE_TUNING_CONFIG["lora_r"] <= 0:
            validation_results["errors"].append("LoRA rank must be positive")
            validation_results["valid"] = False
        
        if cls.TRAINING_CONFIG["learning_rate"] <= 0:
            validation_results["errors"].append("Learning rate must be positive")
            validation_results["valid"] = False
        
        # Check embedding dimension
        if cls.EMBEDDING_DIM <= 0:
            validation_results["errors"].append("Embedding dimension must be positive")
            validation_results["valid"] = False
        
        return validation_results 