# Bengali Legal Advocate AI - Phase 3: Fine-tuning Pipeline

This document provides comprehensive documentation for Phase 3 of the Bengali Legal Advocate AI system, which implements a complete fine-tuning pipeline with Q&A generation, LoRA training, and model evaluation.

## 🎯 Overview

Phase 3 implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) to create specialized Bengali legal models. The pipeline includes:

- **Legal Q&A Generation**: Generates high-quality Bengali legal Q&A pairs from processed documents
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning using 4-bit quantization and LoRA
- **Comprehensive Evaluation**: Multi-metric evaluation system for Bengali legal domain
- **Complete Pipeline**: End-to-end automation with monitoring and reporting

## 🚀 Quick Start

### 1. Installation and Setup

Ensure Phase 1 and Phase 2 are completed, then install additional Phase 3 dependencies:

```bash
pip install -r requirements.txt
```

### 2. Quick Test

Run the quick test to verify all components are working:

```bash
python test_phase3_quick.py
```

### 3. Full Pipeline Execution

Run the complete Phase 3 pipeline:

```bash
# Development mode (recommended for first run)
python main_phase3.py --preset development

# Production mode (full training)
python main_phase3.py --preset production

# Quick test mode (minimal resources)
python main_phase3.py --preset quick_test
```

## 📁 Component Overview

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| Q&A Generator | `src/qa_generator.py` | Generate Bengali legal Q&A pairs |
| Fine-tuning Engine | `src/fine_tuning_engine.py` | LoRA-based model fine-tuning |
| Model Evaluator | `src/model_evaluator.py` | Comprehensive model evaluation |
| Training Config | `configs/training_config.py` | All training configurations |

### Execution Scripts

| Script | Purpose |
|--------|---------|
| `main_phase3.py` | Complete Phase 3 pipeline |
| `generate_qa_standalone.py` | Q&A generation only |
| `evaluate_model_standalone.py` | Model evaluation only |
| `test_phase3_quick.py` | Quick component test |

## 🔧 Configuration

### Training Presets

Phase 3 includes four predefined training presets:

```python
# Quick test - minimal resources
python main_phase3.py --preset quick_test

# Development - balanced resources  
python main_phase3.py --preset development

# Production - full training
python main_phase3.py --preset production

# High quality - maximum quality
python main_phase3.py --preset high_quality
```

### Custom Configuration

Create a custom configuration file:

```json
{
  "training_config": {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "learning_rate": 1e-4
  },
  "qa_generation_config": {
    "num_qa_pairs": 5000,
    "quality_threshold": 0.9
  }
}
```

Then run with:

```bash
python main_phase3.py --config-file custom_config.json
```

## 📊 Usage Examples

### 1. Generate Q&A Dataset Only

```bash
# Generate 1000 Q&A pairs
python generate_qa_standalone.py --num-pairs 1000 --output qa_dataset.json

# Generate with higher quality threshold
python generate_qa_standalone.py \
  --num-pairs 5000 \
  --quality-threshold 0.9 \
  --output high_quality_qa.json \
  --format jsonl
```

### 2. Evaluate Existing Model

```bash
# Evaluate a trained model
python evaluate_model_standalone.py \
  --model-path ./models/fine_tuned_model \
  --test-dataset ./qa_dataset.json \
  --output-dir ./evaluation_results
```

### 3. Full Pipeline with Custom Settings

```bash
# Custom output directory
python main_phase3.py \
  --preset development \
  --output-dir ./custom_output

# Q&A generation only
python main_phase3.py --qa-only --preset development
```

## 🎛️ Configuration Details

### Model Configuration

```python
MODEL_CONFIG = {
    "base_model": "DeepSeek-R1-Distill-Qwen-7B",  # Primary model
    "backup_model": "microsoft/DialoGPT-medium",   # Fallback model
    "model_max_length": 2048,
    "load_in_4bit": True,  # 4-bit quantization for memory efficiency
    "torch_dtype": "float16"
}
```

### LoRA Configuration

```python
LORA_CONFIG = {
    "r": 32,                    # LoRA rank
    "lora_alpha": 64,          # LoRA alpha
    "lora_dropout": 0.1,       # Dropout rate
    "target_modules": [         # Target modules for LoRA
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}
```

### Training Configuration

```python
TRAINING_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "fp16": True,
    "gradient_checkpointing": True  # Memory optimization
}
```

## 📈 Evaluation Metrics

Phase 3 includes comprehensive evaluation across multiple dimensions:

### NLP Metrics
- **BLEU Score**: Translation quality metric
- **ROUGE Scores**: Text summarization metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- **Perplexity**: Language model fluency

### Bengali Language Metrics
- **Fluency Score**: Bengali text naturalness
- **Grammar Score**: Grammatical correctness
- **Script Accuracy**: Proper Bengali script usage

### Legal Domain Metrics
- **Legal Accuracy**: Correctness of legal information
- **Citation Accuracy**: Proper legal citation format
- **Terminology Score**: Use of appropriate legal terminology

### Quality Metrics
- **Completeness**: Response completeness
- **Coherence**: Logical consistency
- **Relevance**: Answer relevance to question

## 🏗️ Technical Architecture

### Memory Optimization
- **4-bit Quantization**: Reduces memory usage by ~75%
- **Gradient Checkpointing**: Trades compute for memory
- **LoRA**: Only fine-tunes <1% of parameters
- **Mixed Precision**: FP16 training for efficiency

### Monitoring and Logging
- **Weights & Biases**: Training monitoring
- **TensorBoard**: Local training visualization
- **Comprehensive Logs**: Detailed execution logs
- **Progress Reports**: Step-by-step progress tracking

### Data Pipeline
- **Quality Filtering**: Multi-level quality assessment
- **Bengali Validation**: Language-specific validation
- **Domain Distribution**: Balanced legal domain coverage
- **Deduplication**: Removes duplicate content

## 🚨 Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Use smaller batch size
   python main_phase3.py --preset quick_test
   
   # Or edit config to reduce batch_size
   ```

2. **Import Errors**
   ```bash
   # Run quick test first
   python test_phase3_quick.py
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

3. **Model Loading Issues**
   ```bash
   # Check if base model is accessible
   # The system will fallback to backup model automatically
   ```

4. **Vector Store Errors**
   ```bash
   # Ensure Phase 2 is completed and vector store is built
   python test_phase2_simple.py
   ```

### Resource Requirements

| Configuration | RAM | GPU Memory | Training Time |
|---------------|-----|------------|---------------|
| quick_test | 8GB | 4GB | 30 minutes |
| development | 16GB | 8GB | 2-4 hours |
| production | 32GB | 12GB | 6-12 hours |
| high_quality | 64GB | 16GB | 12-24 hours |

## 📝 Output Files

### Training Outputs
- `fine_tuned_model/` - Final LoRA model
- `checkpoints/` - Training checkpoints
- `logs/` - Training logs
- `training_config.json` - Used configuration

### Evaluation Outputs
- `evaluation_results.json` - Detailed metrics
- `model_comparison.json` - Multi-model comparison
- `sample_predictions.json` - Example predictions

### Q&A Generation Outputs
- `qa_dataset.json` - Generated Q&A pairs
- `qa_dataset.stats.json` - Generation statistics
- `qa_generation.log` - Generation logs

## 🤝 Integration with Previous Phases

Phase 3 builds on Phase 1 and Phase 2:

### Phase 1 Dependencies
- Processed legal documents in `training_data/`
- Document metadata and structure

### Phase 2 Dependencies
- `LegalRAGEngine` for Q&A generation
- `BengaliLegalProcessor` for text processing
- `LegalVectorStore` for document retrieval

### Integration Points
- Q&A generation uses RAG system from Phase 2
- Fine-tuning uses processed documents from Phase 1
- Evaluation integrates Bengali processing capabilities

## 🎯 Success Criteria

Phase 3 aims to achieve:

- ✅ **10,000+ Q&A pairs** with >90% quality score
- ✅ **>15% improvement** in legal reasoning over base model
- ✅ **>95% Bengali fluency** maintained after fine-tuning
- ✅ **<5% memory overhead** with LoRA fine-tuning
- ✅ **Complete automation** from Q&A generation to evaluation

## 📚 Advanced Usage

### Custom Q&A Generation

```python
from src.qa_generator import LegalQAGenerator

# Initialize with custom config
custom_config = {
    'domain_distribution': {
        'family_law': 0.5,     # Focus on family law
        'property_law': 0.3,
        'constitutional_law': 0.2
    }
}

qa_generator = LegalQAGenerator(
    legal_rag=legal_rag,
    bengali_processor=bengali_processor,
    vector_store=vector_store,
    config=custom_config
)
```

### Custom Evaluation Metrics

```python
from src.model_evaluator import LegalModelEvaluator

# Custom evaluation configuration
eval_config = {
    'metrics': ['bleu', 'rouge', 'legal_accuracy'],
    'legal_term_weight': 0.4,  # Emphasize legal terminology
    'citation_weight': 0.3
}

evaluator = LegalModelEvaluator(
    bengali_processor=bengali_processor,
    config=eval_config
)
```

## 🔮 Future Enhancements

Planned improvements for Phase 3:

1. **Multi-GPU Training**: Support for distributed training
2. **Advanced LoRA**: QLoRA and DoRA implementation
3. **Active Learning**: Iterative dataset improvement
4. **Domain Adaptation**: Specialized legal domain models
5. **Evaluation Benchmarks**: Standardized legal evaluation datasets

## 📞 Support

For Phase 3 specific issues:

1. Check logs in `logs/` directory
2. Run `test_phase3_quick.py` for diagnostics
3. Review configuration with `--preset quick_test`
4. Ensure Phase 1 and Phase 2 are working correctly

---

**Note**: Phase 3 requires substantial computational resources. Start with `quick_test` preset to verify functionality before running full training. 