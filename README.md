# 🏛️ Bengali Legal Advocate AI System

## Advanced Legal AI using Hybrid RAG + Fine-tuning

A comprehensive Bengali Legal Advocate AI system that combines advanced RAG (Retrieval-Augmented Generation) with domain-specific fine-tuning to provide expert legal advice on Bangladesh law.

---

## 🎯 Overview

The Bengali Legal Advocate is designed to process Bengali legal documents and provide expert-level legal advice with proper citations. The system specializes in:

- **Constitutional Law** - বাংলাদেশের সংবিধান
- **Family Law** - পারিবারিক আইন ও তালাক
- **Property Law** - সম্পত্তি আইন ও উত্তরাধিকার
- **Rent Control** - বাড়ি ভাড়া নিয়ন্ত্রণ আইন
- **Court Procedures** - আদালতি প্রক্রিয়া ও কার্যপদ্ধতি

---

## 🏗️ System Architecture

```
legal_advocate/
├── data/                    # Bengali legal PDFs
├── vector_db/              # FAISS vector storage
├── models/                 # Fine-tuned model storage
├── training_data/          # Generated legal Q&A pairs
├── src/
│   ├── document_processor.py    # PDF processing & entity extraction
│   ├── bengali_processor.py     # Bengali text processing
│   ├── vector_store.py          # Multi-level FAISS indexing
│   ├── legal_rag.py            # RAG system implementation
│   └── hybrid_advocate.py      # Main system orchestrator
├── configs/
│   └── model_config.py         # System configuration
├── requirements.txt            # Dependencies
├── setup.py                   # Installation script
└── main.py                    # Main execution script
```

---

## 🚀 Quick Start

### Phase 1: Document Processing & Vector Store

1. **Install Dependencies**
   ```bash
   cd legal_advocate
   pip install -r requirements.txt
   ```

2. **Run Phase 1 Processing**
   ```bash
   python main.py
   ```

3. **Verify Installation**
   ```bash
   python -c "from src.document_processor import LegalDocumentProcessor; print('✅ Installation successful')"
   ```

---

## 📊 Implementation Phases

### ✅ Phase 1: Document Processing & Vector Store (COMPLETED)
- [x] Multi-format PDF text extraction (pdfplumber, PyMuPDF, PyPDF2)
- [x] Bengali legal entity recognition
- [x] Document type classification
- [x] Multi-level FAISS indexing (document, section, paragraph, entity)
- [x] Hybrid search (dense embeddings + BM25)
- [x] Legal text structuring and normalization

### ✅ Phase 2: Legal RAG System with LM Studio Integration (COMPLETED)
- [x] **Intelligent Legal Query Processing** - Advanced Bengali legal query understanding with domain classification
- [x] **Multi-Strategy Retrieval** - 4 specialized retrieval strategies (Direct, Conceptual, Multi-hop, Precedence)
- [x] **Hierarchical Context Building** - Intelligent legal context construction with legal hierarchy
- [x] **Cross-reference Identification** - Automatic legal cross-reference discovery and linking
- [x] **LM Studio + DeepSeek Integration** - Local AI for private, free, and fast response generation
- [x] **Bengali Legal Response Generation** - Professional legal advice in Bengali with proper citations

### 🔄 Phase 3: Fine-tuning Pipeline
- [ ] Generate legal Q&A training data from documents
- [ ] PEFT/LoRA configuration for legal domain
- [ ] Training loop with legal expertise evaluation
- [ ] Model checkpointing and validation

### 🔄 Phase 4: Hybrid Integration
- [ ] Load fine-tuned legal expert model
- [ ] Integrate RAG with fine-tuned model
- [ ] Response enhancement and formatting
- [ ] Evaluation framework

---

## 🔧 Technical Specifications

### Core Components

**Document Processing Engine**
- Multi-method PDF extraction for reliability
- Bengali legal entity recognition
- Hierarchical text structuring
- OCR error correction for Bengali text

**Vector Database**
- Multi-level FAISS indexing
- Hybrid search (dense + sparse)
- Legal relevance scoring
- Cross-reference mapping

**Bengali Language Processing**
- Legal terminology normalization
- Query intent classification
- Response formatting with proper citations
- Unicode and OCR error handling

### Model Configuration

```python
# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Base Language Model
BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-base"  # Preferred
FALLBACK_MODEL = "microsoft/DialoGPT-medium"

# Fine-tuning Configuration
LORA_CONFIG = {
    "r": 32,
    "alpha": 64,
    "dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
}
```

---

## 📈 Performance Metrics

### Phase 1 Results
- **Documents Processed**: 7 Bengali legal PDFs
- **Text Extraction Success**: 100% (multi-method approach)
- **Entity Recognition**: Sections, laws, legal terms, case references
- **Vector Index Size**: 4-level hierarchical structure
- **Search Performance**: Sub-second hybrid retrieval

### Legal Accuracy Targets
- **Citation Accuracy**: >95% correct legal references
- **Semantic Understanding**: >90% query intent classification
- **Response Quality**: Professional legal advice format
- **Bengali Fluency**: Natural Bengali legal language

---

## 🎯 Key Features

### 📚 Document Intelligence
- **Multi-format Support**: PDF, text, scanned documents
- **Legal Entity Extraction**: Automatic identification of sections, laws, cases
- **Document Classification**: Constitutional, family, property, criminal law
- **Hierarchical Structuring**: Sections → Paragraphs → Entities

### 🔍 Advanced Search
- **Hybrid Retrieval**: Dense embeddings + BM25 ranking
- **Multi-level Search**: Document, section, paragraph, entity levels
- **Legal Relevance Scoring**: Domain-specific ranking algorithms
- **Cross-reference Discovery**: Related legal provisions

### 💬 Bengali Language Support
- **Native Bengali Processing**: Full Unicode support
- **Legal Terminology**: Specialized Bengali legal vocabulary
- **Query Understanding**: Intent classification and entity extraction
- **Response Formatting**: Professional legal citation format

---

## 🛠️ Usage Examples

### Document Processing
```python
from src.document_processor import LegalDocumentProcessor

processor = LegalDocumentProcessor(data_dir="data")
processed_docs = processor.process_legal_pdfs()
print(f"Processed {processed_docs['total_processed']} documents")
```

### Vector Search
```python
from src.vector_store import LegalVectorStore

vector_store = LegalVectorStore()
results = vector_store.hybrid_search(
    query="তালাকের আইনি প্রক্রিয়া কী?",
    level="paragraph",
    top_k=5
)
```

### Bengali Text Processing
```python
from src.bengali_processor import BengaliLegalProcessor

processor = BengaliLegalProcessor()
entities = processor.extract_legal_entities(bengali_text)
intent = processor.extract_legal_intent(user_query)
```

### Phase 2: Complete Legal RAG System
```python
# Run Phase 2 with LM Studio integration
python main_phase2.py

# Or test individual components
python test_phase2.py
```

### Interactive Legal Consultation
```python
from src.legal_rag import LegalRAGEngine
from src.response_generator import BengaliLegalResponseGenerator

# Initialize system
rag_engine = LegalRAGEngine(vector_store, bengali_processor, query_processor)
response_generator = BengaliLegalResponseGenerator()

# Process legal query
query = "তালাকের পর খোরপোশের নিয়ম কি?"
rag_output = rag_engine.process_legal_query(query)
response = response_generator.generate_comprehensive_legal_response(rag_output)

print(response['response'])  # Bengali legal advice with citations
```

---

## 🎨 System Capabilities

### Legal Expertise Areas

**Constitutional Law (সাংবিধানিক আইন)**
- Fundamental rights analysis
- Constitutional interpretation
- Government structure and powers
- Judicial review principles

**Family Law (পারিবারিক আইন)**
- Marriage and divorce procedures
- Child custody and maintenance
- Inheritance and succession
- Muslim family law ordinance

**Property Law (সম্পত্তি আইন)**
- Land ownership and transfer
- Property registration procedures
- Inheritance and succession rights
- Tenancy and rental agreements

**Procedural Law (প্রক্রিয়াগত আইন)**
- Court procedures and filing
- Legal notice requirements
- Appeal and revision processes
- Evidence and documentation

---

## 🔬 Evaluation Framework

### Automated Metrics
- **BLEU Score**: Translation quality for legal responses
- **ROUGE Score**: Summary quality and coverage
- **Semantic Similarity**: Query-response relevance
- **Citation Accuracy**: Correct legal reference formatting

### Human Evaluation
- **Legal Expert Review**: Professional accuracy assessment
- **User Experience Testing**: Practical usability evaluation
- **Cultural Appropriateness**: Bengali legal context validation
- **Ethical Compliance**: Responsible AI practices

---

## 🚦 System Requirements

### Hardware Requirements
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: RTX 2050 or better (for training)
- **Storage**: 50GB for models and data
- **CPU**: Multi-core processor (8+ cores recommended)

### Software Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.7+ (for GPU acceleration)
- **Operating System**: Windows 10/11, Linux, macOS

---

## 📝 Training Data Format

### Legal Q&A Format
```json
{
  "system": "আপনি একজন দক্ষ বাংলাদেশী আইনজীবী।",
  "instruction": "তালাকের জন্য আইনি প্রক্রিয়া কী?",
  "context": "[RETRIEVED_LEGAL_SECTIONS]",
  "response": "বাংলাদেশের মুসলিম পারিবারিক আইন অধ্যাদেশ ১৯৬১ অনুযায়ী..."
}
```

---

## 🤝 Contributing

We welcome contributions to improve the Bengali Legal Advocate system:

1. **Code Contributions**: Bug fixes, feature improvements
2. **Legal Content**: Additional legal documents and cases
3. **Testing**: Bengali language testing and validation
4. **Documentation**: Improved documentation and examples

---

## ⚖️ Legal Disclaimer

This system provides general legal information and should not be considered as professional legal advice. For specific legal matters, please consult with qualified legal professionals.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Bengali NLP Community**: For language processing resources
- **Legal AI Research**: For advanced legal reasoning techniques
- **Open Source Libraries**: Transformers, FAISS, LangChain, and others
- **Bangladesh Legal System**: For providing comprehensive legal documentation

---

## 📞 Support

For technical support or questions:
- **Issues**: GitHub Issues
- **Documentation**: [Project Wiki](https://github.com/legal-ai/bengali-legal-advocate/wiki)
- **Community**: [Discussions](https://github.com/legal-ai/bengali-legal-advocate/discussions)

---

**Built with ❤️ for the Bengali legal community** 