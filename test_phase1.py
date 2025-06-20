#!/usr/bin/env python3
"""
Phase 1 Test Script - Bengali Legal Advocate
Quick test of document processing and vector store functionality
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing module imports...")
    
    try:
        from src.document_processor import LegalDocumentProcessor
        print("✅ LegalDocumentProcessor imported successfully")
    except Exception as e:
        print(f"❌ Error importing LegalDocumentProcessor: {e}")
        return False
    
    try:
        from src.bengali_processor import BengaliLegalProcessor
        print("✅ BengaliLegalProcessor imported successfully")
    except Exception as e:
        print(f"❌ Error importing BengaliLegalProcessor: {e}")
        return False
    
    try:
        from src.vector_store import LegalVectorStore
        print("✅ LegalVectorStore imported successfully")
    except Exception as e:
        print(f"❌ Error importing LegalVectorStore: {e}")
        return False
    
    try:
        from configs.model_config import ModelConfig
        print("✅ ModelConfig imported successfully")
    except Exception as e:
        print(f"❌ Error importing ModelConfig: {e}")
        return False
    
    return True

def test_bengali_processor():
    """Test Bengali text processing functionality"""
    print("\n🔤 Testing Bengali text processing...")
    
    try:
        from src.bengali_processor import BengaliLegalProcessor
        
        processor = BengaliLegalProcessor()
        
        # Test text preprocessing
        sample_text = "এই   আইনে   ধারা  ২৫  এ  বলা  হয়েছে  যে"
        processed = processor.preprocess_bengali_legal_text(sample_text)
        print(f"✅ Text preprocessing: '{sample_text}' → '{processed}'")
        
        # Test legal entity extraction
        legal_text = "বাংলাদেশের সংবিধানের ধারা ২৭ অনুযায়ী সকল নাগরিক আইনের দৃষ্টিতে সমান।"
        entities = processor.extract_legal_entities(legal_text)
        print(f"✅ Entity extraction found: {list(entities.keys())}")
        
        # Test query intent
        query = "তালাকের জন্য আইনি প্রক্রিয়া কী?"
        intent = processor.extract_legal_intent(query)
        print(f"✅ Query intent: {intent['intent']} (confidence: {intent['confidence']:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Bengali processor test failed: {e}")
        return False

def test_document_processor():
    """Test document processing functionality"""
    print("\n📄 Testing document processing...")
    
    try:
        from src.document_processor import LegalDocumentProcessor
        
        # Initialize processor
        processor = LegalDocumentProcessor(data_dir="../data")
        
        # Test document type identification
        test_cases = [
            ("বাংলাদেশের সংবিধান.pdf", "mock constitution content"),
            ("তালাক ও খোরপোশ আইন.pdf", "তালাক সংক্রান্ত পারিবারিক"),
            ("বাড়ী ভাড়া নিয়ন্ত্রণ আইন.pdf", "ভাড়া নিয়ন্ত্রণ")
        ]
        
        for filename, content in test_cases:
            doc_type = processor._identify_document_type(filename, content)
            print(f"✅ Document type identification: {filename} → {doc_type}")
        
        # Test entity extraction
        sample_legal_text = """
        ধারা ২৫: সকল নাগরিক আইনের দৃষ্টিতে সমান এবং আইনের সমান আশ্রয় লাভের অধিকারী।
        ১৯৭১ সালের স্বাধীনতার ঘোষণাপত্র আইন অনুযায়ী এই বিধান কার্যকর।
        """
        
        entities = processor.extract_legal_entities(sample_legal_text)
        print(f"✅ Legal entity extraction: {dict(entities)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Document processor test failed: {e}")
        return False

def test_vector_store():
    """Test vector store initialization"""
    print("\n🔍 Testing vector store...")
    
    try:
        from src.vector_store import LegalVectorStore
        
        # Initialize vector store
        vector_store = LegalVectorStore(vector_db_path="test_vector_db")
        
        print("✅ Vector store initialized")
        
        # Test embedding model loading
        test_text = "বাংলাদেশের সংবিধান"
        embedding = vector_store.embedding_model.encode([test_text])
        print(f"✅ Embedding generated: shape {embedding.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def test_config():
    """Test configuration"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from configs.model_config import ModelConfig
        
        # Test config validation
        validation = ModelConfig.validate_config()
        print(f"✅ Config validation: valid={validation['valid']}")
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"⚠️ Warning: {warning}")
        
        # Test directory creation
        ModelConfig.ensure_directories()
        print("✅ Required directories ensured")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏛️ Bengali Legal Advocate - Phase 1 Testing")
    print("=" * 60)
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Suppress info logs during testing
    
    tests = [
        ("Module Imports", test_imports),
        ("Bengali Processing", test_bengali_processor),
        ("Document Processing", test_document_processor),
        ("Vector Store", test_vector_store),
        ("Configuration", test_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}\n")
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Phase 1 is ready to run.")
        print("Next step: Run 'python main.py' to execute full Phase 1 processing")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 