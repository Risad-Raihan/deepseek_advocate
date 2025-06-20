# 🤝 Contributing to Bengali Legal Advocate AI

Thank you for your interest in contributing to the Bengali Legal Advocate AI project! This document provides guidelines for contributing to this open-source legal AI system.

## 🎯 How to Contribute

### 🐛 **Reporting Bugs**
- Use the [GitHub issue tracker](https://github.com/Risad-Raihan/deepseek_advocate/issues)
- Check if the issue already exists before creating a new one
- Include detailed information about the bug and steps to reproduce

### 🚀 **Suggesting Features**
- Open a feature request on [GitHub issues](https://github.com/Risad-Raihan/deepseek_advocate/issues)
- Describe the feature and its potential impact
- Discuss implementation approaches if you have ideas

### 💻 **Code Contributions**

#### **Priority Areas**
1. 📚 **Legal Document Expansion** - Add more Bangladesh legal texts
2. 🧠 **Model Fine-tuning** - Improve legal domain expertise  
3. 🔍 **Retrieval Enhancement** - Advanced search strategies
4. 🌐 **API Development** - REST API for integration
5. 📱 **Frontend Development** - Web interface for legal queries

#### **Development Setup**
```bash
# Fork and clone the repository
git clone https://github.com/your-username/deepseek_advocate.git
cd deepseek_advocate/legal_advocate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

#### **Code Style**
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where appropriate
- Write unit tests for new features

#### **Commit Guidelines**
```bash
# Use conventional commit format
git commit -m "feat: add new retrieval strategy"
git commit -m "fix: resolve Unicode encoding issue"
git commit -m "docs: update installation instructions"
```

#### **Pull Request Process**
1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Update documentation if needed
4. Ensure all tests pass
5. Submit a pull request with a clear description

## 📚 **Adding Legal Documents**

### **Document Requirements**
- Must be public domain or have appropriate licensing
- Should be relevant to Bangladesh law
- Prefer official government sources
- Include proper metadata (title, year, type)

### **Document Processing**
```python
# Add new documents to data/ directory
# Update document processor if needed
# Test with the processing pipeline
python main.py  # Process new documents
```

## 🧠 **Model Improvements**

### **Fine-tuning Contributions**
- Improve training data quality
- Enhance LoRA configurations
- Add domain-specific evaluations
- Optimize model performance

### **RAG Enhancements**
- Develop new retrieval strategies
- Improve context building algorithms
- Enhance cross-reference detection
- Optimize response generation

## 🔍 **Testing**

### **Running Tests**
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python test_phase1.py
python test_phase2_fixed.py

# Test with coverage
python -m pytest --cov=src tests/
```

### **Adding Tests**
- Write unit tests for new functions
- Include integration tests for major features
- Test with sample Bengali legal queries
- Validate output format and accuracy

## 📖 **Documentation**

### **Documentation Standards**
- Update README.md for major changes
- Add docstrings to new functions
- Include usage examples
- Document configuration options

### **API Documentation**
- Use clear parameter descriptions
- Include return value specifications
- Provide example usage
- Document error handling

## 🌐 **Internationalization**

### **Language Support**
- Maintain Bengali language accuracy
- Handle Unicode properly
- Test with various Bengali fonts
- Ensure proper text rendering

### **Legal Terminology**
- Use accurate Bengali legal terms
- Maintain consistency across documents
- Include glossary updates
- Validate with legal experts

## 🚀 **Release Process**

### **Version Management**
- Follow semantic versioning (SemVer)
- Update version in setup.py
- Create release notes
- Tag releases appropriately

### **Quality Assurance**
- All tests must pass
- Code coverage > 80%
- Documentation updated
- Performance benchmarks met

## 📞 **Getting Help**

### **Communication Channels**
- [GitHub Discussions](https://github.com/Risad-Raihan/deepseek_advocate/discussions) - General questions
- [GitHub Issues](https://github.com/Risad-Raihan/deepseek_advocate/issues) - Bug reports and features
- Email: [your-email@example.com] - Direct contact

### **Code Review**
- All contributions will be reviewed
- Feedback will be provided constructively  
- Multiple iterations may be needed
- Maintainers will help with improvements

## 🏆 **Recognition**

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation
- Special recognition for major contributions

## 📜 **Code of Conduct**

### **Our Standards**
- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain professional communication

### **Unacceptable Behavior**
- Harassment or discrimination
- Offensive language or imagery
- Personal attacks or trolling
- Publishing private information

## 🙏 **Thank You**

Your contributions help make legal information more accessible to the Bengali-speaking community. Every contribution, no matter how small, makes a difference!

---

**Happy Contributing! 🎉** 