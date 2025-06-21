"""
Setup script for Bengali Legal Advocate AI System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="bengali-legal-advocate",
    version="1.0.0",
    author="Legal AI Team",
    author_email="legal-ai@example.com",
    description="Advanced Bengali Legal Advocate using Hybrid RAG + Fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/legal-ai/bengali-legal-advocate",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Legal",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: Bengali",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn[standard]>=0.22.0",
            "pydantic>=2.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "bengali-legal-advocate=legal_advocate.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "legal_advocate": [
            "configs/*.py",
            "data/*.pdf",
            "README.md",
        ],
    },
    keywords=[
        "bengali", "legal", "ai", "nlp", "rag", "fine-tuning", 
        "legal-tech", "artificial-intelligence", "bangladesh"
    ],
    project_urls={
        "Bug Reports": "https://github.com/legal-ai/bengali-legal-advocate/issues",
        "Source": "https://github.com/legal-ai/bengali-legal-advocate",
        "Documentation": "https://legal-ai.github.io/bengali-legal-advocate/docs",
    },
) 