"""
Setup configuration for FinancialKG package
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="financial_kg",
    version="0.1.0",
    author="FinancialKG Team",
    description="Multi-Modal Financial Knowledge Graph Construction using Google Gemini",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "examples", "scripts"]),
    python_requires=">=3.9",
    install_requires=[
        "google-generativeai>=0.3.0",
        "langchain>=0.1.0",
        "neo4j>=5.0.0",
        "pandas>=2.0.0",
        "pydantic>=2.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "datasets>=2.0.0",
        "huggingface-hub>=0.20.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.2.0",
        "python-dateutil>=2.8.0",
        "pytz>=2022.1",
        "tqdm>=4.64.0",
        "rich>=13.0.0",
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
        ],
        'viz': [
            'plotly>=5.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.12.0',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
