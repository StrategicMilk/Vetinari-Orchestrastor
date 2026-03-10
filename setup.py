import re
from setuptools import setup, find_packages

# Single source of truth for version — read from vetinari/__init__.py
with open("vetinari/__init__.py") as f:
    _version = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)

setup(
    name="vetinari",
    version=_version,
    packages=find_packages(),
    description="Comprehensive AI Orchestration System with multi-agent assembly-line execution, token optimization, and local-cloud hybrid inference",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Vetinari",
    install_requires=[
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "flask>=2.0.0",
        "ddgs>=6.0.0",
        "apscheduler>=3.10.0",
    ],
    extras_require={
        "cloud": [
            "anthropic>=0.20.0",
            "openai>=1.0.0",
            "google-generativeai>=0.4.0",
            "cohere>=5.0.0",
        ],
        "crypto": [
            "cryptography>=41.0.0",
        ],
        "search": [
            "ddgs>=6.0.0",
            "tavily-python>=0.3.0",
        ],
        "ml": [
            "numpy>=1.24.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "vetinari=vetinari.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
