from setuptools import setup, find_packages

setup(
    name="vetinari",
    version="0.2.0",
    packages=find_packages(),
    description="Comprehensive AI Orchestration System with multi-agent assembly-line execution",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="Vetinari",
    install_requires=[
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "flask>=2.0.0",
        "duckduckgo-search>=4.0.0",
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
            "duckduckgo-search>=4.0.0",
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
