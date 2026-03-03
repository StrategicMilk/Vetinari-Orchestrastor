from setuptools import setup, find_packages

setup(
    name="vetinari",
    version="0.1.0",
    packages=find_packages(),
    description="Local LLM orchestration system",
    author="Vetinari",
    install_requires=[
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "flask>=2.0.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "vetinari=vetinari.cli:main",
        ],
    },
)
