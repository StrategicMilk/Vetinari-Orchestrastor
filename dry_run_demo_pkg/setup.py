from setuptools import setup, find_packages

setup(
    name="dry_run_demo_pkg",
    version="0.1.0",
    packages=find_packages(),
    description="Demo Python package for Vetinari end-to-end dry-run",
    long_description="Minimal scaffold demonstrating Plan -> Approve -> Execute -> Verify",
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    author="Vetinari",
    author_email="vetinari@example.com",
    url="https://github.com/example/vetinari",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
