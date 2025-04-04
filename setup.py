#\!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="sentinel-ai",
    version="0.1.0",
    description="Adaptive transformer implementations with pruning and fine-tuning capabilities",
    author="Sentinel AI Team",
    author_email="example@example.com",
    url="https://github.com/example/sentinel-ai",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "datasets>=1.18.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "sentinel-train=scripts.entry_points.train:main",
            "sentinel-inference=scripts.entry_points.inference:main",
            "sentinel-prune=scripts.entry_points.prune:main",
            "sentinel-benchmark=scripts.entry_points.benchmark:main",
        ],
    },
)
