"""
setup.py — Package setup for P11 Dermatology AI Project.
"""
from setuptools import setup, find_packages

setup(
    name="derma_nlp_project",
    version="1.0.0",
    description="Dermatology Image Classifier with Patient Report NLP (P11)",
    author="P11 Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "spacy>=3.6.0",
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "nltk>=3.8.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.8.0",
    ],
    entry_points={
        "console_scripts": [
            "derma-train=src.dl.train:main",
            "derma-eval=src.dl.evaluate:run_evaluation",
            "derma-demo=demo:main",
        ]
    },
)
