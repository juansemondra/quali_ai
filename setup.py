# setup.py
from setuptools import setup, find_packages

setup(
    name="quali_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "PyPDF2",
        "spacy",
        "nltk",
        "bertopic[all]",
        "sentence-transformers",
        "scikit-learn",
        "transformers",
        "torch",
    ],
    entry_points={
        "console_scripts": [
            # 'quali-ai' será el comando que instalaremos
            "quali-ai = main:main",
        ],
    },
)