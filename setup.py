# setup.py
from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="quantum6g",
    version="1.2.5",
    description="This library is an automatic artificial intelligence library that combines Quantum and 6G technologies.",
    author="Emirhan BULUT",
    author_email="emirhan@isap.solutions",
    url="https://github.com/emirhanai/quantum6g",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=["numpy","tensorflow","pennylane"],
    keywords='quantum machine-learning, quantum 6g, 6g,quantum,AI, quantum neural networks, qnn, quantum ai, artificial intelligence',
    test_suite='tests',
    )
