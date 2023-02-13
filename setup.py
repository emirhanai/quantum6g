# setup.py
from setuptools import setup, find_packages

setup(
    name="quantum6g",
    version="0.1.0",
    description="This library is an automatic artificial intelligence library that combines Quantum and 6G technologies.",
    author="Emirhan BULUT",
    author_email="emirhan@isap.solutions",
    url="https://github.com/emirhanai/quantum6g",
    packages=find_packages(),
    install_requires=["pennylane","numpy"],
    keywords='quantum machine-learning, quantum 6g, 6g,quantum,AI, quantum neural networks, qnn, quantum ai, artificial intelligence',
    test_suite='tests',
    )
