#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: christos
"""

from setuptools import setup, find_packages

setup(
    name="FastApiCloudPlatform",
    version="0.1",
    packages=find_packages(include=["src*"]),  # include all packages under src
    author="Christos Zacharopoulos",
    author_email="christonik[at]gmail.com",
    url="https://github.com/Christonikos/machine_learning_cloud_platform",
    description="Fast API Cloud Platform",
    long_description='"Efficiently deploy, manage, and scale ML models on a cloud platform using FastAPI. Streamline model integration with RESTful APIs, and leverage best practices for containerisation, CI/CD pipelines, and versioning."',
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "pytest",
        "requests",
        "fastapi==0.63.0",
        "uvicorn",
        "gunicorn",
        "spyder-kernels==2.4",
        "qt",
        "matplotlib",
        "ydata-profiling",
        "pandas_dq",
        "reportlab",
        "PyPDF2",
        "pdfreader",
        "PyMuPDF",
        "pdfrw",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
