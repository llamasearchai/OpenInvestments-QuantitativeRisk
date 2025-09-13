"""
Setup script for OpenInvestments Quantitative Risk Platform.
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read README if it exists
readme = ""
if os.path.exists('README.md'):
    with open('README.md') as f:
        readme = f.read()

setup(
    name="openinvestments-quantitative-risk",
    version="1.0.0",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Integrated model risk, valuation, and leveraged-products analytics platform",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/openinvestments-quantitative-risk",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="quantitative finance risk management derivatives pricing monte carlo var",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "ruff>=0.0.215",
            "mypy>=0.981",
        ],
        "gpu": [
            "cupy>=11.0.0",
            "jax[cuda]>=0.3.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openinvestments=openinvestments.cli:main_cli",
            "oi=openinvestments.cli:main_cli",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
