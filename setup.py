"""
Setup configuration for the Research Copilot Paper Collector module.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'docs', 'collector.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="research-copilot-collector",
    version="1.0.0",
    author="Research Copilot Team",
    author_email="team@research-copilot.com",
    description="A comprehensive system for collecting research papers from ArXiv and Google Scholar",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/research-copilot/collector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Indexing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.28.0",
        "python-dateutil>=2.8.0",
        "urllib3>=1.26.0",
    ],
    extras_require={
        "scholar": ["scholarly>=1.7.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "unittest-mock>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "proxy": ["requests[socks]>=2.28.0"],
    },
    entry_points={
        "console_scripts": [
            "collect-papers=collector.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "collector": ["*.py"],
        "tests": ["*.py"],
        "docs": ["*.md"],
    },
    keywords="research papers arxiv scholar scraping academic",
    project_urls={
        "Bug Reports": "https://github.com/research-copilot/collector/issues",
        "Source": "https://github.com/research-copilot/collector",
        "Documentation": "https://research-copilot.readthedocs.io/collector/",
    },
)
