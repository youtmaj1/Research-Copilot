"""
Research Paper Collector Module

A comprehensive system for collecting research papers from ArXiv and Google Scholar
with deduplication, PDF downloading, and metadata management.
"""

from .collector import PaperCollector, RetryConfig
from .arxiv_client import ArxivClient, get_popular_categories
from .scholar_client import ScholarClient, is_available as scholar_available
from .database import PaperDatabase

__version__ = "1.0.0"
__author__ = "Research Copilot Team"

__all__ = [
    'PaperCollector',
    'RetryConfig', 
    'ArxivClient',
    'ScholarClient',
    'PaperDatabase',
    'get_popular_categories',
    'scholar_available'
]
