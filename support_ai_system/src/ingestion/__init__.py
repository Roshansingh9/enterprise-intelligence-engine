"""
Ingestion Package
=================
Data ingestion modules for Excel and other formats.
"""

from .excel_ingestion import ExcelIngestion, DataValidator

__all__ = ['ExcelIngestion', 'DataValidator']
