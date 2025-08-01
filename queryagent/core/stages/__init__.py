"""
5-Stage Query Optimization Workflow

This package implements the core 5-stage workflow for ClickHouse query optimization:
1. Load Check - Verify cluster isn't busy before analysis  
2. Static Analysis - Parse query, explain plans, table schemas
3. Experiment Planning - Design tests based on effort level
4. Execution - Run tagged experiments with monitoring  
5. Analysis - Generate ClickHouse-specific recommendations
"""

from .base import StageResult, StageOutput
from .load_check import LoadCheckStage
from .static_analysis import StaticAnalysisStage

__all__ = [
    'StageResult', 
    'StageOutput',
    'LoadCheckStage',
    'StaticAnalysisStage'
]