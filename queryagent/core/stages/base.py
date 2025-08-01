"""
Base classes and shared components for the 5-stage workflow.
"""

import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class StageResult(Enum):
    """Result status for each stage."""
    SUCCESS = "success"
    FAILURE = "failure" 
    SKIPPED = "skipped"
    ABORT = "abort"


@dataclass
class StageOutput:
    """Output from a workflow stage."""
    stage: str
    result: StageResult
    data: Dict[str, Any]
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


class BaseStage:
    """Base class for all workflow stages."""
    
    def __init__(self, clickhouse, config: Dict[str, Any]):
        self.clickhouse = clickhouse
        self.config = config
        self.safety_config = config.get('safety', {})
    
    def _get_execution_time_ms(self, start_time: float) -> int:
        """Calculate execution time in milliseconds."""
        return int((asyncio.get_event_loop().time() - start_time) * 1000)
    
    def _create_error_output(self, stage_name: str, error: str, start_time: float) -> StageOutput:
        """Create a standardized error output."""
        return StageOutput(
            stage=stage_name,
            result=StageResult.FAILURE,
            data={},
            error=error,
            execution_time_ms=self._get_execution_time_ms(start_time)
        )
    
    def _create_success_output(self, stage_name: str, data: Dict[str, Any], start_time: float) -> StageOutput:
        """Create a standardized success output."""
        return StageOutput(
            stage=stage_name,
            result=StageResult.SUCCESS,
            data=data,
            execution_time_ms=self._get_execution_time_ms(start_time)
        )
    
    def _create_abort_output(self, stage_name: str, data: Dict[str, Any], start_time: float) -> StageOutput:
        """Create a standardized abort output."""
        return StageOutput(
            stage=stage_name,
            result=StageResult.ABORT,
            data=data,
            execution_time_ms=self._get_execution_time_ms(start_time)
        )