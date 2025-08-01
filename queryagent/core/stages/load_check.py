"""
Stage 1: Load Check

Verify the ClickHouse cluster isn't too busy before starting analysis.
Checks for long-running queries (>3 seconds by default).
"""

import asyncio
from typing import Dict, Any

from .base import BaseStage, StageOutput, StageResult
from ...utils.dev_logger import get_dev_logger


class LoadCheckStage(BaseStage):
    """Stage 1: Cluster Load Check"""
    
    def __init__(self, clickhouse, config: Dict[str, Any]):
        super().__init__(clickhouse, config)
        
        # Load thresholds from config or use defaults
        self.max_long_running_queries = self.safety_config.get('max_long_running_queries', 3)
        self.long_query_threshold_seconds = self.safety_config.get('long_query_threshold_seconds', 3.0)
        
        # Development logger
        self.logger = get_dev_logger()
    
    async def execute(self) -> StageOutput:
        """Execute the load check stage."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.log_stage_detail("load_check", f"Starting load check with thresholds: max_long_queries={self.max_long_running_queries}, threshold_seconds={self.long_query_threshold_seconds}")
            # Check for long-running queries
            long_queries_result = await self.clickhouse.run_readonly_query(f"""
                SELECT 
                    count() as long_running_queries,
                    max(elapsed) as longest_query_seconds
                FROM system.processes 
                WHERE query != '' 
                AND query NOT LIKE '%system.processes%'
                AND elapsed > {self.long_query_threshold_seconds}
            """)
            
            if long_queries_result.error:
                return self._create_error_output(
                    "load_check", 
                    f"Failed to check long-running queries: {long_queries_result.error}",
                    start_time
                )
            
            data = long_queries_result.data[0] if long_queries_result.data else {}
            long_running_queries = data.get('long_running_queries', 0)
            longest_query_seconds = data.get('longest_query_seconds', 0.0) or 0.0
            
            # Also get total query count for context
            total_queries_result = await self.clickhouse.run_readonly_query("""
                SELECT count() as total_queries
                FROM system.processes 
                WHERE query != '' 
                AND query NOT LIKE '%system.processes%'
            """)
            
            total_queries = 0
            if not total_queries_result.error and total_queries_result.data:
                total_queries = total_queries_result.data[0].get('total_queries', 0)
            
            # Evaluate load conditions
            result_data = {
                "total_queries": total_queries,
                "long_running_queries": long_running_queries,
                "longest_query_seconds": longest_query_seconds,
                "threshold_seconds": self.long_query_threshold_seconds,
                "max_allowed_long_queries": self.max_long_running_queries,
            }
            
            # Log load check results
            self.logger.log_stage_detail("load_check", f"Load check results: total_queries={total_queries}, long_running={long_running_queries}, longest={longest_query_seconds:.1f}s")
            
            if long_running_queries > self.max_long_running_queries:
                result_data["recommendation"] = f"Cluster is busy with {long_running_queries} long-running queries (>{self.long_query_threshold_seconds}s). Wait and retry later."
                self.logger.log_stage_detail("load_check", f"ABORTING: Cluster too busy ({long_running_queries} > {self.max_long_running_queries} long queries)")
                return self._create_abort_output("load_check", result_data, start_time)
            
            result_data["status"] = f"Cluster load is acceptable - only {long_running_queries} long-running queries"
            self.logger.log_stage_detail("load_check", f"SUCCESS: Cluster load acceptable ({long_running_queries} long queries)")
            return self._create_success_output("load_check", result_data, start_time)
            
        except Exception as e:
            self.logger.log_error("LOAD_CHECK", str(e), "Load check stage failed")
            return self._create_error_output("load_check", f"Load check failed: {str(e)}", start_time)