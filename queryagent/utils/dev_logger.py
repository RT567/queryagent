"""
Development logging utility for comprehensive visibility during QueryAgent runs.
Logs everything: queries, LLM calls, responses, experiments, etc.
"""

import logging
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path


class DevLogger:
    """Comprehensive development logger for QueryAgent debugging."""
    
    def __init__(self, log_file: str = None):
        self.session_id = f"session_{int(time.time())}"
        
        # Set up log file
        if log_file is None:
            log_file = f"reports/queryagent_dev_{self.session_id}.log"
        
        self.log_file = Path(log_file)
        
        # Configure logger
        self.logger = logging.getLogger('queryagent_dev')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(self.log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed formatter
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Start session
        self.log_session_start()
    
    def log_session_start(self):
        """Log session start information."""
        self.logger.info("=" * 80)
        self.logger.info(f"QUERYAGENT DEVELOPMENT SESSION STARTED")
        self.logger.info(f"Session ID: {self.session_id}")
        self.logger.info(f"Log file: {self.log_file.absolute()}")
        self.logger.info("=" * 80)
    
    def log_query_start(self, original_query: str):
        """Log the start of query analysis."""
        self.logger.info("-" * 60)
        self.logger.info("QUERY ANALYSIS STARTED")
        self.logger.info("-" * 60)
        self.logger.info(f"Original Query:")
        self.logger.info(f"{original_query}")
        self.logger.info("-" * 60)
    
    def log_stage_start(self, stage_name: str, stage_number: int):
        """Log the start of a workflow stage."""
        self.logger.info(f"STAGE {stage_number}: {stage_name.upper()} - STARTED")
    
    def log_stage_end(self, stage_name: str, stage_number: int, success: bool, duration: float = None):
        """Log the end of a workflow stage."""
        status = "SUCCESS" if success else "FAILED"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        self.logger.info(f"STAGE {stage_number}: {stage_name.upper()} - {status}{duration_str}")
    
    def log_stage_detail(self, stage_name: str, detail: str):
        """Log detailed information within a stage."""
        self.logger.debug(f"{stage_name.upper()}: {detail}")
    
    def log_clickhouse_query(self, query: str, result_summary: str = None, error: str = None, full_result_data: Any = None):
        """Log ClickHouse database queries with FULL results."""
        self.logger.debug(f"CLICKHOUSE QUERY:")
        self.logger.debug(f"  Query: {query}")
        if result_summary:
            self.logger.debug(f"  Summary: {result_summary}")
        if full_result_data is not None:
            self.logger.debug(f"  FULL RESULT DATA:")
            if isinstance(full_result_data, list):
                for i, row in enumerate(full_result_data):
                    self.logger.debug(f"    Row {i+1}: {json.dumps(row, default=str)}")
            else:
                self.logger.debug(f"    {json.dumps(full_result_data, default=str)}")
        if error:
            self.logger.error(f"  Error: {error}")
    
    def log_llm_call(self, prompt: str, response: str = None, error: str = None, 
                     model: str = None, attempt: int = None):
        """Log LLM API calls and responses - FULL CONTENT, NO TRUNCATION."""
        attempt_str = f" (Attempt {attempt})" if attempt else ""
        self.logger.info(f"LLM CALL{attempt_str}:")
        if model:
            self.logger.info(f"  Model: {model}")
        
        # Log full prompt - NO TRUNCATION
        self.logger.info(f"  Prompt ({len(prompt)} chars):")
        self.logger.info(f"{prompt}")
        
        if response:
            # Log full response - NO TRUNCATION
            self.logger.info(f"  Response ({len(response)} chars):")
            self.logger.info(f"{response}")
        
        if error:
            self.logger.error(f"  Error: {error}")
    
    def log_experiments(self, experiments: List[Dict[str, Any]]):
        """Log generated experiments in detail."""
        self.logger.info(f"EXPERIMENTS GENERATED: {len(experiments)} experiments")
        
        for i, exp in enumerate(experiments, 1):
            self.logger.info(f"EXPERIMENT {i}:")
            self.logger.info(f"  ID: {exp.get('experiment_id', 'unknown')}")
            self.logger.info(f"  Title: {exp.get('title', 'Untitled')}")
            self.logger.info(f"  Target: {exp.get('targeted_inefficiency', 'Unknown')}")
            self.logger.info(f"  Impact: {exp.get('expected_impact', 'unknown')}")
            self.logger.info(f"  Risk: {exp.get('risk_level', 'unknown')}")
            self.logger.info(f"  Hypothesis: {exp.get('hypothesis', 'No hypothesis')}")
            
            # Log expected metrics
            metrics = exp.get('expected_metrics', {})
            if metrics:
                self.logger.info(f"  Expected Metrics: {json.dumps(metrics, indent=4)}")
            
            # Log modified query
            modified_query = exp.get('modified_query', '')
            if modified_query:
                self.logger.info(f"  Modified Query:")
                self.logger.info(f"    {modified_query}")
            
            self.logger.info(f"  Validation: {exp.get('validation_status', 'unknown')}")
            self.logger.info("")
    
    def log_explain_results(self, explain_type: str, result_data: Any, error: str = None):
        """Log EXPLAIN query results - FULL CONTENT, NO TRUNCATION."""
        if error:
            self.logger.warning(f"EXPLAIN {explain_type}: FAILED - {error}")
        else:
            if explain_type == "AST" and isinstance(result_data, list) and result_data:
                # Special handling for AST - LOG EVERYTHING
                ast_content = result_data[0].get('ast_tree', 'No AST data')
                lines = ast_content.count('\n') + 1
                self.logger.debug(f"EXPLAIN {explain_type}: SUCCESS ({lines} lines)")
                # Log FULL AST content
                self.logger.debug(f"FULL AST CONTENT:\n{ast_content}")
            else:
                # Regular EXPLAIN results - LOG EVERYTHING
                row_count = len(result_data) if isinstance(result_data, list) else 0
                self.logger.debug(f"EXPLAIN {explain_type}: SUCCESS ({row_count} rows)")
                
                # Log ALL rows
                if isinstance(result_data, list) and result_data:
                    for i, row in enumerate(result_data):
                        self.logger.debug(f"  Row {i+1}: {json.dumps(row, default=str)}")
    
    def log_table_schema(self, table_name: str, schema_data: Dict[str, Any], error: str = None):
        """Log table schema analysis with FULL DETAILS."""
        if error:
            self.logger.warning(f"TABLE SCHEMA {table_name}: FAILED - {error}")
        else:
            self.logger.debug(f"TABLE SCHEMA {table_name}: SUCCESS")
            
            # Log FULL table info
            if schema_data.get("table_info"):
                self.logger.debug(f"  FULL TABLE INFO:")
                for i, info in enumerate(schema_data["table_info"]):
                    self.logger.debug(f"    Table Info {i+1}: {json.dumps(info, default=str)}")
            
            # Log ALL column details
            if schema_data.get("column_info"):
                self.logger.debug(f"  FULL COLUMN INFO ({len(schema_data['column_info'])} columns):")
                for i, col in enumerate(schema_data["column_info"]):
                    self.logger.debug(f"    Column {i+1}: {json.dumps(col, default=str)}")
            
            # Log FULL schema details
            if schema_data.get("schema"):
                self.logger.debug(f"  FULL SCHEMA DATA:")
                for i, schema_row in enumerate(schema_data["schema"]):
                    self.logger.debug(f"    Schema Row {i+1}: {json.dumps(schema_row, default=str)}")
            
            # Log complete schema object
            self.logger.debug(f"  COMPLETE SCHEMA OBJECT:")
            self.logger.debug(f"    {json.dumps(schema_data, default=str)}")
    
    def log_error(self, component: str, error: str, details: str = None):
        """Log errors with context."""
        self.logger.error(f"{component}: {error}")
        if details:
            self.logger.error(f"  Details: {details}")
    
    def log_config(self, config_data: Dict[str, Any]):
        """Log configuration at session start."""
        self.logger.info("CONFIGURATION:")
        for key, value in config_data.items():
            # Don't log sensitive data
            if 'key' in key.lower() or 'password' in key.lower():
                self.logger.info(f"  {key}: [REDACTED]")
            else:
                self.logger.info(f"  {key}: {value}")
    
    def log_complete_stage_data(self, stage_name: str, stage_data: Dict[str, Any]):
        """Log complete stage data with full details."""
        self.logger.info(f"COMPLETE {stage_name.upper()} STAGE DATA:")
        self.logger.info(f"{json.dumps(stage_data, default=str, indent=2)}")
    
    def log_session_end(self, success: bool, total_duration: float = None):
        """Log session end."""
        self.logger.info("-" * 60)
        status = "SUCCESS" if success else "FAILED"
        duration_str = f" (Total: {total_duration:.2f}s)" if total_duration else ""
        self.logger.info(f"QUERYAGENT SESSION ENDED: {status}{duration_str}")
        self.logger.info("=" * 80)
    
    def get_log_file_path(self) -> str:
        """Get the current log file path."""
        return str(self.log_file.absolute())


# Global logger instance
_dev_logger: Optional[DevLogger] = None

def get_dev_logger() -> DevLogger:
    """Get or create the global development logger."""
    global _dev_logger
    if _dev_logger is None:
        _dev_logger = DevLogger()
    return _dev_logger

def init_dev_logger(log_file: str = None) -> DevLogger:
    """Initialize a new development logger."""
    global _dev_logger
    _dev_logger = DevLogger(log_file)
    return _dev_logger