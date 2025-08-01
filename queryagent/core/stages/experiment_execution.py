"""
Stage 4: Experiment Execution

Executes each generated experiment and monitors performance through system.query_log.
Captures detailed performance metrics for comparison with original query.
Records individual experiment improvements for potential combination.
"""

import asyncio
import time
import uuid
import statistics
from typing import Dict, Any, List

from .base import BaseStage, StageOutput, StageResult
from ...tools.clickhouse import QueryResult
from ...utils.dev_logger import get_dev_logger


class ExperimentExecutionStage(BaseStage):
    """Stage 4: Execute Performance Experiments"""
    
    def __init__(self, clickhouse, config: Dict[str, Any]):
        super().__init__(clickhouse, config)
        
        # Execution options from config
        self.execution_config = config.get('experiment_execution', {})
        self.max_execution_time = self.execution_config.get('max_execution_time_seconds', 300)
        self.warmup_runs = self.execution_config.get('warmup_runs', 3)  # Increased from 1
        self.measurement_runs = self.execution_config.get('measurement_runs', 10)  # Increased from 3
        self.abort_on_timeout = self.execution_config.get('abort_on_timeout', True)
        self.drop_caches = self.execution_config.get('drop_caches_between_runs', True)
        self.outlier_threshold_std = self.execution_config.get('outlier_threshold_std', 2.0)  # Remove measurements >2 std devs
        
        # Development logger
        self.logger = get_dev_logger()
    
    async def execute(self, original_query: str, experiments: List[Dict[str, Any]]) -> StageOutput:
        """Execute all experiments and capture performance metrics."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.log_stage_detail("experiment_execution", f"Starting execution of {len(experiments)} experiments")
            
            execution_data = {
                "original_query": original_query,
                "total_experiments": len(experiments),
                "baseline_performance": None,
                "experiment_results": [],
                "individual_improvements": [],  # Track each experiment's improvement
                "execution_summary": {}
            }
            
            # 1. First run the original query to establish baseline
            self.logger.log_stage_detail("experiment_execution", "Establishing baseline performance with original query")
            baseline_result = await self._execute_baseline_query(original_query)
            execution_data["baseline_performance"] = baseline_result
            
            if baseline_result.get("error"):
                self.logger.log_stage_detail("experiment_execution", f"Baseline query failed: {baseline_result['error']}")
                return self._create_error_output("experiment_execution", f"Baseline query failed: {baseline_result['error']}", start_time)
            
            self.logger.log_stage_detail("experiment_execution", f"Baseline established - avg time: {baseline_result['avg_execution_time_ms']:.2f}ms")
            
            # 2. Execute each experiment
            successful_experiments = 0
            failed_experiments = 0
            
            for i, experiment in enumerate(experiments, 1):
                self.logger.log_stage_detail("experiment_execution", f"Executing experiment {i}/{len(experiments)}: {experiment.get('experiment_id', 'unknown')}")
                
                experiment_result = await self._execute_experiment(experiment, baseline_result)
                execution_data["experiment_results"].append(experiment_result)
                
                if experiment_result.get("execution_successful"):
                    successful_experiments += 1
                    improvement = experiment_result.get("performance_improvement_percent", 0)
                    
                    # Track individual improvement with experiment details
                    improvement_record = {
                        "experiment_id": experiment_result.get("experiment_id"),
                        "title": experiment_result.get("experiment_metadata", {}).get("title", "Unknown"),
                        "targeted_inefficiency": experiment_result.get("experiment_metadata", {}).get("targeted_inefficiency", "Unknown"),
                        "performance_improvement_percent": improvement,
                        "time_saved_ms": baseline_result.get("avg_execution_time_ms", 0) - experiment_result.get("avg_execution_time_ms", 0),
                        "baseline_time_ms": baseline_result.get("avg_execution_time_ms", 0),
                        "experiment_time_ms": experiment_result.get("avg_execution_time_ms", 0),
                        "memory_change_bytes": experiment_result.get("avg_memory_usage", 0) - baseline_result.get("avg_memory_usage", 0),
                        "optimization_type": experiment.get("optimization_type", "unknown"),
                        "combinable": experiment.get("combinable", True),  # Whether this can be combined with others
                        "conflicts_with": experiment.get("conflicts_with", [])  # List of experiment IDs this conflicts with
                    }
                    execution_data["individual_improvements"].append(improvement_record)
                    
                    self.logger.log_stage_detail("experiment_execution", 
                        f"Experiment {i} completed - improvement: {improvement:.1f}% ({improvement_record['time_saved_ms']:.2f}ms saved)")
                else:
                    failed_experiments += 1
                    error = experiment_result.get("error", "unknown error")
                    self.logger.log_stage_detail("experiment_execution", f"Experiment {i} failed: {error}")
            
            # 3. Sort improvements by performance gain
            execution_data["individual_improvements"].sort(key=lambda x: x["performance_improvement_percent"], reverse=True)
            
            # 4. Generate execution summary with detailed improvement analysis
            # Categorize failure reasons for debugging
            failure_reasons = {}
            for exp_result in execution_data["experiment_results"]:
                if not exp_result.get("execution_successful", False):
                    error = exp_result.get("error", "unknown error")
                    # Categorize common error patterns
                    if "No modified query provided" in error:
                        category = "invalid_query_generation"
                    elif "Query execution failed" in error:
                        category = "sql_syntax_error"  
                    elif "timed out" in error:
                        category = "query_timeout"
                    elif "Could not retrieve performance metrics" in error:
                        category = "metrics_retrieval_failure"
                    else:
                        category = "unknown_error"
                    
                    failure_reasons[category] = failure_reasons.get(category, 0) + 1
            
            execution_data["execution_summary"] = {
                "total_experiments": len(experiments),
                "successful_experiments": successful_experiments,
                "failed_experiments": failed_experiments,
                "failure_reasons": failure_reasons,
                "baseline_avg_time_ms": baseline_result.get("avg_execution_time_ms", 0),
                "improvements_analysis": {
                    "best_single_improvement_percent": execution_data["individual_improvements"][0]["performance_improvement_percent"] if execution_data["individual_improvements"] else 0,
                    "total_experiments_with_improvement": len([r for r in execution_data["individual_improvements"] if r["performance_improvement_percent"] > 0]),
                    "total_experiments_with_regression": len([r for r in execution_data["individual_improvements"] if r["performance_improvement_percent"] < 0]),
                    "potentially_combinable_improvements": len([r for r in execution_data["individual_improvements"] if r["combinable"] and r["performance_improvement_percent"] > 0]),
                    "total_time_saved_if_all_combined_ms": sum([r["time_saved_ms"] for r in execution_data["individual_improvements"] if r["performance_improvement_percent"] > 0]),
                },
                "top_improvements": execution_data["individual_improvements"][:5]  # Top 5 improvements
            }
            
            # Log detailed improvement summary
            self.logger.log_stage_detail("experiment_execution", 
                f"Execution completed - {successful_experiments}/{len(experiments)} successful")
            for i, improvement in enumerate(execution_data["individual_improvements"][:3], 1):
                self.logger.log_stage_detail("experiment_execution", 
                    f"  #{i}: {improvement['title']} - {improvement['performance_improvement_percent']:.1f}% improvement ({improvement['time_saved_ms']:.2f}ms saved)")
            
            return self._create_success_output("experiment_execution", execution_data, start_time)
            
        except Exception as e:
            self.logger.log_error("EXPERIMENT_EXECUTION", str(e), "Experiment execution stage failed")
            return self._create_error_output("experiment_execution", f"Experiment execution failed: {str(e)}", start_time)
    
    async def _execute_baseline_query(self, original_query: str) -> Dict[str, Any]:
        """Execute the original query to establish baseline performance."""
        try:
            # Use the original query directly without tagging
            query_to_execute = original_query.strip()
            
            # Execute baseline query multiple times for accurate measurement
            execution_times = []
            memory_usages = []
            
            for run in range(self.warmup_runs + self.measurement_runs):
                self.logger.log_stage_detail("experiment_execution", f"Baseline run {run + 1}/{self.warmup_runs + self.measurement_runs}")
                
                # Record start time for timeout checking and direct timing
                run_start = time.time()
                
                # Execute query
                result = await self.clickhouse.run_readonly_query(query_to_execute)
                
                # Calculate direct execution time as fallback
                direct_execution_time_ms = (time.time() - run_start) * 1000
                
                if result.error:
                    return {"error": f"Baseline query failed: {result.error}"}
                
                # Try to get performance metrics from system.query_log
                metrics = await self._get_query_metrics_by_content(query_to_execute)
                
                if metrics and not metrics.error and metrics.data and len(metrics.data) > 0:
                    # Skip warmup runs for measurement - use system metrics
                    if run >= self.warmup_runs:
                        execution_times.append(metrics.data[0].get('query_duration_ms', 0))
                        memory_usages.append(metrics.data[0].get('memory_usage', 0))
                else:
                    # Fallback to direct timing if system.query_log is unavailable
                    if run >= self.warmup_runs:
                        execution_times.append(direct_execution_time_ms)
                        memory_usages.append(0)  # Can't get memory usage without system.query_log
                        
                    # Log that we're using fallback timing
                    if run == self.warmup_runs:  # Log only once
                        self.logger.log_stage_detail("experiment_execution", 
                            "Using direct execution timing as fallback (system.query_log unavailable)")
                
                # Check for timeout
                if time.time() - run_start > self.max_execution_time:
                    if self.abort_on_timeout:
                        return {"error": f"Baseline query timed out after {self.max_execution_time}s"}
                    else:
                        break
            
            if not execution_times:
                return {"error": "Could not retrieve baseline performance metrics - query may not be appearing in system.query_log"}
            
            # Calculate baseline statistics
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
            
            return {
                "query": query_to_execute,
                "runs_measured": len(execution_times),
                "avg_execution_time_ms": avg_time,
                "min_execution_time_ms": min_time,
                "max_execution_time_ms": max_time,
                "avg_memory_usage": avg_memory,
                "execution_times": execution_times,
                "memory_usages": memory_usages
            }
            
        except Exception as e:
            return {"error": f"Failed to execute baseline query: {str(e)}"}
    
    async def _execute_experiment(self, experiment: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single experiment and compare with baseline."""
        try:
            experiment_id = experiment.get('experiment_id', 'unknown')
            modified_query = experiment.get('modified_query', '')
            
            if not modified_query:
                return {
                    "experiment_id": experiment_id,
                    "execution_successful": False,
                    "error": "No modified query provided",
                    "experiment_metadata": {
                        "title": experiment.get('title', 'Untitled'),
                        "hypothesis": experiment.get('hypothesis', ''),
                        "targeted_inefficiency": experiment.get('targeted_inefficiency', ''),
                        "expected_impact": experiment.get('expected_impact', ''),
                        "risk_level": experiment.get('risk_level', '')
                    }
                }
            
            # Use the modified query directly without tagging
            query_to_execute = modified_query.strip()
            
            # Execute experiment query multiple times
            execution_times = []
            memory_usages = []
            
            for run in range(self.warmup_runs + self.measurement_runs):
                self.logger.log_stage_detail("experiment_execution", f"Experiment {experiment_id} run {run + 1}/{self.warmup_runs + self.measurement_runs}")
                
                run_start = time.time()
                
                # Execute query
                result = await self.clickhouse.run_readonly_query(query_to_execute)
                
                # Calculate direct execution time as fallback
                direct_execution_time_ms = (time.time() - run_start) * 1000
                
                if result.error:
                    return {
                        "experiment_id": experiment_id,
                        "execution_successful": False,
                        "error": f"Query execution failed: {result.error}",
                        "experiment_metadata": {
                            "title": experiment.get('title', 'Untitled'),
                            "hypothesis": experiment.get('hypothesis', ''),
                            "targeted_inefficiency": experiment.get('targeted_inefficiency', ''),
                            "expected_impact": experiment.get('expected_impact', ''),
                            "risk_level": experiment.get('risk_level', '')
                        }
                    }
                
                # Try to get performance metrics from system.query_log
                metrics = await self._get_query_metrics_by_content(query_to_execute)
                
                if metrics and not metrics.error and metrics.data and len(metrics.data) > 0:
                    # Skip warmup runs - use system metrics
                    if run >= self.warmup_runs:
                        execution_times.append(metrics.data[0].get('query_duration_ms', 0))
                        memory_usages.append(metrics.data[0].get('memory_usage', 0))
                else:
                    # Fallback to direct timing if system.query_log is unavailable
                    if run >= self.warmup_runs:
                        execution_times.append(direct_execution_time_ms)
                        memory_usages.append(0)  # Can't get memory usage without system.query_log
                
                # Check timeout
                if time.time() - run_start > self.max_execution_time:
                    if self.abort_on_timeout:
                        return {
                            "experiment_id": experiment_id,
                            "execution_successful": False,
                            "error": f"Experiment timed out after {self.max_execution_time}s",
                            "experiment_metadata": {
                                "title": experiment.get('title', 'Untitled'),
                                "hypothesis": experiment.get('hypothesis', ''),
                                "targeted_inefficiency": experiment.get('targeted_inefficiency', ''),
                                "expected_impact": experiment.get('expected_impact', ''),
                                "risk_level": experiment.get('risk_level', '')
                            }
                        }
                    else:
                        break
            
            if not execution_times:
                return {
                    "experiment_id": experiment_id,
                    "execution_successful": False,
                    "error": "Could not retrieve performance metrics - query may not be appearing in system.query_log",
                    "experiment_metadata": {
                        "title": experiment.get('title', 'Untitled'),
                        "hypothesis": experiment.get('hypothesis', ''),
                        "targeted_inefficiency": experiment.get('targeted_inefficiency', ''),
                        "expected_impact": experiment.get('expected_impact', ''),
                        "risk_level": experiment.get('risk_level', '')
                    }
                }
            
            # Calculate experiment statistics
            avg_time = sum(execution_times) / len(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            avg_memory = sum(memory_usages) / len(memory_usages) if memory_usages else 0
            
            # Compare with baseline
            baseline_avg = baseline.get("avg_execution_time_ms", 1)  # Avoid division by zero
            improvement_percent = ((baseline_avg - avg_time) / baseline_avg) * 100
            
            return {
                "experiment_id": experiment_id,
                "execution_successful": True,
                "query": query_to_execute,
                "runs_measured": len(execution_times),
                "avg_execution_time_ms": avg_time,
                "min_execution_time_ms": min_time,
                "max_execution_time_ms": max_time,
                "avg_memory_usage": avg_memory,
                "baseline_avg_time_ms": baseline_avg,
                "performance_improvement_percent": improvement_percent,
                "execution_times": execution_times,
                "memory_usages": memory_usages,
                "experiment_metadata": {
                    "title": experiment.get('title', 'Untitled'),
                    "hypothesis": experiment.get('hypothesis', ''),
                    "targeted_inefficiency": experiment.get('targeted_inefficiency', ''),
                    "expected_impact": experiment.get('expected_impact', ''),
                    "risk_level": experiment.get('risk_level', '')
                }
            }
            
        except Exception as e:
            return {
                "experiment_id": experiment.get('experiment_id', 'unknown'),
                "execution_successful": False,
                "error": f"Experiment execution failed: {str(e)}",
                "experiment_metadata": {
                    "title": experiment.get('title', 'Untitled'),
                    "hypothesis": experiment.get('hypothesis', ''),
                    "targeted_inefficiency": experiment.get('targeted_inefficiency', ''),
                    "expected_impact": experiment.get('expected_impact', ''),
                    "risk_level": experiment.get('risk_level', '')
                }
            }
    
    async def _get_query_metrics_by_content(self, query_content: str) -> QueryResult:
        """Retrieve performance metrics for a query by matching actual query content."""
        try:
            # Wait longer for query_log to be updated (system.query_log can be delayed)
            await asyncio.sleep(1.5)
            
            # First, let's try a simpler approach - find the most recent query that finished
            # This assumes experiments are run sequentially and we want the last completed query
            simple_metrics_query = """
            SELECT 
                query_duration_ms,
                memory_usage,
                read_rows,
                read_bytes,
                written_rows,
                written_bytes,
                result_rows,
                result_bytes,
                type,
                event_time,
                query
            FROM system.query_log 
            WHERE type = 'QueryFinish'
            AND event_time >= now() - INTERVAL 2 MINUTE
            AND query NOT LIKE '%system.query_log%'
            ORDER BY event_time DESC 
            LIMIT 1
            """
            
            result = await self.clickhouse.run_readonly_query(simple_metrics_query)
            
            # Log debug info if no metrics found
            if not result.error and (not result.data or len(result.data) == 0):
                self.logger.log_stage_detail("experiment_execution", 
                    f"No recent QueryFinish entries found in system.query_log")
                
                # Try to find ANY recent queries to debug
                debug_query = """
                SELECT query, type, event_time, exception
                FROM system.query_log 
                WHERE event_time >= now() - INTERVAL 2 MINUTE
                ORDER BY event_time DESC 
                LIMIT 5
                """
                debug_result = await self.clickhouse.run_readonly_query(debug_query)
                if not debug_result.error and debug_result.data:
                    self.logger.log_stage_detail("experiment_execution", 
                        f"Recent log entries found: {len(debug_result.data)}")
                    for entry in debug_result.data[:2]:  # Show first 2 entries
                        self.logger.log_stage_detail("experiment_execution", 
                            f"  {entry.get('type', 'unknown')} at {entry.get('event_time', 'unknown')}: {entry.get('query', 'no query')[:100]}")
                else:
                    # Check if query logging is enabled at all
                    config_check = "SELECT value FROM system.settings WHERE name = 'log_queries'"
                    config_result = await self.clickhouse.run_readonly_query(config_check)
                    if not config_result.error and config_result.data:
                        log_queries_setting = config_result.data[0].get('value', 'unknown')
                        self.logger.log_stage_detail("experiment_execution", 
                            f"log_queries setting: {log_queries_setting}")
                    
                    # Also check if the query_log table exists and has any data
                    table_check = "SELECT count(*) as total_entries FROM system.query_log LIMIT 1"
                    table_result = await self.clickhouse.run_readonly_query(table_check)
                    if not table_result.error and table_result.data:
                        total_entries = table_result.data[0].get('total_entries', 0)
                        self.logger.log_stage_detail("experiment_execution", 
                            f"Total entries in query_log: {total_entries}")
            
            return result
            
        except Exception as e:
            return QueryResult(error=f"Failed to retrieve query metrics: {str(e)}")