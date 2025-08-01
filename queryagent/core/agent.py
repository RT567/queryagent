"""Simplified QueryAgent for ClickHouse query optimization."""

import asyncio
import time
from typing import Optional

from .config import Config
from .stages import StageResult, LoadCheckStage, StaticAnalysisStage
from .stages.experiment_planning import ExperimentPlanningStage
from .stages.experiment_execution import ExperimentExecutionStage
from ..llm.providers.anthropic import AnthropicProvider
from ..tools.clickhouse import ClickHouseTools
from ..utils.dev_logger import get_dev_logger


class QueryAgent:
    """Simplified AI agent for ClickHouse query optimization."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = get_dev_logger()
        
        # Log configuration 
        self.logger.log_config(self.config.__dict__)
        
        self.llm = AnthropicProvider(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        self.clickhouse = None
        self.load_check_stage = None
        self.static_analysis_stage = None
        self.experiment_planning_stage = None
        self.experiment_execution_stage = None
    
    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            # Initialize ClickHouse connection
            self.clickhouse = ClickHouseTools(
                host=self.config.clickhouse_host,
                port=self.config.clickhouse_port,
                database=self.config.clickhouse_database
            )
            
            # Initialize individual stages
            self.load_check_stage = LoadCheckStage(self.clickhouse, self.config.__dict__)
            self.static_analysis_stage = StaticAnalysisStage(self.clickhouse, self.config.__dict__)
            self.static_analysis_stage.set_llm(self.llm)
            self.experiment_planning_stage = ExperimentPlanningStage(self.clickhouse, self.config.__dict__)
            self.experiment_planning_stage.set_llm(self.llm)
            self.experiment_execution_stage = ExperimentExecutionStage(self.clickhouse, self.config.__dict__)
            
            # Test connection
            result = await self.clickhouse.run_readonly_query("SELECT 1")
            return not result.error
            
        except Exception as e:
            print(f"Failed to initialize: {e}")
            return False
    
    async def check_load(self) -> bool:
        return True
        

    
    async def optimize_query(self, query: str) -> dict:
        """Analyze and optimize a ClickHouse query using 5-stage workflow."""
        start_time = time.time()
        
        try:
            # Log query start
            self.logger.log_query_start(query)
            
            print("🚀 Starting 5-stage query optimization workflow...")
            
            # Stage 1: Load Check
            print("📊 Stage 1: Checking cluster load...")
            stage_start = time.time()
            self.logger.log_stage_start("load_check", 1)
            
            load_check = await self.load_check_stage.execute()
            
            stage_duration = time.time() - stage_start
            
            if load_check.result == StageResult.ABORT:
                self.logger.log_stage_end("load_check", 1, False, stage_duration)
                self.logger.log_error("LOAD_CHECK", "Cluster too busy - aborting optimization")
                print(f"⚠️ Cluster too busy - aborting optimization")
                return {
                    "success": False,
                    "stage": "load_check",
                    "abort_reason": "cluster_too_busy",
                    "load_data": load_check.data,
                    "original_query": query
                }
            
            if load_check.result == StageResult.FAILURE:
                self.logger.log_stage_end("load_check", 1, False, stage_duration)
                self.logger.log_error("LOAD_CHECK", load_check.error)
                print(f"❌ Load check failed: {load_check.error}")
                return {
                    "success": False,
                    "stage": "load_check", 
                    "error": load_check.error,
                    "original_query": query
                }
            
            self.logger.log_stage_end("load_check", 1, True, stage_duration)
            self.logger.log_complete_stage_data("load_check", load_check.data)
            print(f"✅ Cluster load acceptable - proceeding with optimization")
            print(f"   Total queries: {load_check.data['total_queries']}")
            print(f"   Long-running queries (>{load_check.data['threshold_seconds']}s): {load_check.data['long_running_queries']}")
            if load_check.data['longest_query_seconds'] > 0:
                print(f"   Longest query: {load_check.data['longest_query_seconds']:.1f}s")
            
            # Stage 2: Static Analysis
            print("🔍 Stage 2: Performing static analysis...")
            stage_start = time.time()
            self.logger.log_stage_start("static_analysis", 2)
            
            static_analysis = await self.static_analysis_stage.execute(query)
            stage_duration = time.time() - stage_start
            
            if static_analysis.result == StageResult.FAILURE:
                self.logger.log_stage_end("static_analysis", 2, False, stage_duration)
                self.logger.log_error("STATIC_ANALYSIS", static_analysis.error)
                print(f"❌ Static analysis failed: {static_analysis.error}")
                return {
                    "success": False,
                    "stage": "static_analysis",
                    "error": static_analysis.error,
                    "stages_completed": ["load_check"],
                    "load_check": load_check.data,
                    "original_query": query
                }
            
            self.logger.log_stage_end("static_analysis", 2, True, stage_duration)
            self.logger.log_complete_stage_data("static_analysis", static_analysis.data)
            print("✅ Static analysis completed")
            self._print_static_analysis_results(static_analysis.data)
            
            # Stage 3: Experiment Planning
            print("🧪 Stage 3: Planning performance experiments...")
            stage_start = time.time()
            self.logger.log_stage_start("experiment_planning", 3)
            
            experiment_planning = await self.experiment_planning_stage.execute(query, static_analysis.data)
            stage_duration = time.time() - stage_start
            
            if experiment_planning.result == StageResult.FAILURE:
                self.logger.log_stage_end("experiment_planning", 3, False, stage_duration)
                self.logger.log_error("EXPERIMENT_PLANNING", experiment_planning.error)
                print(f"❌ Experiment planning failed: {experiment_planning.error}")
                return {
                    "success": False,
                    "stage": "experiment_planning",
                    "error": experiment_planning.error,
                    "stages_completed": ["load_check", "static_analysis"],
                    "load_check": load_check.data,
                    "static_analysis": static_analysis.data,
                    "original_query": query
                }
            
            self.logger.log_stage_end("experiment_planning", 3, True, stage_duration)
            self.logger.log_complete_stage_data("experiment_planning", experiment_planning.data)
            
            # Log the generated experiments
            experiments = experiment_planning.data.get("experiments", [])
            if experiments:
                self.logger.log_experiments(experiments)
            
            print("✅ Experiment planning completed")
            self._print_experiment_planning_results(experiment_planning.data)
            
            # Stage 4: Experiment Execution
            print("⚡ Stage 4: Executing performance experiments...")
            stage_start = time.time()
            self.logger.log_stage_start("experiment_execution", 4)
            
            experiments = experiment_planning.data.get("experiments", [])
            if not experiments:
                self.logger.log_stage_end("experiment_execution", 4, False, 0)
                print("⚠️ No experiments to execute - skipping execution stage")
                experiment_execution_data = {
                    "skipped": True,
                    "reason": "no_experiments_generated",
                    "baseline_performance": None,
                    "experiment_results": [],
                    "execution_summary": {}
                }
            else:
                experiment_execution = await self.experiment_execution_stage.execute(query, experiments)
                stage_duration = time.time() - stage_start
                
                if experiment_execution.result == StageResult.FAILURE:
                    self.logger.log_stage_end("experiment_execution", 4, False, stage_duration)
                    self.logger.log_error("EXPERIMENT_EXECUTION", experiment_execution.error)
                    print(f"❌ Experiment execution failed: {experiment_execution.error}")
                    return {
                        "success": False,
                        "stage": "experiment_execution",
                        "error": experiment_execution.error,
                        "stages_completed": ["load_check", "static_analysis", "experiment_planning"],
                        "load_check": load_check.data,
                        "static_analysis": static_analysis.data,
                        "experiment_planning": experiment_planning.data,
                        "original_query": query
                    }
                
                self.logger.log_stage_end("experiment_execution", 4, True, stage_duration)
                self.logger.log_complete_stage_data("experiment_execution", experiment_execution.data)
                experiment_execution_data = experiment_execution.data
                
                print("✅ Experiment execution completed")
                self._print_experiment_execution_results(experiment_execution_data)
            
            # TODO: Implement Stage 5 (Final Analysis)
            print("📋 Stage 5 (Final Analysis) coming soon...")
            
            total_duration = time.time() - start_time
            self.logger.log_session_end(True, total_duration)
            
            return {
                "success": True,
                "stages_completed": ["load_check", "static_analysis", "experiment_planning", "experiment_execution"],
                "load_check": load_check.data,
                "static_analysis": static_analysis.data,
                "experiment_planning": experiment_planning.data,
                "experiment_execution": experiment_execution_data,
                "original_query": query
            }
            
        except Exception as e:
            total_duration = time.time() - start_time
            self.logger.log_error("AGENT", str(e), f"Unexpected error during query optimization")
            self.logger.log_session_end(False, total_duration)
            return {
                "success": False,
                "error": str(e),
                "original_query": query
            }
    
    def _print_static_analysis_results(self, analysis_data: dict) -> None:
        """Print static analysis results to command line."""
        print()
        
        # Print table extraction results
        if "table_extraction" in analysis_data:
            extraction = analysis_data["table_extraction"]
            print("📋 Table Discovery:")
            print(f"   Tables found: {', '.join(extraction.get('final_tables', []))}")
            
            if "llm_method" in extraction:
                hardcoded = set(extraction.get('hardcoded_method', []))
                llm_found = set(extraction.get('llm_method', []))
                if llm_found - hardcoded:
                    print(f"   🤖 LLM discovered additional: {', '.join(llm_found - hardcoded)}")
                if hardcoded - llm_found:
                    print(f"   📝 Regex-only found: {', '.join(hardcoded - llm_found)}")
        
        # Print comprehensive analysis results
        if "comprehensive_analysis" in analysis_data and analysis_data["comprehensive_analysis"]:
            comprehensive = analysis_data["comprehensive_analysis"]
            print()
            print("🔍 Comprehensive Query Analysis:")
            
            # Check each analysis component
            analyses = [
                ("PLAN", comprehensive.plan),
                ("PIPELINE", comprehensive.pipeline), 
                ("ESTIMATE", comprehensive.estimate),
                ("INDEXES", comprehensive.plan_with_indexes),
                ("PROJECTIONS", comprehensive.plan_with_projections),
                ("SYNTAX", comprehensive.syntax),
                ("AST", comprehensive.ast)
            ]
            
            for name, result in analyses:
                if result.error:
                    print(f"   ❌ {name}: {result.error}")
                else:
                    # Show brief data summary to confirm it's actually working
                    data_size = len(result.data) if result.data else 0
                    if name == "AST" and result.data and result.data[0].get("ast_tree"):
                        lines = result.data[0]["ast_tree"].count('\n') + 1
                        print(f"   ✅ {name}: Available ({lines} lines)")
                    else:
                        print(f"   ✅ {name}: Available ({data_size} rows)")
        
        # Print table schema summaries
        if "table_schemas" in analysis_data:
            schemas = analysis_data["table_schemas"]
            print()
            print("🗄️ Table Schemas:")
            
            for table_name, schema_data in schemas.items():
                if "error" in schema_data:
                    print(f"   ❌ {table_name}: {schema_data['error']}")
                else:
                    print(f"   ✅ {table_name}:")
                    
                    # Show table info if available
                    if schema_data.get("table_info") and len(schema_data["table_info"]) > 0:
                        info = schema_data["table_info"][0]
                        print(f"      Engine: {info.get('engine', 'unknown')}")
                        print(f"      Rows: {info.get('total_rows', 'unknown')}")
                        if info.get('order_by'):
                            print(f"      Order By: {info.get('order_by')}")
                    
                    # Show column count
                    if schema_data.get("column_info"):
                        col_count = len(schema_data["column_info"])
                        print(f"      Columns: {col_count}")
        
        # Print analysis summary
        if "analysis_summary" in analysis_data:
            summary = analysis_data["analysis_summary"]
            print()
            print("📊 Analysis Summary:")
            print(f"   Query complexity: {summary.get('query_complexity', 'unknown')}")
            print(f"   Tables analyzed: {summary.get('tables_analyzed', 0)}")
            
            if summary.get('potential_issues'):
                print("   ⚠️ Potential issues:")
                for issue in summary['potential_issues']:
                    print(f"      • {issue}")
        
        print()
    
    def _print_experiment_planning_results(self, planning_data: dict) -> None:
        """Print experiment planning results to command line."""
        experiments = planning_data.get("experiments", [])
        if not experiments:
            print("   ⚠️ No experiments generated")
            return
        
        print(f"   📋 Generated {len(experiments)} experiments:")
        for i, exp in enumerate(experiments[:3], 1):  # Show top 3
            title = exp.get("title", "Untitled")
            risk = exp.get("risk_level", "unknown")
            print(f"      {i}. {title} (risk: {risk})")
        
        if len(experiments) > 3:
            print(f"      ... and {len(experiments) - 3} more")
        print()
    
    def _print_experiment_execution_results(self, execution_data: dict) -> None:
        """Print experiment execution results to command line."""
        if execution_data.get("skipped"):
            print("   ⚠️ Execution skipped - no experiments to run")
            return
        
        summary = execution_data.get("execution_summary", {})
        baseline_time = summary.get("baseline_avg_time_ms", 0)
        successful = summary.get("successful_experiments", 0)
        total = summary.get("total_experiments", 0)
        
        print(f"   📊 Execution Summary:")
        print(f"      Baseline query time: {baseline_time:.2f}ms")
        print(f"      Experiments completed: {successful}/{total}")
        print()
        
        # Show ALL experiment results
        experiment_results = execution_data.get("experiment_results", [])
        if experiment_results:
            print(f"   📋 All Experiment Results:")
            
            for i, exp_result in enumerate(experiment_results, 1):
                exp_id = exp_result.get("experiment_id", "unknown")
                successful = exp_result.get("execution_successful", False)
                
                if successful:
                    # Show successful experiment with performance data
                    exp_time = exp_result.get("avg_execution_time_ms", 0)
                    improvement = exp_result.get("performance_improvement_percent", 0)
                    time_saved = exp_result.get("baseline_avg_time_ms", 0) - exp_time
                    
                    # Get experiment metadata for title
                    metadata = exp_result.get("experiment_metadata", {})
                    title = metadata.get("title", f"Experiment {exp_id}")
                    
                    status_icon = "✅" if improvement > 0 else "⚠️" if improvement == 0 else "📉"
                    print(f"      {i}. {status_icon} {title}")
                    print(f"         ID: {exp_id}")
                    print(f"         Result: {exp_time:.2f}ms ({improvement:+.1f}%, {time_saved:+.2f}ms)")
                else:
                    # Show failed experiment with error
                    error = exp_result.get("error", "unknown error")
                    
                    # Try to get title from experiment metadata if available
                    metadata = exp_result.get("experiment_metadata", {})
                    title = metadata.get("title", f"Experiment {exp_id}")
                    
                    print(f"      {i}. ❌ {title}")
                    print(f"         ID: {exp_id}")
                    print(f"         Error: {error}")
                
                print()  # Space between experiments
        
        print()