"""
Stage 2: Static Analysis

Comprehensive query analysis using all EXPLAIN variants, table schema analysis.
Uses comprehensive_query_analysis() for efficient multi-EXPLAIN data gathering.
"""

import asyncio
from typing import Dict, Any, List

from .base import BaseStage, StageOutput, StageResult
from ...tools.clickhouse import QueryResult, QueryAnalysisResult
from ...utils.dev_logger import get_dev_logger


class StaticAnalysisStage(BaseStage):
    """Stage 2: Static Query Analysis"""
    
    def __init__(self, clickhouse, config: Dict[str, Any]):
        super().__init__(clickhouse, config)
        
        # Analysis options from config
        self.analysis_config = config.get('static_analysis', {})
        self.analyze_tables = self.analysis_config.get('analyze_tables', True)
        self.check_settings = self.analysis_config.get('check_settings', True)
        self.use_llm_validation = self.analysis_config.get('use_llm_validation', True)
        
        # We'll need LLM access for enhanced analysis
        self.llm = None  # Will be injected by orchestrator
        
        # Development logger
        self.logger = get_dev_logger()
    
    async def execute(self, query: str) -> StageOutput:
        """Execute the static analysis stage."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            self.logger.log_stage_detail("static_analysis", f"Starting static analysis for query: {query[:100]}...")
            
            analysis_data = {
                "original_query": query,
                "comprehensive_analysis": None,
                "table_schemas": {},
                "query_settings": None,
                "analysis_summary": {}
            }
            
            # 1. Run comprehensive EXPLAIN analysis
            self.logger.log_stage_detail("static_analysis", "Running comprehensive EXPLAIN analysis")
            comprehensive_result = await self.clickhouse.comprehensive_query_analysis(query)
            analysis_data["comprehensive_analysis"] = comprehensive_result
            
            # Log EXPLAIN results summary
            explain_count = 0
            explain_errors = 0
            if comprehensive_result:
                explains = [comprehensive_result.plan, comprehensive_result.pipeline, comprehensive_result.estimate, 
                           comprehensive_result.plan_with_indexes, comprehensive_result.plan_with_projections, 
                           comprehensive_result.syntax, comprehensive_result.ast]
                explain_count = len(explains)
                explain_errors = len([e for e in explains if e.error])
            self.logger.log_stage_detail("static_analysis", f"EXPLAIN analysis: {explain_count-explain_errors}/{explain_count} successful")
            
            # 2. Analyze referenced tables with LLM validation
            if self.analyze_tables:
                # Get table names using hard-coded regex
                hardcoded_tables = self._extract_table_names(query)
                
                # Validate and enhance with LLM if available
                if self.use_llm_validation and self.llm:
                    self.logger.log_stage_detail("static_analysis", f"Using LLM to validate table extraction. Hardcoded found: {hardcoded_tables}")
                    llm_tables = await self._llm_extract_table_names(query)
                    self.logger.log_stage_detail("static_analysis", f"LLM found tables: {llm_tables}")
                    # Combine both approaches - union of results
                    all_tables = list(set(hardcoded_tables + llm_tables))
                    analysis_data["table_extraction"] = {
                        "hardcoded_method": hardcoded_tables,
                        "llm_method": llm_tables,
                        "final_tables": all_tables
                    }
                else:
                    all_tables = hardcoded_tables
                    analysis_data["table_extraction"] = {
                        "hardcoded_method": hardcoded_tables,
                        "final_tables": all_tables
                    }
                
                # Analyze each identified table
                self.logger.log_stage_detail("static_analysis", f"Analyzing {len(all_tables)} tables: {all_tables}")
                for table_name in all_tables:
                    schema_result = await self._analyze_table_schema(table_name)
                    if not schema_result.error:
                        analysis_data["table_schemas"][table_name] = schema_result.data
                        self.logger.log_stage_detail("static_analysis", f"Successfully analyzed table: {table_name}")
                        self.logger.log_table_schema(table_name, schema_result.data)
                    else:
                        analysis_data["table_schemas"][table_name] = {"error": schema_result.error}
                        self.logger.log_stage_detail("static_analysis", f"Failed to analyze table {table_name}: {schema_result.error}")
                        self.logger.log_table_schema(table_name, {}, error=schema_result.error)
            
            # 3. Get relevant query settings
            if self.check_settings:
                settings_result = await self._get_query_settings()
                if not settings_result.error:
                    analysis_data["query_settings"] = settings_result.data
            
            # 4. Generate analysis summary
            analysis_data["analysis_summary"] = self._generate_summary(analysis_data)
            
            self.logger.log_stage_detail("static_analysis", f"Static analysis completed. Tables: {len(analysis_data['table_schemas'])}, Complexity: {analysis_data['analysis_summary'].get('query_complexity', 'unknown')}")
            return self._create_success_output("static_analysis", analysis_data, start_time)
            
        except Exception as e:
            self.logger.log_error("STATIC_ANALYSIS", str(e), "Static analysis stage failed")
            return self._create_error_output("static_analysis", f"Static analysis failed: {str(e)}", start_time)
    
    async def _analyze_table_schema(self, table_name: str) -> 'QueryResult':
        """
        MCP-style interface for comprehensive table analysis.
        """
        try:
            # Get basic table info
            describe_result = await self.clickhouse.get_table_info(table_name)
            if describe_result.error:
                return describe_result
            
            # Get table engine and storage info - fix column names for this ClickHouse version
            table_info_query = f"""
                SELECT 
                    engine,
                    total_rows,
                    total_bytes
                FROM system.tables 
                WHERE name = '{table_name}' 
                AND database = currentDatabase()
            """
            
            table_info_result = await self.clickhouse.run_readonly_query(table_info_query)
            
            # Get column info from system.columns
            column_info_query = f"""
                SELECT 
                    name,
                    type,
                    position,
                    default_kind,
                    default_expression
                FROM system.columns 
                WHERE table = '{table_name}' 
                AND database = currentDatabase()
                ORDER BY position
            """
            
            column_info_result = await self.clickhouse.run_readonly_query(column_info_query)
            
            # Combine all results
            analysis = {
                "table_name": table_name,
                "schema": describe_result.data,
                "table_info": table_info_result.data if not table_info_result.error else None,
                "column_info": column_info_result.data if not column_info_result.error else None,
            }
            
            from ...tools.clickhouse import QueryResult
            return QueryResult(data=analysis)
            
        except Exception as e:
            from ...tools.clickhouse import QueryResult
            return QueryResult(error=f"Failed to analyze table schema: {e}")
    
    async def _get_query_settings(self) -> 'QueryResult':
        """Get current ClickHouse settings that affect query execution."""
        try:
            settings_query = """
                SELECT name, value, description
                FROM system.settings 
                WHERE name IN (
                    'max_memory_usage',
                    'max_execution_time', 
                    'max_threads',
                    'join_algorithm',
                    'join_use_nulls',
                    'enable_optimize_predicate_expression',
                    'optimize_move_to_prewhere',
                    'prefer_localhost_replica'
                )
                ORDER BY name
            """
            
            return await self.clickhouse.run_readonly_query(settings_query)
            
        except Exception as e:
            from ...tools.clickhouse import QueryResult
            return QueryResult(error=f"Failed to get query settings: {e}")
    
    def _extract_table_names(self, query: str) -> List[str]:
        """Extract table names from query (simple implementation)."""
        table_names = []
        try:
            upper_query = query.upper()
            
            # Look for FROM clauses
            import re
            from_matches = re.findall(r'FROM\s+(\w+(?:\.\w+)?)', upper_query)
            for match in from_matches:
                # Remove database prefix if present
                table_name = match.split('.')[-1].lower()
                if table_name not in table_names:
                    table_names.append(table_name)
            
            # Look for JOIN clauses
            join_matches = re.findall(r'JOIN\s+(\w+(?:\.\w+)?)', upper_query)
            for match in join_matches:
                table_name = match.split('.')[-1].lower()
                if table_name not in table_names:
                    table_names.append(table_name)
                    
        except Exception:
            # If parsing fails, return empty list
            pass
        
        return table_names
    
    def _generate_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of the static analysis."""
        summary = {
            "query_complexity": "unknown",
            "tables_analyzed": len(analysis_data["table_schemas"]),
            "comprehensive_analysis_available": analysis_data["comprehensive_analysis"] is not None,
            "potential_issues": [],
            "optimization_opportunities": []
        }
        
        # Analyze comprehensive results if available
        comprehensive = analysis_data.get("comprehensive_analysis")
        if comprehensive:
            # Count successful EXPLAIN operations
            explain_results = [
                comprehensive.plan, comprehensive.pipeline, comprehensive.estimate,
                comprehensive.plan_with_indexes, comprehensive.plan_with_projections,
                comprehensive.syntax, comprehensive.ast
            ]
            successful_explains = len([r for r in explain_results if not r.error])
            summary["explain_operations_successful"] = successful_explains
            summary["total_explain_operations"] = len(explain_results)
        
        # Note: Removed generic query pattern analysis since LLM provides much more 
        # sophisticated analysis based on actual EXPLAIN output
        
        if len(analysis_data["table_schemas"]) > 2:
            summary["query_complexity"] = "high"
        elif len(analysis_data["table_schemas"]) > 1:
            summary["query_complexity"] = "medium"
        else:
            summary["query_complexity"] = "low"
        
        return summary
    
    async def _llm_extract_table_names(self, query: str) -> List[str]:
        """Use LLM to extract table names from the query."""
        try:
            if not self.llm:
                return []
            
            prompt = f"""Analyze this ClickHouse SQL query and extract ALL table names referenced:

Query:
{query}

Please identify ALL tables referenced in this query, including:
- Tables in FROM clauses
- Tables in JOIN clauses  
- Tables in subqueries
- Tables in CTEs/WITH clauses
- Tables in INSERT/UPDATE statements

Return ONLY a JSON list of table names (without database prefixes), like:
["table1", "table2", "table3"]

If no tables are found, return: []
"""
            
            response = await self.llm.ask(prompt)
            self.logger.log_stage_detail("static_analysis", f"LLM table extraction response: {response.content[:200]}...")
            
            # Parse LLM response to extract table names
            import json
            import re
            
            # Try to find JSON array in response
            json_match = re.search(r'\[.*?\]', response.content)
            if json_match:
                try:
                    tables = json.loads(json_match.group())
                    # Clean and validate table names
                    clean_tables = []
                    for table in tables:
                        if isinstance(table, str) and table.strip():
                            # Remove database prefix if present
                            clean_name = table.split('.')[-1].strip().lower()
                            if clean_name not in clean_tables:
                                clean_tables.append(clean_name)
                    return clean_tables
                except json.JSONDecodeError:
                    pass
            
            # Fallback: extract table-like words from response
            words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', response.content)
            return [w.lower() for w in words if len(w) > 2][:5]  # Limit to 5 tables max
            
        except Exception as e:
            # If LLM fails, return empty list (fallback to hardcoded method)
            return []
    
    async def _llm_enhance_table_analysis(self, table_name: str, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to provide additional insights about the table schema."""
        try:
            if not self.llm:
                return {"llm_insights": "LLM not available"}
            
            # Format schema information for LLM
            schema_summary = f"Table: {table_name}\n"
            
            if schema_data.get("table_info") and len(schema_data["table_info"]) > 0:
                table_info = schema_data["table_info"][0]
                schema_summary += f"Engine: {table_info.get('engine', 'unknown')}\n"
                schema_summary += f"Order By: {table_info.get('order_by', 'none')}\n"
                schema_summary += f"Partition By: {table_info.get('partition_by', 'none')}\n"
                schema_summary += f"Rows: {table_info.get('total_rows', 'unknown')}\n"
            
            if schema_data.get("column_info"):
                schema_summary += "\nColumns:\n"
                for col in schema_data["column_info"][:10]:  # Limit to first 10 columns
                    schema_summary += f"- {col.get('name', '')}: {col.get('type', '')}\n"
            
            prompt = f"""Analyze this ClickHouse table schema and provide optimization insights:

{schema_summary}

Please provide:
1. Potential performance issues with this table structure
2. Optimization opportunities (indexing, partitioning, ordering)
3. Query patterns that would work well/poorly with this schema
4. Any red flags or concerns

Keep response concise and ClickHouse-specific.
"""
            
            response = await self.llm.ask(prompt)
            
            return {
                "llm_insights": response.content,
                "analysis_type": "table_optimization"
            }
            
        except Exception as e:
            return {"llm_insights": f"LLM analysis failed: {str(e)}"}
    
    def set_llm(self, llm):
        """Inject LLM provider for enhanced analysis."""
        self.llm = llm