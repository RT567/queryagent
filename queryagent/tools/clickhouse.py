"""Simple ClickHouse tools for query execution."""

from typing import Dict, Any
from dataclasses import dataclass

try:
    import clickhouse_connect
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False

from ..utils.dev_logger import get_dev_logger

class QueryResult:
    """Simple result from a ClickHouse query."""
    
    def __init__(self, data=None, error=None):
        self.data = data or []
        self.error = error

@dataclass 
class QueryAnalysisResult:
    """Comprehensive query analysis from EXPLAIN commands."""
    plan: QueryResult
    pipeline: QueryResult
    estimate: QueryResult
    plan_with_indexes: QueryResult
    plan_with_projections: QueryResult
    syntax: QueryResult
    ast: QueryResult

class ClickHouseTools:
    """Simple ClickHouse connection and query tools."""
    
    def __init__(self, host: str = "localhost", port: int = 8123, 
                 database: str = "default", username: str = "default", 
                 password: str = ""):
        
        if not CLICKHOUSE_AVAILABLE:
            raise ImportError("clickhouse-connect not available. Run: pip install clickhouse-connect")
        
        self.logger = get_dev_logger()
        
        self.client = clickhouse_connect.get_client(
            host=host,
            port=port,
            database=database,
            username=username,
            password=password,
            send_receive_timeout=30
        )
    
    async def run_readonly_query(self, query: str) -> QueryResult:
        """Execute a read-only query."""
        try:
            # Special handling for AST queries - use command() instead of query()
            if "EXPLAIN AST" in query.upper():
                result = self.client.command(query)
                
                # Log and return AST as structured data
                if result:
                    lines = result.count('\n') + 1
                    ast_data = [{"ast_tree": result.strip()}]
                    self.logger.log_clickhouse_query(query, f"AST result: {lines} lines", full_result_data=ast_data)
                    return QueryResult(data=ast_data)
                else:
                    empty_data = [{"ast_tree": "Empty AST result"}]
                    self.logger.log_clickhouse_query(query, "Empty AST result", full_result_data=empty_data)
                    return QueryResult(data=empty_data)
            
            # Regular queries use query() method
            result = self.client.query(query)
            
            # Convert to list of dicts
            if result.result_rows:
                columns = result.column_names
                data = []
                for row in result.result_rows:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        row_dict[col] = row[i] if i < len(row) else None
                    data.append(row_dict)
            else:
                data = []
            
            # Log successful query with row count only (not full data to avoid massive logs)
            row_count = len(data)
            
            # Only log full data for non-experiment queries (experiments can return 100k+ rows)
            # Detect if query is likely from metrics retrieval (system.query_log queries)
            is_metrics_query = "system.query_log" in query.lower()
            # For now, avoid logging full data for very large result sets
            should_log_full_data = row_count <= 100 or is_metrics_query
            
            if should_log_full_data:
                self.logger.log_clickhouse_query(query, f"Success: {row_count} rows returned", full_result_data=data)
            else:
                self.logger.log_clickhouse_query(query, f"Success: {row_count} rows returned (data not logged - result set too large)")
            
            return QueryResult(data=data)
            
        except Exception as e:
            # Log failed query
            self.logger.log_clickhouse_query(query, error=str(e))
            return QueryResult(error=str(e))
    
    async def get_table_info(self, table_name: str) -> QueryResult:
        """Get basic table information."""
        try:
            query = f"DESCRIBE TABLE {table_name}"
            return await self.run_readonly_query(query)
        except Exception as e:
            return QueryResult(error=f"Failed to get table info: {e}")
    
    async def check_system_load(self) -> QueryResult:
        """Check cluster load metrics and return readiness assessment."""
        try:
            query = """
            /* Put your cluster name here – yours is `default`                          */
WITH 'default' AS cl

SELECT
    /* user‑facing load (queries already >1 s) --------------------------- */
    countIf(p.elapsed > 1)                           AS active_q_1s,
    max(p.elapsed)                                   AS longest_q_sec,

    /* memory held by those queries ------------------------------------- */
    round(sum(p.memory_usage) / 1e9, 2)              AS mem_gb,

    /* background MergeTree housekeeping -------------------------------- */
    sumIf(m.value, m.metric = 'BackgroundMergesAndMutationsPoolTask') AS merge_tasks,
    sumIf(m.value, m.metric = 'DelayedInserts')                       AS delayed_inserts
FROM
    /* live queries from every replica */
    (
        SELECT elapsed, memory_usage
        FROM clusterAllReplicas(cl, 'system', 'processes')
    ) AS p

    /* instantaneous server counters */
CROSS JOIN
    (
        SELECT metric, value
        FROM clusterAllReplicas(cl, 'system', 'metrics')
    ) AS m

SETTINGS skip_unavailable_shards = 1;   -- avoids an error if a replica is down

            """
            result = await self.run_readonly_query(query)
            
            if result.error:
                return result
            
            # Apply readiness logic in Python
            if result.data:
                metrics = result.data[0]
                active_queries = metrics.get('active_queries', 0)
                max_duration = metrics.get('max_query_duration_sec', 0) or 0
                memory_gb = metrics.get('total_memory_gb', 0) or 0
                
                # Determine if cluster is ready for experiments
                ready = (active_queries < 5 and 
                        max_duration < 300 and 
                        memory_gb < 50)
                
                metrics['ready'] = ready
                result.data = [metrics]
            
            return result
        except Exception as e:
            return QueryResult(error=f"Failed to check system load: {e}")
    
    async def comprehensive_query_analysis(self, query: str) -> QueryAnalysisResult:
        """
        Gather all available information using all EXPLAIN variants.
        Returns raw results for LLM analysis without hardcoded interpretations.
        """
        # Execute all EXPLAIN variants in parallel would be ideal, but let's do sequential for now
        plan = await self._explain_query(query, "PLAN")
        pipeline = await self._explain_query(query, "PIPELINE") 
        estimate = await self._explain_query(query, "ESTIMATE")
        plan_with_indexes = await self._explain_query(query, "PLAN", {"indexes": 1})
        plan_with_projections = await self._explain_query(query, "PLAN", {"projections": 1})
        syntax = await self._explain_query(query, "SYNTAX", {"run_query_tree_passes": 1})
        
        # Try AST with detailed error reporting
        ast = await self._explain_query(query, "AST")
        
        return QueryAnalysisResult(
            plan=plan,
            pipeline=pipeline,
            estimate=estimate,
            plan_with_indexes=plan_with_indexes,
            plan_with_projections=plan_with_projections,
            syntax=syntax,
            ast=ast
        )
    
    async def _explain_query(self, query: str, explain_type: str, settings: Dict[str, Any] = None) -> QueryResult:
        """Execute EXPLAIN query with specified type and settings."""
        if settings:
            settings_parts = [f"{k} = {v}" for k, v in settings.items()]
            settings_str = " " + ", ".join(settings_parts)
            explain_query = f"EXPLAIN {explain_type}{settings_str} {query}"
        else:
            explain_query = f"EXPLAIN {explain_type} {query}"
        
        return await self.run_readonly_query(explain_query)