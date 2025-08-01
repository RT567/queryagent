"""
Stage 3: Experiment Planning

Uses LLM to analyze EXPLAIN data and generate targeted performance experiments.
Focuses on query-level optimizations that can be determined from EXPLAIN output.
"""

import asyncio
from typing import Dict, Any, List
import json
import re
import uuid
import time

from .base import BaseStage, StageOutput, StageResult
from ...tools.clickhouse import QueryResult


class ExperimentPlanningStage(BaseStage):
    """Stage 3: Generate Performance Experiments"""
    
    def __init__(self, clickhouse, config: Dict[str, Any]):
        super().__init__(clickhouse, config)
        
        # Planning options from config
        self.planning_config = config.get('experiment_planning', {})
        self.effort_level = config.get('effort_level', 'medium')
        
        # Effort-based experiment limits (maximum, not required)
        self.effort_limits = {
            'low': 5,
            'medium': 8, 
            'high': 12
        }
        
        # Retry configuration for robust experiment generation
        self.max_retry_attempts = 5
        self.retry_delay_seconds = 1
        
        # LLM provider for experiment generation
        self.llm = None  # Will be injected by orchestrator
        
        # Development logger
        from ...utils.dev_logger import get_dev_logger
        self.logger = get_dev_logger()
    
    async def execute(self, query: str, static_analysis_data: Dict[str, Any]) -> StageOutput:
        """Execute the experiment planning stage."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if not self.llm:
                return self._create_error_output("experiment_planning", 
                    "LLM provider required for experiment planning", start_time)
            
            # Determine maximum experiments based on effort level
            max_experiments = self.effort_limits.get(self.effort_level, 8)
            
            planning_data = {
                "original_query": query,
                "effort_level": self.effort_level,
                "max_experiments": max_experiments,
                "experiments": [],
                "planning_summary": {},
                "llm_attempts": []
            }
            
            print(f"🧪 Stage 3: Generating experiments (effort level: {self.effort_level}, max: {max_experiments})")
            
            # Generate experiments using LLM (with retry on parsing failure)
            experiments = await self._generate_experiments_with_retry(query, static_analysis_data, max_experiments, planning_data)
            planning_data["experiments"] = experiments
            
            # Print experiment results
            self._print_experiment_results(experiments, planning_data)
            
            # Generate planning summary
            planning_data["planning_summary"] = self._generate_planning_summary(experiments)
            
            if not experiments:
                return self._create_error_output("experiment_planning", 
                    "Failed to generate valid experiments after multiple LLM attempts", start_time)
            
            return self._create_success_output("experiment_planning", planning_data, start_time)
            
        except Exception as e:
            return self._create_error_output("experiment_planning", 
                f"Experiment planning failed: {str(e)}", start_time)
    
    async def _generate_experiments_with_retry(self, query: str, static_analysis_data: Dict[str, Any], 
                                             max_experiments: int, planning_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experiments with robust retry logic until valid experiments are obtained."""
        
        context = self._prepare_analysis_context(query, static_analysis_data)
        last_response = ""
        
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                print(f"   🔄 Attempt {attempt}/{self.max_retry_attempts}: ", end="")
                
                # Determine prompt strategy based on attempt
                if attempt == 1:
                    prompt = self._create_main_experiment_prompt(context, max_experiments)
                    prompt_type = "main_generation"
                    print("Generating experiments...")
                else:
                    prompt = self._create_parsing_fix_prompt(last_response, max_experiments)
                    prompt_type = "parsing_fix"
                    print("Fixing parsing issues...")
                
                # Make LLM call with attempt logging
                response = await self.llm.ask(prompt, attempt=attempt)
                last_response = response.content
                
                # Track attempt
                attempt_info = {
                    "attempt": attempt,
                    "type": prompt_type,
                    "success": False,
                    "response_length": len(response.content),
                    "timestamp": time.time()
                }
                
                # Parse and validate experiments
                experiments = self._parse_experiments_response(response.content, max_experiments)
                
                if experiments:
                    print(f"      📝 Parsed {len(experiments)} experiments")
                    
                    # Validate experiment syntax
                    validated_experiments = await self._validate_experiments(experiments)
                    print(f"      ✅ {len(validated_experiments)} experiments passed validation")
                    
                    if validated_experiments:
                        # Enhance experiments with required fields
                        enhanced_experiments = self._enhance_experiments(validated_experiments, query)
                        
                        attempt_info["success"] = True
                        attempt_info["experiments_parsed"] = len(enhanced_experiments)
                        attempt_info["experiments_validated"] = len(enhanced_experiments)
                        planning_data["llm_attempts"].append(attempt_info)
                        
                        print(f"      🎯 Successfully generated {len(enhanced_experiments)} valid experiments")
                        return enhanced_experiments
                    else:
                        attempt_info["parse_error"] = "No experiments passed validation"
                        print(f"      ❌ No experiments passed validation")
                else:
                    attempt_info["parse_error"] = "No valid experiments parsed from response"
                    print(f"      ❌ Failed to parse experiments from response")
                
                planning_data["llm_attempts"].append(attempt_info)
                
                # Add delay before retry (except last attempt)
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(self.retry_delay_seconds)
                    
            except Exception as e:
                attempt_info = {
                    "attempt": attempt,
                    "type": prompt_type if 'prompt_type' in locals() else "unknown",
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                }
                planning_data["llm_attempts"].append(attempt_info)
                
                # Add delay before retry (except last attempt)
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(self.retry_delay_seconds)
        
        # All attempts exhausted
        return []
    
    def _create_main_experiment_prompt(self, context: str, max_experiments: int) -> str:
        """Create the main experiment generation prompt."""
        return f"""You are a ClickHouse query optimization expert. Based on the comprehensive EXPLAIN analysis below, design targeted performance experiments to test different query-level optimization strategies.

ANALYSIS CONTEXT:
{context}

EXPERIMENT REQUIREMENTS:
1. Generate as many meaningful experiments as you see optimization opportunities (up to {max_experiments} maximum)
2. Each experiment should test a specific query rewriting hypothesis
3. Focus ONLY on query-level optimizations, NOT database configuration changes
4. Base optimizations on what you see in the EXPLAIN output - look for inefficiencies in the actual execution plan
5. Only suggest experiments where you can see clear potential for improvement

OPTIMIZATION CATEGORIES TO CONSIDER (based on EXPLAIN analysis):

**Query Structure Changes:**
- CTE reordering when EXPLAIN shows suboptimal execution order
- Manual JOIN reordering when optimizer chooses poor join sequence
- Subquery vs JOIN conversion when EXPLAIN reveals performance differences
- EXISTS vs IN vs JOIN rewriting based on execution plan analysis

**Filter Optimization from EXPLAIN PIPELINE:**
- WHERE to PREWHERE conversion when pipeline shows late filtering
- Filter condition reordering when EXPLAIN shows inefficient predicate evaluation
- Push filters into subqueries when they're applied too late in the plan

**SELECT Clause Optimization:**
- Replace SELECT * with specific columns when EXPLAIN shows unnecessary reads
- Column reordering for better processing efficiency
- Eliminate unused expressions in SELECT/GROUP BY

**Query Structure Simplification:**
- Eliminate redundant CTEs that EXPLAIN shows aren't optimized away
- Flatten nested queries when EXPLAIN reveals unnecessary complexity
- Break complex expressions when EXPLAIN shows expensive evaluation

**ClickHouse-Specific Query Optimizations:**
- Add FINAL when needed (but not automatically applied)
- Use SAMPLE for approximate results when appropriate
- Optimize array/string functions based on EXPLAIN cost analysis

IMPORTANT: 
- If you only see 1-2 clear optimization opportunities, only suggest 1-2 experiments
- If you see many opportunities, suggest more experiments (up to the maximum)
- Don't create experiments just to reach a number - only suggest meaningful ones
- Base each experiment on specific issues you observe in the EXPLAIN output

Return response as a valid JSON array ONLY. No other text.

[
  {{
    "title": "Brief description of the optimization being tested",
    "targeted_inefficiency": "Specific inefficiency observed in EXPLAIN output",
    "hypothesis": "What you expect to improve based on the EXPLAIN output and why", 
    "modified_query": "The optimized SQL query to test",
    "expected_impact": "low/medium/high",
    "risk_level": "low/medium/high",
    "explanation": "Technical reasoning based on the EXPLAIN analysis",
    "expected_metrics": {{
      "query_duration_ms": "lower/higher/same",
      "read_rows": "lower/higher/same", 
      "read_bytes": "lower/higher/same",
      "memory_usage": "lower/higher/same"
    }}
  }}
]"""
    
    def _create_parsing_fix_prompt(self, previous_response: str, max_experiments: int) -> str:
        """Create a prompt to fix parsing issues from the previous response."""
        return f"""The previous response could not be parsed as valid JSON. Please extract the experiment information from the response below and format it as a proper JSON array.

PREVIOUS RESPONSE:
{previous_response[:2000]}

Please reformat this into a valid JSON array. Include as many experiment objects as are described in the response (don't add or remove experiments, just fix the formatting). Each object must have these fields:
- title (string)
- targeted_inefficiency (string)
- hypothesis (string)
- modified_query (string)
- expected_impact (string: "low", "medium", or "high")
- risk_level (string: "low", "medium", or "high") 
- explanation (string)
- expected_metrics (object with query_duration_ms, read_rows, read_bytes, memory_usage)

Return ONLY the JSON array, no other text:

[
  {{
    "title": "...",
    "targeted_inefficiency": "...",
    "hypothesis": "...",
    "modified_query": "...",
    "expected_impact": "...",
    "risk_level": "...",
    "explanation": "...",
    "expected_metrics": {{
      "query_duration_ms": "lower",
      "read_rows": "lower",
      "read_bytes": "lower",
      "memory_usage": "lower"
    }}
  }}
]"""
    
    def _prepare_analysis_context(self, query: str, static_analysis_data: Dict[str, Any]) -> str:
        """Prepare comprehensive analysis context for LLM."""
        
        context_parts = []
        
        # Original query
        context_parts.append(f"ORIGINAL QUERY:\n{query}\n")
        
        # EXPLAIN analysis results
        comprehensive = static_analysis_data.get("comprehensive_analysis")
        if comprehensive:
            context_parts.append("EXPLAIN ANALYSIS RESULTS:")
            
            # Include successful EXPLAIN results with more detail
            explain_results = [
                ("PLAN", comprehensive.plan),
                ("PIPELINE", comprehensive.pipeline),
                ("ESTIMATE", comprehensive.estimate),
                ("PLAN_WITH_INDEXES", comprehensive.plan_with_indexes),
                ("PLAN_WITH_PROJECTIONS", comprehensive.plan_with_projections),
                ("SYNTAX", comprehensive.syntax)
            ]
            
            for name, result in explain_results:
                if not result.error and result.data:
                    # Include detailed results for LLM analysis
                    result_text = str(result.data)[:1500]
                    context_parts.append(f"\n{name}:\n{result_text}")
                elif result.error:
                    context_parts.append(f"\n{name}: ERROR - {result.error}")
        
        # Table schemas
        table_schemas = static_analysis_data.get("table_schemas", {})
        if table_schemas:
            context_parts.append("\nTABLE SCHEMAS:")
            for table_name, schema_data in table_schemas.items():
                if "error" not in schema_data:
                    context_parts.append(f"\nTable: {table_name}")
                    
                    # Add table info
                    if schema_data.get("table_info") and len(schema_data["table_info"]) > 0:
                        info = schema_data["table_info"][0]
                        context_parts.append(f"  Engine: {info.get('engine', 'unknown')}")
                        context_parts.append(f"  Rows: {info.get('total_rows', 'unknown')}")
                        context_parts.append(f"  Order By: {info.get('order_by', 'none')}")
                    
                    # Add column info (limited)
                    if schema_data.get("column_info"):
                        context_parts.append(f"  Columns ({len(schema_data['column_info'])} total):")
                        for col in schema_data["column_info"][:10]:
                            context_parts.append(f"    - {col.get('name', '')}: {col.get('type', '')}")
        
        return "\n".join(context_parts)
    
    def _parse_experiments_response(self, response_content: str, max_experiments: int) -> List[Dict[str, Any]]:
        """Parse LLM response to extract experiment definitions."""
        experiments = []
        
        try:
            # Clean the response - remove any text before/after JSON
            content = response_content.strip()
            
            # Find JSON array
            start_idx = content.find('[')
            end_idx = content.rfind(']')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = content[start_idx:end_idx + 1]
                
                try:
                    parsed_experiments = json.loads(json_str)
                    
                    if isinstance(parsed_experiments, list):
                        for exp in parsed_experiments:
                            if isinstance(exp, dict):
                                # Validate required fields
                                required_fields = ['title', 'hypothesis', 'modified_query']
                                if all(field in exp and exp[field] for field in required_fields):
                                    experiment = {
                                        "title": str(exp.get('title', '')).strip(),
                                        "targeted_inefficiency": str(exp.get('targeted_inefficiency', exp.get('title', ''))).strip(),
                                        "hypothesis": str(exp.get('hypothesis', '')).strip(),
                                        "modified_query": str(exp.get('modified_query', '')).strip(),
                                        "expected_impact": str(exp.get('expected_impact', 'medium')).lower(),
                                        "risk_level": str(exp.get('risk_level', 'low')).lower(),
                                        "explanation": str(exp.get('explanation', '')).strip(),
                                        "expected_metrics": exp.get('expected_metrics', {
                                            "query_duration_ms": "lower",
                                            "read_rows": "lower",
                                            "read_bytes": "lower", 
                                            "memory_usage": "lower"
                                        })
                                    }
                                    
                                    # Basic validation
                                    if (len(experiment["modified_query"]) > 20 and 
                                        len(experiment["title"]) > 5):
                                        experiments.append(experiment)
                                        
                except json.JSONDecodeError:
                    pass
                
        except Exception:
            pass
        
        # Respect the maximum but don't enforce it strictly
        return experiments[:max_experiments] if len(experiments) > max_experiments else experiments
    
    async def _validate_experiments(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate experiment queries for syntax correctness."""
        validated_experiments = []
        
        for experiment in experiments:
            try:
                modified_query = experiment.get('modified_query', '').strip()
                
                # Skip experiments with obviously invalid queries
                if len(modified_query) < 10 or not modified_query.upper().startswith('SELECT'):
                    continue
                
                # Test query syntax using EXPLAIN
                try:
                    explain_query = f"EXPLAIN SYNTAX {modified_query}"
                    result = await self.clickhouse.run_readonly_query(explain_query)
                    
                    # If EXPLAIN SYNTAX succeeds, query is syntactically valid
                    if not result.error:
                        validated_experiments.append(experiment)
                    else:
                        # Log validation failure but continue
                        experiment['validation_error'] = result.error
                        
                except Exception as e:
                    # If validation fails, skip this experiment
                    experiment['validation_error'] = str(e)
                    
            except Exception as e:
                # Skip malformed experiments
                continue
        
        return validated_experiments
    
    def _enhance_experiments(self, experiments: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Enhance experiments with required fields for execution stage."""
        enhanced_experiments = []
        
        for i, experiment in enumerate(experiments):
            # Generate unique experiment ID
            experiment_id = f"exp_{int(time.time())}_{i+1:03d}"
            
            enhanced_experiment = {
                "experiment_id": experiment_id,
                "original_query": original_query,
                "title": experiment.get('title', f'Experiment {i+1}'),
                "targeted_inefficiency": experiment.get('targeted_inefficiency', experiment.get('title', 'Unknown inefficiency')),
                "hypothesis": experiment.get('hypothesis', ''),
                "modified_query": experiment.get('modified_query', ''),
                "expected_impact": experiment.get('expected_impact', 'medium'),
                "risk_level": experiment.get('risk_level', 'low'),
                "explanation": experiment.get('explanation', ''),
                "expected_metrics": experiment.get('expected_metrics', {
                    "query_duration_ms": "lower",
                    "read_rows": "lower", 
                    "read_bytes": "lower",
                    "memory_usage": "lower"
                }),
                "created_timestamp": time.time(),
                "validation_status": "passed"
            }
            
            # Add validation error if present
            if 'validation_error' in experiment:
                enhanced_experiment['validation_error'] = experiment['validation_error']
                enhanced_experiment['validation_status'] = "failed"
            
            enhanced_experiments.append(enhanced_experiment)
        
        return enhanced_experiments
    
    def _print_experiment_results(self, experiments: List[Dict[str, Any]], planning_data: Dict[str, Any]) -> None:
        """Print detailed experiment results for development debugging."""
        print()
        print("🧪 EXPERIMENT PLANNING RESULTS")
        print("=" * 50)
        
        # Print LLM attempt summary
        attempts = planning_data.get("llm_attempts", [])
        successful_attempts = [a for a in attempts if a.get("success")]
        if successful_attempts:
            print(f"✅ Success after {len(attempts)} attempts")
        else:
            print(f"❌ Failed after {len(attempts)} attempts")
        
        if not experiments:
            print("No experiments generated")
            return
        
        print(f"\n📊 Generated {len(experiments)} experiments:")
        print()
        
        for i, exp in enumerate(experiments, 1):
            print(f"🔬 EXPERIMENT {i}: {exp.get('title', 'Untitled')}")
            print(f"   ID: {exp.get('experiment_id', 'unknown')}")
            print(f"   Target: {exp.get('targeted_inefficiency', 'Unknown')}")
            print(f"   Impact: {exp.get('expected_impact', 'unknown')} | Risk: {exp.get('risk_level', 'unknown')}")
            print(f"   Hypothesis: {exp.get('hypothesis', 'No hypothesis')[:100]}...")
            
            # Print expected metrics
            metrics = exp.get('expected_metrics', {})
            if metrics:
                metric_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
                print(f"   Expected: {metric_str}")
            
            # Print modified query (truncated)
            modified_query = exp.get('modified_query', '')
            if modified_query:
                query_preview = modified_query.replace('\n', ' ').strip()[:120]
                print(f"   Query: {query_preview}...")
            
            # Print validation status
            validation_status = exp.get('validation_status', 'unknown')
            if validation_status == 'passed':
                print(f"   Status: ✅ Validated")
            else:
                print(f"   Status: ❌ {validation_status}")
                if 'validation_error' in exp:
                    print(f"   Error: {exp['validation_error'][:100]}...")
            
            print()
        
        # Print summary
        impact_counts = {}
        risk_counts = {}
        for exp in experiments:
            impact = exp.get('expected_impact', 'unknown')
            risk = exp.get('risk_level', 'unknown')
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        print("📈 SUMMARY:")
        print(f"   Impact distribution: {dict(impact_counts)}")
        print(f"   Risk distribution: {dict(risk_counts)}")
        validated_count = len([e for e in experiments if e.get('validation_status') == 'passed'])
        print(f"   Validation success rate: {validated_count}/{len(experiments)} ({100*validated_count//len(experiments) if experiments else 0}%)")
        print()
    
    def _generate_planning_summary(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary of the experiment planning."""
        
        if not experiments:
            return {
                "total_experiments": 0,
                "status": "failed",
                "message": "No experiments generated"
            }
        
        # Count experiments by expected impact
        impact_counts = {"low": 0, "medium": 0, "high": 0}
        risk_counts = {"low": 0, "medium": 0, "high": 0}
        
        for exp in experiments:
            impact = exp.get("expected_impact", "medium")
            risk = exp.get("risk_level", "low")
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        return {
            "total_experiments": len(experiments),
            "status": "success", 
            "impact_distribution": impact_counts,
            "risk_distribution": risk_counts,
            "experiment_titles": [exp.get("title", "Untitled") for exp in experiments],
            "high_impact_count": impact_counts.get("high", 0),
            "low_risk_count": risk_counts.get("low", 0)
        }
    
    def set_llm(self, llm):
        """Inject LLM provider for experiment generation."""
        self.llm = llm