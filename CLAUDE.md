# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QueryAgent is an AI-powered tool for optimizing complex ClickHouse queries involving cascading CTEs. The agent analyzes query performance, runs experiments, and provides ClickHouse-specific optimization recommendations and information visualisations.



### Development Workflow
```bash
# Run the agent CLI
python -m queryagent.cli.main --query "SELECT ..." 
```

## Code Architecture

### Core Components
- `core/agent.py`: Main orchestration loop implementing 5-stage workflow
- `llm/providers/`: Multi-LLM support (OpenAI, Anthropic, Gemini)
- `tools/clickhouse.py`: Embedded MCP-style ClickHouse operations
- `stages.py`: Implementation of the analysis stages


### Multi-LLM Provider System
All LLM providers implement the `LLMProvider` interface:
- `ask()`: Single request/response
- `stream()`: Streaming responses
- `get_capabilities()`: Provider-specific features

Configuration in `./queryagent.yaml`

## ClickHouse-Specific Knowledge

### Key System Tables for Analysis
- `system.query_log`: Query execution history and metrics
- `system.query_thread_log`: Thread-level performance data
- `system.parts`: Table storage information
- `system.columns`: Column metadata and compression
- `system.metrics`: Real-time cluster metrics
