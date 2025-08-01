"""Simple CLI for QueryAgent."""

import asyncio
import sys
import os

from ..core.agent import QueryAgent
from ..core.config import Config


def print_result(result):
    """Print optimization result."""
    if result["success"]:
        print("\n✅ Analysis Complete!")
        print(f"\n📋 Analysis:\n{result['analysis']}")
        
        if result["recommendations"]:
            print("\n💡 Key Recommendations:")
            for i, rec in enumerate(result["recommendations"], 1):
                print(f"{i}. {rec}")
    else:
        print(f"\n❌ Error: {result['error']}")


async def main():
    """Main CLI function."""
    if len(sys.argv) < 2:
        print("Usage: python3 -m queryagent.cli.main 'SELECT ...'")
        print("Example: python3 -m queryagent.cli.main 'SELECT * FROM my_table'")
        sys.exit(1)
    
    query = sys.argv[1]
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: Set ANTHROPIC_API_KEY environment variable")
        print("Example: export ANTHROPIC_API_KEY='your_key_here'")
        sys.exit(1)
    
    print("🚀 QueryAgent - ClickHouse Query Optimizer")
    print(f"🔍 Analyzing query: {query[:50]}{'...' if len(query) > 50 else ''}")
    
    # Initialize development logging
    from ..utils.dev_logger import init_dev_logger
    dev_logger = init_dev_logger()
    print(f"📝 Development log: {dev_logger.get_log_file_path()}")
    
    try:
        # Initialize agent
        config = Config.load()
        print("Config Loaded")
        agent = QueryAgent(config)
        
        # Initialize and run
        if await agent.initialize():
            print("✅ Connected to ClickHouse")
            result = await agent.optimize_query(query)
            print_result(result)
        else:
            print("❌ Failed to connect to ClickHouse")
            print("💡 Make sure ClickHouse is running: docker start clickhouse-server3")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())