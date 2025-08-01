# QueryAgent

AI-powered ClickHouse query optimization tool.

## Setup

**Requirements:**
- Python 3.8+
- ClickHouse server
- Anthropic API key

**Installation:**

```bash
# Check Python version
python3 --version

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API key
export ANTHROPIC_API_KEY="your_key_here"
```

## Usage

```bash
python -m queryagent.cli.main --query "SELECT ..." --effort-level medium
```
