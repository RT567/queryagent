"""Simple configuration for QueryAgent."""

import yaml
from pathlib import Path
from urllib.parse import urlparse


class Config:
    """Simple configuration class."""
    
    def __init__(self, config_file: str = None):
        # Set defaults
        self.llm_provider = 'anthropic'
        self.llm_model = 'claude-sonnet-4-20250514'
        self.llm_temperature = 0.1
        self.llm_max_tokens = 4000
        
        self.clickhouse_url = 'clickhouse://localhost:8123'
        self.clickhouse_database = 'default'
        self.clickhouse_readonly = True
        
        # Initialize logger after setting defaults
        from ..utils.dev_logger import get_dev_logger
        self.logger = get_dev_logger()
        
        # Log default configuration
        self.logger.log_config({"source": "defaults", "config_file": config_file})
        
        # Load from file if provided
        if config_file:
            self._load_from_file(config_file)
    
    @property
    def clickhouse_host(self) -> str:
        """Parse host from clickhouse_url."""
        parsed = urlparse(self.clickhouse_url)
        return parsed.hostname or "localhost"
    
    @property
    def clickhouse_port(self) -> int:
        """Parse port from clickhouse_url."""
        parsed = urlparse(self.clickhouse_url)
        return parsed.port or 8123
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        try:
            config_path = Path(config_file).expanduser()
            if config_path.exists():
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Update configuration with file values
                if 'llm' in data:
                    llm = data['llm']
                    self.llm_provider = llm.get('provider', self.llm_provider)
                    self.llm_model = llm.get('model', self.llm_model)
                    self.llm_temperature = llm.get('temperature', self.llm_temperature)
                    self.llm_max_tokens = llm.get('max_tokens', self.llm_max_tokens)
                
                if 'clickhouse' in data:
                    ch = data['clickhouse']
                    self.clickhouse_url = ch.get('url', self.clickhouse_url)
                    self.clickhouse_database = ch.get('database', self.clickhouse_database)
                    self.clickhouse_readonly = ch.get('readonly', self.clickhouse_readonly)
                
                # Log loaded configuration
                self.logger.log_config({"source": "file", "file_path": config_file, "data_loaded": data})
                    
        except Exception as e:
            self.logger.log_error("CONFIG", str(e), f"Failed to load config file: {config_file}")
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    @classmethod
    def load(cls, config_file: str = None) -> 'Config':
        """Load configuration from default locations."""
        if config_file:
            return cls(config_file)
        
        # Try default locations
        default_locations = [
            './queryagent.yaml'
        ]
        
        # Create temporary instance to access logger
        temp_config = cls()
        temp_config.logger.log_config({"action": "searching_default_locations", "locations": default_locations})
        
        for location in default_locations:
            path = Path(location).expanduser()
            if path.exists():
                temp_config.logger.log_config({"action": "found_config_file", "location": str(path)})
                return cls(str(path))
        
        # Return default config if no file found
        temp_config.logger.log_config({"action": "using_defaults", "reason": "no_config_file_found"})
        return temp_config