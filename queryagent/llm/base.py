"""Simple LLM provider interface."""

from typing import Dict, Any, Optional


class LLMResponse:
    """Simple LLM response container."""
    
    def __init__(self, content: str, usage: Optional[Dict[str, Any]] = None, 
                 model: Optional[str] = None, finish_reason: Optional[str] = None):
        self.content = content
        self.usage = usage
        self.model = model
        self.finish_reason = finish_reason


class LLMProvider:
    """Simple base class for LLM providers."""
    
    def __init__(self, model: str = "", temperature: float = 0.1, max_tokens: int = 4000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def ask(self, prompt: str, **kwargs) -> LLMResponse:
        """Single request/response interaction."""
        raise NotImplementedError("Subclass must implement ask()")