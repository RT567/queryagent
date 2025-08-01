"""Simple Anthropic Claude provider."""

import os
from ..base import LLMProvider, LLMResponse
from ...utils.dev_logger import get_dev_logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(LLMProvider):
    """Simple Anthropic Claude provider."""
    
    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.1, max_tokens: int = 4000):
        super().__init__(model, temperature, max_tokens)
        
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.logger = get_dev_logger()
    
    async def ask(self, prompt: str, attempt: int = None, **kwargs) -> LLMResponse:
        """Send request to Anthropic Claude."""
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_content = response.content[0].text
            
            # Log the complete interaction
            self.logger.log_llm_call(prompt, response=response_content, model=self.model, attempt=attempt)
            
            return LLMResponse(
                content=response_content,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            )
        except Exception as e:
            # Log the error
            self.logger.log_llm_call(prompt, error=str(e), model=self.model, attempt=attempt)
            raise RuntimeError(f"Anthropic API error: {e}")