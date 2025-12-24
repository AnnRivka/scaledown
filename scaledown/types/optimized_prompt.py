from pydantic import BaseModel, computed_field
from typing import Tuple, Optional
from .metrics import OptimizerMetrics


class OptimizedContext(BaseModel):
    """
    A Pydantic model representing optimized context from an optimizer.
    Similar to CompressedPrompt but for optimizer outputs.
    """
    
    content: str
    metrics: OptimizerMetrics
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"OptimizedContext({len(self.content)} chars, {self.compression_ratio:.2f}x compression)"
    
    @computed_field
    @property
    def tokens(self) -> Tuple[int, int]:
        """Return (original_tokens, optimized_tokens)."""
        return (
            self.metrics.original_tokens,
            self.metrics.optimized_tokens
        )
    
    @computed_field
    @property
    def compression_ratio(self) -> float:
        """Compression ratio (original / optimized)."""
        orig = self.metrics.original_tokens
        opt = self.metrics.optimized_tokens
        if not opt:
            return 0.0
        return orig / opt
    
    @computed_field
    @property
    def savings_percent(self) -> float:
        """Percentage of tokens saved."""
        orig = self.metrics.original_tokens
        opt = self.metrics.optimized_tokens
        if not orig:
            return 0.0
        return ((orig - opt) / orig) * 100
    
    @property
    def latency(self) -> int:
        return self.metrics.latency_ms
    
    def print_stats(self) -> None:
        """Print optimization statistics."""
        print(f"Optimization Stats:")
        print(f" - Tokens: {self.tokens[0]} -> {self.tokens[1]}")
        print(f" - Compression: {self.compression_ratio:.2f}x")
        print(f" - Savings: {self.savings_percent:.1f}%")
        print(f" - Chunks Retrieved: {self.metrics.chunks_retrieved}")
        print(f" - AST Fidelity: {self.metrics.ast_fidelity:.2f}")
        print(f" - Latency: {self.latency}ms")
    
    @classmethod
    def from_api_response(cls, content: str, raw_response: dict) -> 'OptimizedContext':
        """Create OptimizedContext from API response."""
        metrics_data = raw_response.get('metrics', raw_response)
        metrics = OptimizerMetrics(
            original_tokens=metrics_data.get('original_tokens', 0),
            optimized_tokens=metrics_data.get('optimized_tokens', 0),
            chunks_retrieved=metrics_data.get('chunks_retrieved', 0),
            compression_ratio=metrics_data.get('compression_ratio', 1.0),
            latency_ms=metrics_data.get('latency_ms', 0),
            retrieval_mode=metrics_data.get('retrieval_mode', 'hybrid'),
            ast_fidelity=metrics_data.get('ast_fidelity', 1.0)
        )
        return cls(content=content, metrics=metrics)
