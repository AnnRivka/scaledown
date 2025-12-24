from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime

class CompressionMetrics(BaseModel):
    """Validated metrics from the compression API."""
    original_prompt_tokens: int = 0
    compressed_prompt_tokens: int = 0
    latency_ms: int = 0
    timestamp: Optional[datetime] = None
    
    @field_validator('original_prompt_tokens', 'compressed_prompt_tokens', 'latency_ms')
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError('Must be non-negative')
        return v

class OptimizerMetrics(BaseModel):
    """Metrics from context optimization (e.g., HASTE)."""
    original_tokens: int = 0
    optimized_tokens: int = 0
    chunks_retrieved: int = 0
    compression_ratio: float = 1.0
    latency_ms: int = 0
    retrieval_mode: str = "hybrid"
    ast_fidelity: Optional[float] = None  # AST structure preservation score
    
    @field_validator('original_tokens', 'optimized_tokens', 'chunks_retrieved', 'latency_ms')
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError('Must be non-negative')
        return v
    
    @field_validator('compression_ratio', 'ast_fidelity')
    @classmethod
    def validate_ratio(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError('Must be non-negative')
        return v