from typing import List, Tuple, Union, Optional
from .optimizer.base import BaseOptimizer
from .compressor.base import BaseCompressor
from .types import OptimizedContext, CompressedPrompt


class Pipeline:
    """
    Pipeline for chaining optimizers and compressors.
    
    Example
    -------
    >>> from scaledown.pipeline import Pipeline
    >>> from scaledown.optimizer import HasteOptimizer
    >>> from scaledown.compressor import ScaleDownCompressor
    >>> 
    >>> pipe = Pipeline([
    ...     ('haste', HasteOptimizer()),
    ...     ('compressor', ScaleDownCompressor(model="gpt-4o"))
    ... ])
    >>> 
    >>> result = pipe.run(context=code, query="Add type hints", prompt="Explain changes")
    """
    
    def __init__(self, steps: List[Tuple[str, Union[BaseOptimizer, BaseCompressor]]]):
        """
        Initialize pipeline with ordered steps.
        
        Parameters
        ----------
        steps : List[Tuple[str, Union[BaseOptimizer, BaseCompressor]]]
            List of (name, transformer) tuples
        """
        self.steps = steps
        self._validate_steps()
    
    def _validate_steps(self):
        """Validate pipeline structure."""
        if not self.steps:
            raise ValueError("Pipeline must have at least one step")
        
        # Check that optimizers come before compressors
        seen_compressor = False
        for name, step in self.steps:
            if isinstance(step, BaseCompressor):
                seen_compressor = True
            elif isinstance(step, BaseOptimizer) and seen_compressor:
                raise ValueError(
                    f"Optimizer '{name}' cannot come after a compressor. "
                    "Pipeline order must be: optimizers -> compressors"
                )
    
    def run(
        self,
        context: Union[str, List[str]],
        query: Optional[str] = None,
        prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[OptimizedContext, CompressedPrompt]:
        """
        Run the full pipeline.
        
        Parameters
        ----------
        context : str or List[str]
            Input context (code, documents, etc.)
        query : str, optional
            Query for optimizers
        prompt : str, optional
            Prompt for compressors
        max_tokens : int, optional
            Token budget
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        OptimizedContext or CompressedPrompt
            Final output from the last step
        """
        current_output = context
        
        for name, step in self.steps:
            if isinstance(step, BaseOptimizer):
                # Optimizer step
                current_output = step.optimize(
                    context=current_output if isinstance(current_output, (str, list)) else str(current_output),
                    query=query,
                    max_tokens=max_tokens,
                    **kwargs
                )
            elif isinstance(step, BaseCompressor):
                # Compressor step
                if isinstance(current_output, OptimizedContext):
                    context_str = str(current_output)
                else:
                    context_str = current_output
                
                current_output = step.compress(
                    context=context_str,
                    prompt=prompt or query or "",
                    max_tokens=max_tokens,
                    **kwargs
                )
        
        return current_output
    
    def get_step(self, name: str) -> Union[BaseOptimizer, BaseCompressor]:
        """Get a step by name."""
        for step_name, step in self.steps:
            if step_name == name:
                return step
        raise KeyError(f"Step '{name}' not found in pipeline")
    
    def __repr__(self) -> str:
        step_names = [name for name, _ in self.steps]
        return f"Pipeline(steps={step_names})"


def make_pipeline(*steps) -> Pipeline:
    """
    Helper function to create a pipeline.
    
    Parameters
    ----------
    *steps : tuples
        Variable number of (name, transformer) tuples
        
    Returns
    -------
    Pipeline
        Configured pipeline
        
    Example
    -------
    >>> pipe = make_pipeline(
    ...     ('haste', HasteOptimizer()),
    ...     ('compress', ScaleDownCompressor())
    ... )
    """
    return Pipeline(list(steps))
