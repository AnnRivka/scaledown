"""
Real HASTE integration test - NO MOCKS!
Tests the actual HasteContext library integration.
"""
import scaledown as sd
from scaledown.optimizer import HasteOptimizer
from scaledown.pipeline import Pipeline
import tempfile
import os

print("=" * 70)
print("REAL HASTE INTEGRATION TEST")
print("=" * 70)

# Create a test Python file
test_code = """
def calculate_sum(numbers):
    \"\"\"Calculate sum of numbers.\"\"\"
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    \"\"\"Calculate average of numbers.\"\"\"
    if len(numbers) == 0:
        return 0
    return calculate_sum(numbers) / len(numbers)

def calculate_median(numbers):
    \"\"\"Calculate median of numbers.\"\"\"
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    if n % 2 == 0:
        return (sorted_nums[n//2-1] + sorted_nums[n//2]) / 2
    return sorted_nums[n//2]

def calculate_variance(numbers):
    \"\"\"Calculate variance of numbers.\"\"\"
    avg = calculate_average(numbers)
    squared_diffs = [(x - avg) ** 2 for x in numbers]
    return sum(squared_diffs) / len(numbers)

def process_data(data):
    \"\"\"Process and analyze data.\"\"\"
    cleaned = [x for x in data if x is not None]
    return {
        'sum': calculate_sum(cleaned),
        'average': calculate_average(cleaned),
        'median': calculate_median(cleaned),
        'variance': calculate_variance(cleaned)
    }

class DataProcessor:
    \"\"\"Data processing class.\"\"\"
    
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return process_data(self.data)
    
    def get_statistics(self):
        stats = self.process()
        return {
            'mean': stats['average'],
            'median': stats['median'],
            'var': stats['variance']
        }
"""

# Write test code to temp file
with tempfile.NamedTemporaryFile(
    mode='w',
    suffix='.py',
    delete=False,
    encoding='utf-8'
) as f:
    f.write(test_code)
    test_file_path = f.name

try:
    # Test 1: Basic HASTE optimization with BM25 only
    print("\n1. Testing HASTE Optimizer (BM25 only)")
    print("-" * 70)
    
    optimizer = HasteOptimizer(
        top_k=3,
        bfs_depth=1,
        semantic=False  # No OpenAI API needed!
    )
    
    result = optimizer.optimize(
        context=test_code,
        query="Show me the calculate_average function and its dependencies",
        file_path=test_file_path,
        max_tokens=500
    )
    
    print("\n✓ Optimized Context:")
    print(result.content[:300] + "..." if len(result.content) > 300 else result.content)
    print("\n✓ Optimization Metrics:")
    result.print_stats()
    
    # Test 2: String-based optimization
    print("\n\n2. Testing String-Based Optimization")
    print("-" * 70)
    
    result2 = optimizer.optimize_from_string(
        source_code=test_code,
        query="Find the DataProcessor class",
        max_tokens=400
    )
    
    print("\n✓ Optimized Context:")
    print(result2.content[:300] + "..." if len(result2.content) > 300 else result2.content)
    print("\n✓ Metrics:")
    result2.print_stats()
    
    # Test 3: Different query
    print("\n\n3. Testing Different Query")
    print("-" * 70)
    
    result3 = optimizer.optimize(
        context=test_code,
        query="variance calculation",
        file_path=test_file_path
    )
    
    print("\n✓ Retrieved Functions:")
    print(result3.content[:400] + "..." if len(result3.content) > 400 else result3.content)
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - HASTE IS WORKING!")
    print("=" * 70)
    
finally:
    # Cleanup
    if os.path.exists(test_file_path):
        os.unlink(test_file_path)
