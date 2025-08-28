#!/usr/bin/env python3
"""
Performance Benchmarking Script for WhaleSearch
Compares original vs optimized search engine performance
"""

import time
import statistics
import asyncio
from typing import List, Dict, Tuple
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whalesearch.src.search_engine import SearchEngine
from whalesearch.src.search_engine_optimized import OptimizedSearchEngine, CacheConfig

class PerformanceBenchmark:
    def __init__(self):
        self.test_queries = [
            "what is artificial intelligence",
            "explain machine learning basics",
            "how do neural networks work",
            "what is deep learning",
            "explain natural language processing",
            "what are the latest AI developments",
            "how does computer vision work",
            "explain reinforcement learning",
            "what is transfer learning",
            "how do transformers work in AI"
        ]
        
        # Initialize engines
        self.original_engine = SearchEngine()
        self.optimized_engine = OptimizedSearchEngine(
            CacheConfig(max_size=512, ttl_seconds=3600)
        )
        
    def benchmark_single_query(self, engine, query: str, is_image_search: bool = False) -> Dict:
        """Benchmark a single query"""
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            result = engine.search_and_answer(query, is_image_search)
            processing_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            return {
                "success": True,
                "processing_time": processing_time,
                "memory_delta": memory_after - memory_before,
                "result_length": len(result.get("answer", "")),
                "sources_count": len(result.get("sources", [])),
                "error": None
            }
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "success": False,
                "processing_time": processing_time,
                "memory_delta": 0,
                "result_length": 0,
                "sources_count": 0,
                "error": str(e)
            }
    
    def benchmark_engine(self, engine, engine_name: str, queries: List[str], 
                        is_image_search: bool = False) -> Dict:
        """Benchmark an entire engine with multiple queries"""
        print(f"\n🔍 Benchmarking {engine_name}...")
        
        results = []
        total_time = 0
        successful_queries = 0
        
        for i, query in enumerate(queries, 1):
            print(f"  Query {i}/{len(queries)}: {query[:50]}...")
            
            result = self.benchmark_single_query(engine, query, is_image_search)
            results.append(result)
            
            if result["success"]:
                successful_queries += 1
                total_time += result["processing_time"]
                print(f"    ✅ {result['processing_time']:.2f}s")
            else:
                print(f"    ❌ Error: {result['error']}")
        
        # Calculate statistics
        processing_times = [r["processing_time"] for r in results if r["success"]]
        memory_deltas = [r["memory_delta"] for r in results if r["success"]]
        
        if processing_times:
            stats = {
                "engine_name": engine_name,
                "total_queries": len(queries),
                "successful_queries": successful_queries,
                "success_rate": successful_queries / len(queries),
                "total_time": total_time,
                "avg_processing_time": statistics.mean(processing_times),
                "median_processing_time": statistics.median(processing_times),
                "min_processing_time": min(processing_times),
                "max_processing_time": max(processing_times),
                "std_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
                "avg_memory_delta": statistics.mean(memory_deltas) if memory_deltas else 0,
                "total_memory_delta": sum(memory_deltas),
                "queries_per_second": successful_queries / total_time if total_time > 0 else 0
            }
        else:
            stats = {
                "engine_name": engine_name,
                "total_queries": len(queries),
                "successful_queries": 0,
                "success_rate": 0,
                "total_time": 0,
                "avg_processing_time": 0,
                "median_processing_time": 0,
                "min_processing_time": 0,
                "max_processing_time": 0,
                "std_processing_time": 0,
                "avg_memory_delta": 0,
                "total_memory_delta": 0,
                "queries_per_second": 0
            }
        
        return stats
    
    def benchmark_cache_performance(self, engine, query: str, iterations: int = 5) -> Dict:
        """Benchmark cache performance with repeated queries"""
        print(f"\n🔄 Benchmarking cache performance for: {query[:50]}...")
        
        times = []
        for i in range(iterations):
            start_time = time.time()
            try:
                result = engine.search_and_answer(query, False)
                processing_time = time.time() - start_time
                times.append(processing_time)
                print(f"  Iteration {i+1}: {processing_time:.2f}s")
            except Exception as e:
                print(f"  Iteration {i+1}: Error - {e}")
                times.append(float('inf'))
        
        # Calculate cache improvement
        if len(times) >= 2:
            first_query_time = times[0]
            avg_cached_time = statistics.mean(times[1:])
            cache_improvement = ((first_query_time - avg_cached_time) / first_query_time) * 100
            
            return {
                "first_query_time": first_query_time,
                "avg_cached_time": avg_cached_time,
                "cache_improvement_percent": cache_improvement,
                "all_times": times
            }
        else:
            return {"error": "Insufficient data for cache analysis"}
    
    def benchmark_batch_processing(self, engine, queries: List[str]) -> Dict:
        """Benchmark batch processing performance"""
        print(f"\n🚀 Benchmarking batch processing...")
        
        # Individual processing
        start_time = time.time()
        individual_results = []
        for query in queries:
            try:
                result = engine.search_and_answer(query, False)
                individual_results.append(result)
            except Exception as e:
                individual_results.append({"error": str(e)})
        individual_time = time.time() - start_time
        
        # Batch processing (if available)
        batch_time = float('inf')
        batch_results = []
        if hasattr(engine, 'batch_search'):
            start_time = time.time()
            try:
                batch_results = engine.batch_search(queries, False)
                batch_time = time.time() - start_time
            except Exception as e:
                print(f"  Batch processing error: {e}")
        
        return {
            "individual_time": individual_time,
            "batch_time": batch_time,
            "speedup": individual_time / batch_time if batch_time < float('inf') else 1,
            "individual_results_count": len(individual_results),
            "batch_results_count": len(batch_results)
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def run_full_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        print("🚀 Starting WhaleSearch Performance Benchmark")
        print("=" * 60)
        
        # Set API key for testing
        test_api_key = os.getenv("SERPAPI_API_KEY", "test_key")
        os.environ["SERPAPI_API_KEY"] = test_api_key
        
        benchmark_results = {}
        
        # Test with subset of queries for faster benchmarking
        test_subset = self.test_queries[:5]
        
        # Benchmark original engine
        original_stats = self.benchmark_engine(
            self.original_engine, 
            "Original Search Engine", 
            test_subset
        )
        benchmark_results["original"] = original_stats
        
        # Benchmark optimized engine
        optimized_stats = self.benchmark_engine(
            self.optimized_engine, 
            "Optimized Search Engine", 
            test_subset
        )
        benchmark_results["optimized"] = optimized_stats
        
        # Cache performance test
        cache_stats = self.benchmark_cache_performance(
            self.optimized_engine, 
            self.test_queries[0]
        )
        benchmark_results["cache_performance"] = cache_stats
        
        # Batch processing test
        batch_stats = self.benchmark_batch_processing(
            self.optimized_engine, 
            test_subset
        )
        benchmark_results["batch_processing"] = batch_stats
        
        # Performance comparison
        if original_stats["successful_queries"] > 0 and optimized_stats["successful_queries"] > 0:
            speedup = original_stats["avg_processing_time"] / optimized_stats["avg_processing_time"]
            benchmark_results["comparison"] = {
                "speedup_factor": speedup,
                "time_improvement_percent": ((original_stats["avg_processing_time"] - optimized_stats["avg_processing_time"]) / original_stats["avg_processing_time"]) * 100,
                "memory_efficiency": original_stats["total_memory_delta"] / max(optimized_stats["total_memory_delta"], 1)
            }
        
        return benchmark_results
    
    def print_results(self, results: Dict):
        """Print benchmark results in a formatted way"""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)
        
        # Original vs Optimized comparison
        if "original" in results and "optimized" in results:
            orig = results["original"]
            opt = results["optimized"]
            
            print(f"\n🏁 ENGINE COMPARISON")
            print(f"{'Metric':<30} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
            print("-" * 75)
            print(f"{'Avg Processing Time':<30} {orig['avg_processing_time']:<15.2f}s {opt['avg_processing_time']:<15.2f}s {((orig['avg_processing_time'] - opt['avg_processing_time']) / orig['avg_processing_time'] * 100):<15.1f}%")
            print(f"{'Success Rate':<30} {orig['success_rate']:<15.1%} {opt['success_rate']:<15.1%} {((opt['success_rate'] - orig['success_rate']) / orig['success_rate'] * 100):<15.1f}%")
            print(f"{'Queries/Second':<30} {orig['queries_per_second']:<15.2f} {opt['queries_per_second']:<15.2f} {((opt['queries_per_second'] - orig['queries_per_second']) / orig['queries_per_second'] * 100):<15.1f}%")
        
        # Cache performance
        if "cache_performance" in results:
            cache = results["cache_performance"]
            if "cache_improvement_percent" in cache:
                print(f"\n🔄 CACHE PERFORMANCE")
                print(f"First Query Time: {cache['first_query_time']:.2f}s")
                print(f"Average Cached Time: {cache['avg_cached_time']:.2f}s")
                print(f"Cache Improvement: {cache['cache_improvement_percent']:.1f}%")
        
        # Batch processing
        if "batch_processing" in results:
            batch = results["batch_processing"]
            print(f"\n🚀 BATCH PROCESSING")
            print(f"Individual Processing: {batch['individual_time']:.2f}s")
            print(f"Batch Processing: {batch['batch_time']:.2f}s")
            if batch['speedup'] > 1:
                print(f"Speedup: {batch['speedup']:.2f}x")
        
        # Overall comparison
        if "comparison" in results:
            comp = results["comparison"]
            print(f"\n📈 OVERALL IMPROVEMENT")
            print(f"Speedup Factor: {comp['speedup_factor']:.2f}x")
            print(f"Time Improvement: {comp['time_improvement_percent']:.1f}%")
            print(f"Memory Efficiency: {comp['memory_efficiency']:.2f}x")
        
        print("\n" + "=" * 60)

def main():
    """Main benchmark execution"""
    try:
        benchmark = PerformanceBenchmark()
        results = benchmark.run_full_benchmark()
        benchmark.print_results(results)
        
        # Save results to file
        import json
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to benchmark_results.json")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()