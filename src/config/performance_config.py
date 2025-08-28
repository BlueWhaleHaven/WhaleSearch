"""
Performance Configuration for WhaleSearch
Centralized configuration for all performance-related settings
"""

import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class LLMConfig:
    """LLM performance configuration"""
    model_path: str = "Phi-3-mini-4k-instruct-fp16.gguf"
    n_gpu_layers: int = -1
    max_tokens: int = 500
    n_ctx: int = 2048
    seed: int = 42
    verbose: bool = False
    n_batch: int = 512
    n_threads: int = os.cpu_count() or 4
    use_mmap: bool = True
    use_mlock: bool = False
    f16_kv: bool = True
    rope_freq_base: float = 10000.0
    rope_freq_scale: float = 1.0

@dataclass
class EmbeddingsConfig:
    """Embeddings performance configuration"""
    model_name: str = 'thenlper/gte-small'
    device: str = 'cpu'  # 'cpu' or 'cuda'
    normalize_embeddings: bool = True
    batch_size: int = 32
    max_length: int = 512

@dataclass
class CacheConfig:
    """Cache performance configuration"""
    max_size: int = 256
    ttl_seconds: int = 7200  # 2 hours
    enable_disk_cache: bool = True
    disk_cache_path: str = "./cache"
    enable_redis: bool = False
    redis_url: str = "redis://localhost:6379"
    compression: bool = True

@dataclass
class VectorDBConfig:
    """Vector database performance configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 100
    persist_directory: str = "./chroma_db_optimized"
    collection_name: str = "whalesearch_docs"
    distance_metric: str = "cosine"
    n_results: int = 3
    enable_hnsw: bool = True
    hnsw_m: int = 16
    hnsw_ef_construction: int = 100

@dataclass
class ThreadingConfig:
    """Threading and concurrency configuration"""
    max_workers: int = 4
    thread_pool_size: int = 8
    enable_async: bool = True
    max_concurrent_requests: int = 10
    request_timeout: int = 120

@dataclass
class APIConfig:
    """API and external service configuration"""
    serpapi_timeout: int = 30
    serpapi_retries: int = 3
    enable_rate_limiting: bool = True
    requests_per_minute: int = 60
    batch_size: int = 5

@dataclass
class MonitoringConfig:
    """Performance monitoring configuration"""
    enable_metrics: bool = True
    metrics_port: int = 8000
    enable_profiling: bool = False
    profile_output: str = "./profiles"
    log_level: str = "INFO"
    enable_tracing: bool = False

@dataclass
class PerformanceConfig:
    """Main performance configuration class"""
    llm: LLMConfig = None
    embeddings: EmbeddingsConfig = None
    cache: CacheConfig = None
    vector_db: VectorDBConfig = None
    threading: ThreadingConfig = None
    api: APIConfig = None
    monitoring: MonitoringConfig = None
    
    def __post_init__(self):
        """Initialize default configurations if not provided"""
        if self.llm is None:
            self.llm = LLMConfig()
        if self.embeddings is None:
            self.embeddings = EmbeddingsConfig()
        if self.cache is None:
            self.cache = CacheConfig()
        if self.vector_db is None:
            self.vector_db = VectorDBConfig()
        if self.threading is None:
            self.threading = ThreadingConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
    
    @classmethod
    def from_env(cls) -> 'PerformanceConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # LLM config from env
        if os.getenv("LLM_N_BATCH"):
            config.llm.n_batch = int(os.getenv("LLM_N_BATCH"))
        if os.getenv("LLM_N_THREADS"):
            config.llm.n_threads = int(os.getenv("LLM_N_THREADS"))
        if os.getenv("LLM_USE_MMAP"):
            config.llm.use_mmap = os.getenv("LLM_USE_MMAP").lower() == "true"
        
        # Cache config from env
        if os.getenv("CACHE_MAX_SIZE"):
            config.cache.max_size = int(os.getenv("CACHE_MAX_SIZE"))
        if os.getenv("CACHE_TTL_HOURS"):
            config.cache.ttl_seconds = int(os.getenv("CACHE_TTL_HOURS")) * 3600
        
        # Threading config from env
        if os.getenv("MAX_WORKERS"):
            config.threading.max_workers = int(os.getenv("MAX_WORKERS"))
        if os.getenv("THREAD_POOL_SIZE"):
            config.threading.thread_pool_size = int(os.getenv("THREAD_POOL_SIZE"))
        
        # API config from env
        if os.getenv("SERPAPI_TIMEOUT"):
            config.api.serpapi_timeout = int(os.getenv("SERPAPI_TIMEOUT"))
        if os.getenv("REQUESTS_PER_MINUTE"):
            config.api.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE"))
        
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'PerformanceConfig':
        """Create configuration from JSON file"""
        import json
        try:
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            print(f"Warning: Could not load config from {file_path}: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "llm": self.llm.__dict__,
            "embeddings": self.embeddings.__dict__,
            "cache": self.cache.__dict__,
            "vector_db": self.vector_db.__dict__,
            "threading": self.threading.__dict__,
            "api": self.api.__dict__,
            "monitoring": self.monitoring.__dict__
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file"""
        import json
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def get_optimization_profile(self, profile: str) -> 'PerformanceConfig':
        """Get pre-configured optimization profiles"""
        profiles = {
            "development": PerformanceConfig(
                cache=CacheConfig(max_size=64, ttl_seconds=1800),
                threading=ThreadingConfig(max_workers=2, thread_pool_size=4),
                monitoring=MonitoringConfig(enable_metrics=True, log_level="DEBUG")
            ),
            "production": PerformanceConfig(
                cache=CacheConfig(max_size=1024, ttl_seconds=14400),
                threading=ThreadingConfig(max_workers=8, thread_pool_size=16),
                monitoring=MonitoringConfig(enable_metrics=True, log_level="WARNING")
            ),
            "high_performance": PerformanceConfig(
                cache=CacheConfig(max_size=2048, ttl_seconds=28800),
                threading=ThreadingConfig(max_workers=16, thread_pool_size=32),
                api=APIConfig(requests_per_minute=120, batch_size=10),
                monitoring=MonitoringConfig(enable_metrics=True, enable_profiling=True)
            ),
            "low_resource": PerformanceConfig(
                llm=LLMConfig(n_batch=256, n_threads=2),
                cache=CacheConfig(max_size=128, ttl_seconds=3600),
                threading=ThreadingConfig(max_workers=2, thread_pool_size=4),
                api=APIConfig(requests_per_minute=30, batch_size=3)
            )
        }
        
        return profiles.get(profile, self)

# Default configuration
DEFAULT_CONFIG = PerformanceConfig()

# Environment-specific configurations
def get_config() -> PerformanceConfig:
    """Get configuration based on environment"""
    env = os.getenv("WHALESEARCH_ENV", "development")
    
    if env == "production":
        return DEFAULT_CONFIG.get_optimization_profile("production")
    elif env == "high_performance":
        return DEFAULT_CONFIG.get_optimization_profile("high_performance")
    elif env == "low_resource":
        return DEFAULT_CONFIG.get_optimization_profile("low_resource")
    else:
        return DEFAULT_CONFIG.get_optimization_profile("development")