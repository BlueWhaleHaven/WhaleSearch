import os
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from contextlib import contextmanager

from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from serpapi import GoogleSearch

@dataclass
class CacheConfig:
    max_size: int = 128
    ttl_seconds: int = 3600  # 1 hour

class OptimizedSearchEngine:
    def __init__(self, cache_config: CacheConfig = None):
        self.cache_config = cache_config or CacheConfig()
        self._llm = None
        self._embeddings = None
        self._vector_db = None
        self._rag_chain = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._cache = {}
        self._cache_timestamps = {}
        
        # Initialize components lazily
        self._initialized = False
        
    def _ensure_initialized(self):
        """Lazy initialization of components"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._llm = self._init_llm()
                    self._embeddings = self._init_embeddings()
                    self._initialized = True
    
    @property
    def llm(self):
        self._ensure_initialized()
        return self._llm
    
    @property
    def embeddings(self):
        self._ensure_initialized()
        return self._embeddings
    
    def _init_llm(self):
        """Initialize LLM with optimized settings"""
        return LlamaCpp(
            model_path="Phi-3-mini-4k-instruct-fp16.gguf",
            n_gpu_layers=-1,
            max_tokens=500,
            n_ctx=2048,
            seed=42,
            verbose=False,
            # Performance optimizations
            n_batch=512,
            n_threads=os.cpu_count(),
            use_mmap=True,
            use_mlock=False
        )
    
    def _init_embeddings(self):
        """Initialize embeddings with caching"""
        return HuggingFaceEmbeddings(
            model_name='thenlper/gte-small',
            model_kwargs={'device': 'cpu'},  # Force CPU for consistency
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _get_cache_key(self, query: str, is_image_search: bool) -> str:
        """Generate cache key for query results"""
        return hashlib.md5(f"{query}_{is_image_search}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[cache_key] < self.cache_config.ttl_seconds
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_timestamps.items()
            if current_time - timestamp > self.cache_config.ttl_seconds
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
    
    def _add_to_cache(self, cache_key: str, result: Dict):
        """Add result to cache with size management"""
        self._cleanup_cache()
        
        # Remove oldest entries if cache is full
        if len(self._cache) >= self.cache_config.max_size:
            oldest_key = min(self._cache_timestamps.keys(), 
                           key=lambda k: self._cache_timestamps[k])
            self._cache.pop(oldest_key, None)
            self._cache_timestamps.pop(oldest_key, None)
        
        self._cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()
    
    @lru_cache(maxsize=32)
    def _get_serp_results_cached(self, query: str, is_image_search: bool, num_results: int) -> List[Document]:
        """Cached version of SERP API calls"""
        try:
            params = {
                "engine": "google_images" if is_image_search else "google",
                "q": query,
                "api_key": os.getenv("SERPAPI_API_KEY"),
                "num": num_results
            }
            
            search = GoogleSearch(params)
            results = search.get_dict()
            
            documents = []
            if is_image_search and "images_results" in results:
                for result in results["images_results"]:
                    doc = Document(
                        page_content=f"Title: {result.get('title', '')}\n\nDescription: {result.get('snippet', '')}\n\nContent: Image URL: {result.get('original', '')}",
                        metadata={
                            "source": result.get("source", ""),
                            "title": result.get("title", ""),
                            "position": result.get("position", 0),
                            "image_url": result.get("original", ""),
                            "thumbnail": result.get("thumbnail", "")
                        }
                    )
                    documents.append(doc)
            elif not is_image_search and "organic_results" in results:
                for result in results["organic_results"]:
                    doc = Document(
                        page_content=f"Title: {result.get('title', '')}\n\nSnippet: {result.get('snippet', '')}\n\nContent: {result.get('content', '')}",
                        metadata={
                            "source": result.get("link", ""),
                            "title": result.get("title", ""),
                            "position": result.get("position", 0)
                        }
                    )
                    documents.append(doc)
                    
            if not documents:
                doc = Document(
                    page_content=f"No results found for query: {query}",
                    metadata={
                        "source": "",
                        "title": "No Results",
                        "position": 0
                    }
                )
                documents.append(doc)
                
            return documents
            
        except Exception as e:
            doc = Document(
                page_content=f"Error occurred while searching: {str(e)}",
                metadata={
                    "source": "",
                    "title": "Error",
                    "position": 0
                }
            )
            return [doc]
    
    def get_serp_results(self, query: str, is_image_search: bool = False, num_results: int = 5) -> List[Document]:
        """Get search results with caching"""
        return self._get_serp_results_cached(query, is_image_search, num_results)
    
    def create_vector_db(self, documents: List[Document]) -> Optional[Chroma]:
        """Create vector database with optimized text splitting"""
        if not documents:
            return None
            
        # Optimized text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased chunk size for better context
            chunk_overlap=100,  # Increased overlap for better continuity
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            return None
            
        # Use persistent directory for better performance
        return Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory="./chroma_db_optimized"
        )
    
    def setup_rag(self, db: Optional[Chroma], is_image_search: bool = False) -> Optional[RetrievalQA]:
        """Setup RAG chain with optimized prompts"""
        if db is None:
            return None
            
        if is_image_search:
            template = """<|user|>
            Relevant information including images:
            {context}
            Question: {question}
            
            Please provide a detailed answer using the above information. Include relevant citations and image descriptions where available.
            <|end|>
            <|assistant|>"""
        else:
            template = """<|user|>
            Relevant information:
            {context}
            Question: {question}
            
            Please provide a detailed answer using the above information. Include relevant citations.
            <|end|>
            <|assistant|>"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def search_and_answer(self, query: str, is_image_search: bool = False) -> Dict:
        """Main search and answer method with caching and optimization"""
        # Check cache first
        cache_key = self._get_cache_key(query, is_image_search)
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        # Process query
        start_time = time.time()
        
        # Get search results
        search_results = self.get_serp_results(query, is_image_search)
        
        if not search_results:
            result = {
                "answer": "No results found for your query.",
                "sources": [],
                "is_image_search": is_image_search,
                "processing_time": time.time() - start_time
            }
            self._add_to_cache(cache_key, result)
            return result
        
        # Create vector database
        db = self.create_vector_db(search_results)
        if db is None:
            result = {
                "answer": "Unable to process search results.",
                "sources": [],
                "is_image_search": is_image_search,
                "processing_time": time.time() - start_time
            }
            self._add_to_cache(cache_key, result)
            return result
            
        # Setup RAG chain
        rag_chain = self.setup_rag(db, is_image_search)
        if rag_chain is None:
            result = {
                "answer": "Unable to generate response.",
                "sources": [],
                "is_image_search": is_image_search,
                "processing_time": time.time() - start_time
            }
            self._add_to_cache(cache_key, result)
            return result
            
        try:
            # Process with RAG
            response = rag_chain.invoke(query)
            
            sources = []
            for doc in response["source_documents"]:
                source_info = {
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("source", ""),
                    "position": doc.metadata.get("position", 0)
                }
                
                if is_image_search:
                    source_info["image_url"] = doc.metadata.get("image_url", "")
                    source_info["thumbnail"] = doc.metadata.get("thumbnail", "")
                
                sources.append(source_info)
            
            result = {
                "answer": response["result"],
                "sources": sources,
                "is_image_search": is_image_search,
                "processing_time": time.time() - start_time
            }
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            return result
            
        except Exception as e:
            result = {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "is_image_search": is_image_search,
                "processing_time": time.time() - start_time
            }
            self._add_to_cache(cache_key, result)
            return result
    
    def batch_search(self, queries: List[str], is_image_search: bool = False) -> List[Dict]:
        """Process multiple queries in parallel"""
        futures = []
        for query in queries:
            future = self._executor.submit(self.search_and_answer, query, is_image_search)
            futures.append((query, future))
        
        results = []
        for query, future in futures:
            try:
                result = future.result(timeout=60)  # 60 second timeout
                results.append(result)
            except Exception as e:
                results.append({
                    "answer": f"Error processing query '{query}': {str(e)}",
                    "sources": [],
                    "is_image_search": is_image_search,
                    "processing_time": 0
                })
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "cache_size": len(self._cache),
            "cache_hit_rate": len([k for k in self._cache.keys() if self._is_cache_valid(k)]) / max(len(self._cache), 1),
            "initialized": self._initialized,
            "thread_pool_size": self._executor._max_workers
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if self._executor:
            self._executor.shutdown(wait=True)
        if self._vector_db:
            self._vector_db.persist()
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()