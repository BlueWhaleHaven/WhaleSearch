# Optimized UI with performance improvements
from typing import Tuple, List
import gradio as gr
import os
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from whalesearch.src.search_engine_optimized import OptimizedSearchEngine, CacheConfig

class OptimizedSearchEngineUI:
    def __init__(self):
        # Initialize with optimized cache configuration
        cache_config = CacheConfig(
            max_size=256,  # Increased cache size
            ttl_seconds=7200  # 2 hour TTL
        )
        self.search_engine = OptimizedSearchEngine(cache_config)
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._processing_lock = threading.Lock()
        
    def process_query(self, query: str, api_key: str, is_image_search: bool) -> Tuple[str, str]:
        """Process query with performance monitoring"""
        os.environ["SERPAPI_API_KEY"] = api_key
        
        # Check if already processing
        if self._processing_lock.locked():
            yield "Another query is being processed. Please wait...", "", None
            return
        
        with self._processing_lock:
            start_time = time.time()
            yield "Processing your query...", "", None
            
            try:
                # Process query asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Run in thread pool to avoid blocking
                future = self._executor.submit(
                    self.search_engine.search_and_answer, 
                    query, 
                    is_image_search
                )
                
                results = future.result(timeout=120)  # 2 minute timeout
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Format sources
                sources_text = f"\n\nSources (processed in {processing_time:.2f}s):\n"
                gallery_images = []
                
                for idx, source in enumerate(results["sources"], 1):
                    if results["is_image_search"]:
                        sources_text += f"{idx}. [{source['title']}]({source['url']}) - [View Image]({source['image_url']})\n"
                        if source['image_url']:
                            gallery_images.append(source['image_url'])
                    else:
                        sources_text += f"{idx}. [{source['title']}]({source['url']})\n"
                
                # Add performance metrics
                perf_stats = self.search_engine.get_performance_stats()
                performance_info = f"\n\n--- Performance Metrics ---\n"
                performance_info += f"Cache Size: {perf_stats['cache_size']}\n"
                performance_info += f"Cache Hit Rate: {perf_stats['cache_hit_rate']:.2%}\n"
                performance_info += f"Processing Time: {processing_time:.2f}s"
                
                sources_text += performance_info
                
                yield results["answer"], sources_text, gr.Gallery(value=gallery_images) if gallery_images else None
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                yield error_msg, f"Error: {error_msg}", None
    
    def batch_process_queries(self, queries: List[str], api_key: str, is_image_search: bool) -> Tuple[str, str]:
        """Process multiple queries in batch for better performance"""
        os.environ["SERPAPI_API_KEY"] = api_key
        
        if not queries or not queries[0].strip():
            yield "Please provide at least one query.", "", None
            return
        
        yield "Processing batch queries...", "", None
        
        try:
            # Filter out empty queries
            valid_queries = [q.strip() for q in queries if q.strip()]
            
            if not valid_queries:
                yield "No valid queries found.", "", None
                return
            
            # Process in batch
            results = self.search_engine.batch_search(valid_queries, is_image_search)
            
            # Format batch results
            batch_answer = "## Batch Search Results\n\n"
            batch_sources = "## All Sources\n\n"
            
            for i, (query, result) in enumerate(zip(valid_queries, results)):
                batch_answer += f"### Query {i+1}: {query}\n\n"
                batch_answer += f"{result['answer']}\n\n"
                batch_answer += f"Processing time: {result.get('processing_time', 0):.2f}s\n\n"
                batch_answer += "---\n\n"
                
                # Collect sources
                for idx, source in enumerate(result["sources"], 1):
                    if result["is_image_search"]:
                        batch_sources += f"{i+1}.{idx}. [{source['title']}]({source['url']}) - [View Image]({source['image_url']})\n"
                    else:
                        batch_sources += f"{i+1}.{idx}. [{source['title']}]({source['url']})\n"
            
            # Add performance summary
            perf_stats = self.search_engine.get_performance_stats()
            batch_sources += f"\n\n--- Batch Performance ---\n"
            batch_sources += f"Total Queries: {len(valid_queries)}\n"
            batch_sources += f"Cache Size: {perf_stats['cache_size']}\n"
            batch_sources += f"Cache Hit Rate: {perf_stats['cache_hit_rate']:.2%}"
            
            yield batch_answer, batch_sources, None
            
        except Exception as e:
            error_msg = f"Error processing batch queries: {str(e)}"
            yield error_msg, f"Error: {error_msg}", None
    
    def clear_cache(self) -> str:
        """Clear the search engine cache"""
        try:
            # Reset the search engine to clear cache
            cache_config = CacheConfig(max_size=256, ttl_seconds=7200)
            self.search_engine = OptimizedSearchEngine(cache_config)
            return "Cache cleared successfully!"
        except Exception as e:
            return f"Error clearing cache: {str(e)}"
    
    def get_performance_stats(self) -> str:
        """Get current performance statistics"""
        try:
            stats = self.search_engine.get_performance_stats()
            stats_text = "## Performance Statistics\n\n"
            stats_text += f"**Cache Size:** {stats['cache_size']}\n"
            stats_text += f"**Cache Hit Rate:** {stats['cache_hit_rate']:.2%}\n"
            stats_text += f"**Initialized:** {stats['initialized']}\n"
            stats_text += f"**Thread Pool Size:** {stats['thread_pool_size']}\n"
            return stats_text
        except Exception as e:
            return f"Error getting stats: {str(e)}"
    
    def create_interface(self):
        """Create optimized Gradio interface"""
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="WhaleSearch 🐳 - Optimized",
            css="""
            .performance-metrics { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                border-radius: 10px;
                margin: 10px 0;
            }
            .cache-info {
                background: #f8f9fa;
                border-left: 4px solid #007bff;
                padding: 10px;
                margin: 10px 0;
            }
            """
        ) as interface:
            gr.Markdown("# WhaleSearch 🐳 - Performance Optimized", elem_id="app-title")
            
            # Performance metrics display
            with gr.Row():
                with gr.Column():
                    perf_stats_btn = gr.Button("📊 Show Performance Stats", variant="secondary")
                    clear_cache_btn = gr.Button("🗑️ Clear Cache", variant="secondary")
                with gr.Column():
                    perf_display = gr.Markdown("Click 'Show Performance Stats' to see current metrics")
            
            # Main search interface
            with gr.Row():
                with gr.Column():
                    query_input = gr.Textbox(
                        label="Explore",
                        placeholder="Enter your search query here...",
                        lines=2
                    )
                    api_key = gr.Textbox(
                        label="SerpAPI Key",
                        placeholder="Enter your SerpAPI key",
                        type="password"
                    )
                    is_img_query = gr.Checkbox(label="Image Search")
                    search_button = gr.Button("🔍 Search", variant="primary")
            
            # Batch search interface
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Batch Search (Multiple Queries)")
                    batch_queries = gr.Textbox(
                        label="Batch Queries",
                        placeholder="Enter multiple queries, one per line...",
                        lines=5
                    )
                    batch_search_btn = gr.Button("🚀 Batch Search", variant="primary")
            
            # Results display
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Markdown(label="Answer")
                    sources_output = gr.Markdown(label="Sources")
                    image_gallery = gr.Gallery(label="Image Results", visible=False)
            
            # Gallery visibility toggle
            def update_gallery_visibility(is_image):
                return gr.Gallery(visible=is_image)
            
            is_img_query.change(
                fn=update_gallery_visibility,
                inputs=[is_img_query],
                outputs=[image_gallery]
            )
            
            # Event handlers
            search_button.click(
                fn=self.process_query,
                inputs=[query_input, api_key, is_img_query],
                outputs=[answer_output, sources_output, image_gallery],
                show_progress=True
            )
            
            batch_search_btn.click(
                fn=self.batch_process_queries,
                inputs=[batch_queries, api_key, is_img_query],
                outputs=[answer_output, sources_output, image_gallery],
                show_progress=True
            )
            
            perf_stats_btn.click(
                fn=self.get_performance_stats,
                outputs=[perf_display]
            )
            
            clear_cache_btn.click(
                fn=self.clear_cache,
                outputs=[perf_display]
            )
            
            # Examples
            gr.Examples(
                examples=[
                    ["what are the latest developments in octopus recipes?"],
                    ["show me pictures of rare deep sea creatures"],
                    ["what is the current state of internally achieved AGI?"],
                    ["explain quantum computing basics", "what is machine learning", "how does neural networks work"]
                ],
                inputs=[query_input, batch_queries]
            )
            
            # Performance tips
            gr.Markdown("""
            ### 🚀 Performance Tips:
            - **Caching**: Results are cached for 2 hours to improve response times
            - **Batch Processing**: Use batch search for multiple related queries
            - **Connection Pooling**: Optimized thread management for better throughput
            - **Lazy Loading**: Models are loaded only when needed
            
            ### 📋 Notes:
            - You need a valid SerpAPI key to use this search engine
            - Results are cached to improve performance on repeated queries
            - Batch search processes multiple queries in parallel
            - Performance metrics show cache efficiency and processing times
            """)
        
        return interface

# Initialize and launch
if __name__ == "__main__":
    ui = OptimizedSearchEngineUI()
    interface = ui.create_interface()
    
    # Launch with optimized settings
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        # Performance optimizations
        max_threads=40,
        show_tips=False,
        quiet=True
    )