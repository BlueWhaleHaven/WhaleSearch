#!/usr/bin/env python3
"""
Run Optimized WhaleSearch with Performance Monitoring
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main entry point for optimized WhaleSearch"""
    print("🚀 Starting Optimized WhaleSearch...")
    print("=" * 50)
    
    try:
        # Import and run optimized app
        from app_optimized import OptimizedSearchEngineUI
        
        print("✅ Optimized components loaded successfully")
        print("🔧 Performance optimizations enabled:")
        print("   - Intelligent caching system")
        print("   - Lazy loading & resource management")
        print("   - Concurrent processing")
        print("   - Performance monitoring")
        print("   - Batch search capabilities")
        
        # Initialize UI
        ui = OptimizedSearchEngineUI()
        interface = ui.create_interface()
        
        print("\n🌐 Launching optimized interface...")
        print("📊 Performance metrics will be displayed in the UI")
        print("💡 Use 'Show Performance Stats' to monitor performance")
        print("🚀 Use 'Batch Search' for multiple queries")
        
        # Launch with optimized settings
        interface.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
            max_threads=40,
            show_tips=False,
            quiet=True
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the correct directory and all dependencies are installed")
        print("📦 Run: pip install -r requirements_optimized.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Error starting optimized WhaleSearch: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)