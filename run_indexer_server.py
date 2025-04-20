#!/usr/bin/env python3
"""
Background indexer server using thread approach

This script demonstrates how to run the indexer in a background thread,
sharing the same RocksRepo instance with the main process.

Usage:
  python run_indexer_server.py [--db_path=./test] [--partial_interval=1] [--full_interval=5]

Options:
  --db_path=PATH           Path to the database directory (default: ./test)
  --partial_interval=SECS  Seconds between partial index operations (default: 1)
  --full_interval=SECS     Seconds between full index operations (default: 5)
"""

import sys
import time
import argparse
import signal
from pathlib import Path
from percolate.models import Agent
from naturaldb.repo import RocksRepo

def parse_args():
    parser = argparse.ArgumentParser(description="Run the background indexer thread")
    parser.add_argument("--db_path", type=str, default="./test",
                        help="Path to the database directory")
    parser.add_argument("--partial_interval", type=int, default=1,
                        help="Seconds between partial index operations")
    parser.add_argument("--full_interval", type=int, default=5,
                        help="Seconds between full index operations")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting indexer with db_path={args.db_path}, "
          f"partial_interval={args.partial_interval}s, "
          f"full_interval={args.full_interval}s")
    
    # Create repository with background thread enabled and configured intervals
    repo = RocksRepo(
        model_cls=Agent, 
        db_path=args.db_path, 
        use_background_thread=True,
        partial_index_interval=args.partial_interval,
        full_index_interval=args.full_interval
    )
    
    # Setup signal handler to gracefully exit
    def signal_handler(sig, frame):
        print("\nStopping indexer...")
        repo.index._stop_background_thread()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep the main thread alive
    try:
        print("Indexer is running in background thread. Press Ctrl+C to stop.")
        # In a real application, you would use the repo here for DB operations
        # while the background thread handles indexing
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nIndexer stopped by user")
        repo.index._stop_background_thread()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())