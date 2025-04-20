from pydantic import BaseModel
from pathlib import Path
import json
from collections import defaultdict
import numpy as np
from pathlib import Path
import hnswlib
import hashlib
from datetime import datetime
import typing
import traceback
import time
import fcntl
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

class Indexer:
    def __init__(self, repo, client_mode=True, use_background_thread=False, 
                 partial_index_interval:int=1, full_index_interval:int=5):
        """
        Initialize the indexer with the specified settings
        
        Args:
            repo: Repository instance to be indexed
            client_mode: Whether to operate in client mode (True) or server mode (False)
            use_background_thread: Whether to use a background thread for indexing
            partial_index_interval: Seconds between partial index operations
            full_index_interval: Seconds between full index operations
        """
        self.repo = repo  # Keep reference to the parent repository
        self.db = repo.db
        self._path = repo._path
        self.model_logic = repo.model_logic  # Access to model logic for key handling
        self.records: list[BaseModel] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures = []
        self.client_mode = client_mode  # Mode flag
        
        # Background thread processing
        self.use_background_thread = use_background_thread
        self.background_thread = None
        self.task_queue = queue.Queue()
        self.should_stop = threading.Event()
        
        # Store indexing interval configuration
        self.partial_index_interval = partial_index_interval
        self.full_index_interval = full_index_interval
        
        # Set keys for tracking indexing state in RocksDB
        self.index_state_key = b"index:state:last_full_index_time"
        
        # Track touched entities for efficient full indexing
        self.touched_entities = {}
        
        # Get last full index time from RocksDB if available
        self._initialize_index_state()
        
 
        # Define paths for different index types
        self.paths = {
            # Semantic index paths
            "semantic": lambda ns, name: Path(f"{self._path}/index/semantic/{ns}/{name}.bin"),
            # Predicate index paths (parquet files)
            "parquet": lambda ns, name: Path(f"{self._path}/index/structures/{ns}/{name}/data.parquet"),
            # Temporary parquet files for partial indexing
            "temp_parquet": lambda ns, name: Path(f"{self._path}/index/structures/{ns}/{name}/temp.parquet"),
            # Lock files for parquet file access
            "parquet_lock": lambda ns, name: Path(f"{self._path}/index/structures/{ns}/{name}.lock"),
            # Graph index paths
            "graph": lambda ns, name: Path(f"{self._path}/index/graph/{ns}/{name}.json"),
            # Base directories
            "semantic_dir": lambda ns, name=None: Path(f"{self._path}/index/semantic/{ns}"),
            "parquet_dir": lambda ns, name=None: Path(f"{self._path}/index/structures/{ns}"),
            "graph_dir": lambda ns, name=None: Path(f"{self._path}/index/graph/{ns}"),
        }

        self.index_tasks = [
            self._add_semantic_index,
            self._add_graph_index,
            self._add_predicate_index,
            self._clean_up  # Still need this to clear in-memory records
        ]
        
        self.partial_index_tasks = [
            self._add_partial_semantic_index,
            self._add_partial_predicate_index,
            self._clean_up  # Still need this to clear in-memory records
        ]
        
               # If background thread mode is enabled, start the worker thread
        if use_background_thread:
            self._start_background_thread(
                partial_index_interval=self.partial_index_interval,
                full_index_interval=self.full_index_interval
            )

        """setup partial index tasks for dumping the predicate and vector state for use with hybrid search
        the full indexers dont need to change since they do complete rebuilds by necessity
        """
        
    def _clean_up(self):
        """Clean up records and temporary data"""
        # Always clear in-memory records
        self.records = []
        # We don't clear queue here - queue purging is handled at the end of processing cycles
        # to ensure all indexers have a chance to process the queue items

    def _enqueue_record(self, record: BaseModel):
        """Add a record to the queue in RocksDB"""
        ns = record.get_model_namespace()
        name = record.get_model_name()
        queue_key = f"queue:{ns}:{name}:{record.id}"
        self.db[queue_key.encode('utf-8')] = record.model_dump_json().encode('utf-8')
    
    def _get_queued_records(self):
        """Retrieve all records from the queue in RocksDB"""
        queued_records = []
        for key, value in self.repo.seek('queue:'):
            try:
                # Parse the record data
                record_data = json.loads(value.decode('utf-8'))
                # Extract namespace and name from key
                parts = key.split(':')
                if len(parts) >= 4:
                    ns, name = parts[1:3]
                    # Try to dynamically import and create model instance
                    # This is a simplified approach - you might need to enhance this
                    # based on how your model classes are organized
                    model_instance = self.repo.model_cls.model_validate(record_data)
                    queued_records.append(model_instance)
            except Exception as e:
                print(f"Error loading queued record {key}: {e}")
        return queued_records
    
    def _delete_queue_item(self, key):
        """Remove an item from the queue"""
        if isinstance(key, str):
            key = key.encode('utf-8')
        if key in self.db:
            del self.db[key]
            
    def purge_queue(self):
        """Purge all items from the queue after all indexers have processed them
        
        This should only be called at the end of a full processing cycle,
        after all indexers have had a chance to process the queue items.
        """
        removed_count = 0
        for key, _ in self.repo.seek('queue:'):
            self._delete_queue_item(key)
            removed_count += 1
        
        if removed_count > 0:
            print(f"✔️ Purged queue: removed {removed_count} queue items after full processing cycle")

    def add_records(self, records: typing.List[BaseModel]):
        """Add records to be indexed"""
        if not isinstance(records, list):
            records = [records]
            
        # Always add to the persistent queue
        for record in records:
            self._enqueue_record(record)
            
        if self.use_background_thread:
            # In background thread mode, we don't need to do anything else
            # The background thread will pick up the queued records
            return
        elif self.client_mode:
            # In client mode without background thread, add to in-memory records
            # for immediate processing when run/run_partial is called
            self.records += records

    def run_partial(self):
        """
        Run partial indexing - works differently in client and server modes
        
        In client mode: Process in-memory records for partial indexing
        In server mode: Process queue items from RocksDB
        
        Records can be enqueued at any time, and the indexer will process them when run or run_partial is called.
        Queue purging is handled separately at the end of full processing cycles.
        """
        if self.client_mode:
            # In client mode, submit tasks to the executor
            for task in self.partial_index_tasks:
                future = self.executor.submit(self._safe_run, task)
                self.futures.append(future)
        else:
            # In server mode, we load records from the queue and process them
            # This mode is meant to be run repeatedly by a background process
            queued_records = self._get_queued_records()
            if not queued_records:
                print("No records in queue")
                return
                
            print(f"⚙️ Processing {len(queued_records)} records from queue")
            self.records = queued_records
            
            # Process the tasks sequentially in the current thread
            # This is because we assume the server mode is already running in a background process
            for task in self.partial_index_tasks:
                self._safe_run(task)
            
            # Note: We don't purge the queue here, as different indexers might still need to process these items
            # Queue purging happens at the end of a full indexing cycle
        
    def run(self):
        """
        Run full indexing - works differently in client and server modes
        
        In client mode: Queues the tasks for the executor
        In server mode: Processes all items in the queue
        
        Records can be enqueued at any time, and the indexer will process them when run or run_partial is called.
        Queue purging is handled separately at the end of full processing cycles.
        """
        if self.client_mode:
            # In client mode, queue tasks for the executor
            for task in self.index_tasks:
                future = self.executor.submit(self._safe_run, task)
                self.futures.append(future)
        else:
            # In server mode, load from queue and process everything
            queued_records = self._get_queued_records()
            if not queued_records:
                print("No records in queue")
                return
                
            print(f"⚙️ Processing {len(queued_records)} records from queue")
            self.records = queued_records
            
            # Process all tasks sequentially in the current thread
            for task in self.index_tasks:
                self._safe_run(task)
                
            # After processing all tasks in a full indexing cycle, purge the queue
            self.purge_queue()
        
    def _safe_run(self, task_fn):
        """Safely run a task function with error handling"""
        try:
            task_fn()
            #print(f"✅ Finished {task_fn.__name__} ")
       
        except Exception as e:
            print(f"❌ Error in {task_fn.__name__}: {traceback.format_exc()}")

    def wait_for_completion(self):
        """Wait for all submitted tasks to complete"""
        if self.client_mode:
            # In client mode, wait for futures to complete
            for future in as_completed(self.futures):
                future.result()  # re-raises exceptions if any
            self.futures.clear()
            self.records = []
            print("✅ All tasks done.")
        else:
            # In server mode, this is a no-op since processing is synchronous
            pass
            
    def _start_background_thread(self, partial_index_interval:int=1, full_index_interval:int=5):
        """
        Start the background thread for processing indexing tasks
        
        Args:
            partial_index_interval: Seconds between partial index operations
            full_index_interval: Seconds between full index operations
        """
        if self.background_thread is not None and self.background_thread.is_alive():
            print("Background thread is already running")
            return
            
        self.should_stop.clear()
        self.background_thread = threading.Thread(
            target=self._background_worker,
            args=(partial_index_interval, full_index_interval),
            daemon=True,
            name="IndexerBackgroundThread"
        )
        self.background_thread.start()
        print(f"Started background indexer thread (partial_interval={partial_index_interval}s, full_interval={full_index_interval}s)")
    
    def _stop_background_thread(self):
        """Stop the background thread"""
        if self.background_thread is None or not self.background_thread.is_alive():
            return
            
        self.should_stop.set()
        self.task_queue.put(None)  # Sentinel to unblock queue.get()
        self.background_thread.join(timeout=5.0)
        if self.background_thread.is_alive():
            print("Warning: Background thread did not terminate gracefully")
        else:
            print("Background thread stopped successfully")
    
    def _initialize_index_state(self):
        """Initialize index state from RocksDB or set default values"""
        # Get last full index time from RocksDB
        if self.index_state_key in self.db:
            try:
                last_index_time_bytes = self.db.get(self.index_state_key)
                self.last_full_index_time = float(last_index_time_bytes.decode('utf-8'))
                print(f"Loaded last full index time: {datetime.fromtimestamp(self.last_full_index_time).isoformat()}")
            except Exception as e:
                print(f"Error loading index state: {e}")
                self.last_full_index_time = 0  # Default to epoch time to force full reindexing
        else:
            # No record found, set default value to force full reindexing on first run
            self.last_full_index_time = 0
            print("No previous index state found, will perform full indexing on first run")
    
    def _update_index_state(self):
        """Save current index state to RocksDB"""
        try:
            self.db[self.index_state_key] = str(self.last_full_index_time).encode('utf-8')
        except Exception as e:
            print(f"Error saving index state: {e}")
    
    def get_entity_timestamp(self, namespace, entity):
        """
        Get the last modified timestamp for an entity from RocksDB
        
        Args:
            namespace: Entity namespace
            entity: Entity name
            
        Returns:
            float: Timestamp when the entity was last modified
        """
        key = f"table_stats:{namespace}:{entity}".encode('utf-8')
        try:
            if key in self.db:
                stats = json.loads(self.db.get(key).decode('utf-8'))
                if 'last_updated' in stats:
                    # Convert ISO timestamp to float
                    return datetime.fromisoformat(stats['last_updated']).timestamp()
        except Exception as e:
            print(f"Error getting entity timestamp for {namespace}:{entity}: {e}")
        
        # Return default timestamp to force indexing if no record found
        return 0
    
    def get_modified_entities(self):
        """
        Get set of entities that have been modified since last full index
        
        Returns:
            set: Set of (namespace, entity) tuples
        """
        modified_entities = set()
        
        # Look at all table stats entries to find modified entities
        for key, value in self.repo.seek('table_stats_partial:'):
            try:
                # Parse key to extract namespace and entity
                parts = key.split(':')
                if len(parts) >= 3:
                    namespace = parts[1]
                    entity = parts[2]
                    
                    # Get modification time from the stats
                    stats = json.loads(value.decode('utf-8'))
                    if 'last_updated' in stats:
                        update_time = datetime.fromisoformat(stats['last_updated']).timestamp()
                        
                        # Check if modified after last full index
                        if update_time > self.last_full_index_time:
                            modified_entities.add((namespace, entity))
            except Exception as e:
                print(f"Error parsing entity stats {key}: {e}")
                
        return modified_entities
    
    def _background_worker(self, partial_index_interval:int=1, full_index_interval:int=5):
        """
        Core worker function for processing indexing tasks at regular intervals
        
        This is the single source of truth for indexing logic, used by both
        the background thread and the legacy process_queue_continuously method.
        
        Args:
            partial_index_interval: Seconds between partial index operations
            full_index_interval: Seconds between full index operations
        """
   
        counter = 0
        last_partial_index_time = time.time()
        
        # Thread mode check function
        should_stop = lambda: self.should_stop.is_set() if hasattr(self, 'should_stop') else False
        
        try:
            while not should_stop():
                current_time = time.time()
                queued_records = self._get_queued_records()
                
                # Run partial index on a faster interval if there are records to process
                if (current_time - last_partial_index_time >= partial_index_interval) and queued_records:
                    print(f"⚙️ Processing {len(queued_records)} records for partial indexing")
            
                    self.records = queued_records
                    
                    if len(self.records):
                        # Process partial indexing tasks
                        for task in self.partial_index_tasks:
                            self._safe_run(task)
            
                        last_partial_index_time = current_time
                        print(f"✅ Partial indexing complete. Processed {len(self.records)} modified entities.")
                
                # Run full index on a longer interval
                if current_time - self.last_full_index_time >= full_index_interval:
                    # Process full indexing tasks
                    for task in self.index_tasks:
                        self._safe_run(task)
                    
                    # Update last full index time in memory and persist to RocksDB
                    modified_count = len(self.get_modified_entities())
                    if modified_count:
                        print(f"✅ Full indexing complete. Processed {modified_count} modified entities.")
                    self.last_full_index_time = current_time
                    self._update_index_state()
                
                # Purge the queue after all indexers have completed
                self.purge_queue()
                
                # If in thread mode, check stop flag more frequently
                if hasattr(self, 'should_stop'):
                    for _ in range(20):  # Check stop flag more frequently than we check queue
                        if self.should_stop.is_set():
                            break
                        time.sleep(0.1)
                else:
                    # In non-thread mode (legacy process_queue_continuously), just sleep
                    time.sleep(partial_index_interval)
                
                counter += 1
                
        except Exception as e:
            print(f"Error in background worker: {e}")
            traceback.print_exc()
            
        print("Background indexer stopped")
    
    def process_queue_continuously(self, interval:int=5, full_index_interval_factor:int=12):
        """
        Legacy function that delegates to background worker implementation
        
        This method is maintained for backwards compatibility and now 
        uses the shared background worker implementation.
        
        Args:
            interval: Seconds to wait between queue processing attempts
            full_index_interval_factor: Run full index every (interval * full_index_interval_factor) seconds
        """
        # Convert this legacy method to use the background worker implementation
        print(f"Starting continuous queue processing (delegating to background worker)")
        
        # Store current state to restore afterward
        original_use_background_thread = self.use_background_thread
        
        # Use the background worker implementation but run it synchronously
        self.use_background_thread = False
        
        # Make sure the index_tasks and partial_index_tasks attributes are defined
        # These are typically defined in __init__, but we re-define them here to be safe
        if not hasattr(self, 'index_tasks'):
            self.index_tasks = [
                self._add_semantic_index,
                self._add_graph_index,
                self._add_predicate_index,
                self._clean_up
            ]
            
        if not hasattr(self, 'partial_index_tasks'):
            self.partial_index_tasks = [
                self._add_partial_semantic_index,
                self._add_partial_predicate_index,
                self._clean_up
            ]
        
        try:
            # Match the parameters to the background worker
            partial_index_interval = interval
            full_index_interval = interval * full_index_interval_factor
            
            # Call the core worker function directly with the configured intervals
            self._background_worker(
                partial_index_interval=partial_index_interval,
                full_index_interval=full_index_interval
            )
                
        except KeyboardInterrupt:
            print("Continuous processing stopped by user")
        except Exception as e:
            print(f"Error in continuous processing: {e}")
            traceback.print_exc()
        finally:
            # Restore original state
            self.use_background_thread = original_use_background_thread

 
    # Actual implementations (shells for now)
    def get_path(self, path_type: str, namespace: str=None, name: typing.Optional[str] = None) -> Path:
        """Get a path for a specified index type and entity
        we default to our repo/model entity details which is the normal use case
    
        Args:
            path_type: Type of path ('semantic', 'parquet', 'graph', etc.)
            namespace: Entity namespace
            name: Entity name (if applicable)
        
        Returns:
            Path object for the specified resource
        """
        name = name or self.repo.model_logic.name
        namespace = namespace or self.repo.model_logic.namespace
        
        path = self.paths[path_type](namespace, name)
            
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
            
    def _add_semantic_index(self, dim:int=1536):
        """Build and save semantic vector indexes for all indexed entities
        
        - Creates HNSW index files for efficient approximate nearest neighbor search
        - Maps index hash IDs back to entity IDs for lookup
        - Clears temporary vector mappings after building the full index
        - Only processes entities that have been modified since last full index
        """
        grouped_vecs = defaultdict(list)
        grouped_ids = defaultdict(list)
        
        # Get the set of entities that need full indexing based on persistent tracking
        entities_to_index = self.get_modified_entities()
        
        if not entities_to_index:
            print("No entities have been modified since last full semantic indexing, skipping")
            return
            
        print(f"Running semantic full indexing for {len(entities_to_index)} modified entities: {entities_to_index}")

        """this repo should only iterate over the model"""
        for key, val in self.repo.iter_model_items():
            
            parts = key.split(":")
            if parts[0] == "vindex":
                """TODO: dangerous depending on id orde"""
                _, ns, ent, oid = parts
                
                # Skip entities that haven't been modified
                if (ns, ent) not in entities_to_index:
                    continue
                    
                vec = self.repo.decode_vector(val)
                if vec.shape[0] != dim:
                    print(f"❌ Dimension mismatch {key}")
                    continue
                grouped_vecs[(ns, ent)].append(vec)
                grouped_ids[(ns, ent)].append(oid)

        # Build indexes for each entity type
        for (ns, ent), vecs in grouped_vecs.items():
            arr = np.vstack(vecs)
            ids = [] 
            # Create ID mapping
            for oid in grouped_ids[(ns, ent)]:
                hash_id = str(int(hashlib.sha1(oid.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF)
                # Use model logic to generate the proper key: the one is the link between the id in the index and our key
                # this is used when we match nearby vectors and need to recover our records
                key = self.model_logic.get_semmap_from_namespace_and_name(ns,ent, hash_id)
                self.db[key] = oid.encode('utf-8')
                ids.append(int(hash_id))

            # Create and configure HNSW index
            idx = hnswlib.Index(space='cosine', dim=dim)
            idx.init_index(max_elements=arr.shape[0], ef_construction=200, M=16)
            idx.add_items(arr, np.array(ids, dtype=np.int64))
            idx.set_ef(50)  # Search quality parameter

            # Save index to file
            index_path = self.get_path("semantic", ns, ent)
            idx.save_index(str(index_path))

            print(f"✔️ Saved HNSW index: {index_path}")
            
            # Clear temporary vector mappings after building the full index
            partial_key = f"partial-vindex:{ns}:{ent}:all"
            if partial_key.encode('utf-8') in self.db:
                self.db[partial_key.encode('utf-8')] = set()
                print(f"✔️ Cleared partial vector index: {partial_key}")


    def _add_graph_index(self, namespace:str='concept'):
        """
        Graph indexing that handles:
        - Adding global node pointers to named entities
        - Creating graph connections based on graph_paths attribute
        
        The graph index only indexes new items while the semantic and 
        predicate index work on batches at the moment.
        """
        # Create a GraphLogic instance to handle graph operations
        graph = self.repo.graph
   
        for obj in self.records:
            # Global node index by name
            name = getattr(obj, "name", None)
           
            if name:
                # Use model_logic to generate the proper global key
                global_key = self.model_logic.get_global_id(name)
                # Get entity key using model_logic
                entity_key = self.model_logic.get_hash_id(obj.id, encode=False)
                entity_type = f"{obj.get_model_namespace()}.{obj.get_model_name()}"
                
                # Build the value to store
                value = {
                    entity_type: {
                        "key": entity_key,
                        "updated_at": getattr(obj, "updated_at", datetime.utcnow().isoformat())
                    }
                }

                # Update or create global entry
                existing = self.repo._get_raw(global_key)
                if existing:
                    existing_dict = json.loads(existing.decode("utf-8"))
                    existing_dict.update(value)
                    self.repo._add_raw(global_key, existing_dict)
                else:
                    self.repo._add_raw(global_key, value)

            # Graph path indexing - this is used to stitch content nodes into a graph connected by concept nodes
            graph_paths = getattr(obj, "graph_paths", None) or []
            origin_id = obj.id
            timestamp = datetime.utcnow().isoformat()

            # Process each graph path
            for path in graph_paths:
                nodes = path.split("/")
                for node in nodes:
                    # Upsert category node with timestamp using GraphLogic
                    node_key = graph.node_key(node)
                    node_data = {
                        "updated_at": timestamp
                    }
                    self.repo._add_raw(node_key, node_data)

                    # Create bidirectional edges using GraphLogic
                    edge_1 = graph.edge_key(origin_id, node)
                    edge_2 = graph.edge_key(node, origin_id)
                    edge_val = {"timestamp": timestamp}

                    self.repo._add_raw(edge_1, edge_val)
                    self.repo._add_raw(edge_2, edge_val)

    def _add_predicate_index(self):
        """Build predicate indexes (Parquet files) for SQL querying
        
        - Collects all records of each type and exports them to Parquet files
        - Stores table stats for query planning
        - Removes temporary parquet files after full index build
        - Only processes entities that have been modified since last full index
        """
        import polars as pl
        
        # Get the set of entities that need full indexing based on persistent tracking
        entities_to_index = self.get_modified_entities()
        
        if not entities_to_index:
            print("No entities have been modified since last full predicate indexing, skipping")
            return
            
        print(f"Running predicate full indexing for {len(entities_to_index)} modified entities: {entities_to_index}")
        
        # Group records by namespace and entity type
        grouped_data = defaultdict(list)
        
        """decodes by default"""
        for key, val in self.repo.iter_model_items():
            parts = key.split(":")
            
            # Check for hash pattern (namespace:entity:hash:id)
            if len(parts) == 4 and parts[0] == "hash":
                """todo this is dangerous because we need to use some sort of model logic parsing"""
                _, namespace, table, obj_id = parts
                
                # Skip entities that haven't been modified
                if (namespace, table) not in entities_to_index:
                    continue
                
                record_data = json.loads(val.decode("utf-8")) 
                grouped_data[(namespace, table)].append(record_data)
 
        for (namespace, table), records in grouped_data.items(): 
            df = pl.DataFrame(records)
            parquet_path = self.get_path("parquet", namespace, table)
            df.write_parquet(parquet_path)
            print(f"✔️ Wrote index to {parquet_path}")
            
            # Remove temporary parquet file after full index is built
            self._remove_temp_file(namespace, table)
            
            # Store table stats for query planning
            stats = {
                'items': len(df),
                'last_updated': datetime.utcnow().isoformat(),
                'columns': df.columns
            }
            # Store with a consistent key pattern
            stats_key = f"table_stats:{namespace}:{table}"
            self.repo._add_raw(stats_key, json.dumps(stats))
            
            # Clear the partial stats entry
            stats_key_partial = f"table_stats_partial:{namespace}:{table}"
            if self.repo._get_raw(stats_key_partial):
                self.repo._delete_raw(stats_key_partial)
                print(f"✔️ Cleared partial table stats: {stats_key_partial}")

    def seek(self, prefix):
        """Wrapper around repo's seek method for backward compatibility
        
        Args:
            prefix: Key prefix to search for
            
        Yields:
            Tuples of (key_str, value) matching the prefix
        """
        # Use repo's efficient prefix-based seek
        return self.repo.seek(prefix)
                
    def _add_partial_semantic_index(self):
        """Build and save partial semantic vector indexes
        
        - Creates a list of vector keys for efficient hybrid search
        - Stores keys in a set for "partial-vindex:namespace:agent:all"
        - Clears processed items from queue after indexing
        - Tracks touched entities for efficient full indexing
        """
        partial_vectors = set()
        processed_records = []
        entity_namespaces = set()  # Track which entities are being processed
        
        for obj in self.records:
            key = self.model_logic.get_vindex_for_model_instance(obj)
            partial_vectors.add(key)
            # Track processed records for queue cleanup
            namespace = obj.get_model_namespace()
            table = obj.get_model_name()
            processed_records.append((namespace, table, obj.id))
            
            # Track the entity namespace pair
            entity_namespaces.add((namespace, table))
        
        # Store the set of vector keys
        if partial_vectors:
            # Create a key for the partial vector index
            partial_key = f"partial-vindex:{self.model_logic.namespace}:{self.model_logic.name}:all"
            # Store the set as set because our provider will auto pickle but test interface
            # TODO: broken abstractions with direct db access
            existing = self.db.get(partial_key.encode('utf-8'))
            if existing:
                existing_set = existing  # Assuming the DB implementation handles deserialization
                partial_vectors |= existing_set  # Union instead of intersection to add new keys
            self.db[partial_key] = partial_vectors
            
            print(f"✔️ Saved partial vector index: {partial_key} with {len(partial_vectors)} keys")
        
        # Update the touched entities tracking
        current_time = time.time()
        for entity_ns in entity_namespaces:
            self.touched_entities[entity_ns] = current_time
            
        if entity_namespaces:
            print(f"✔️ Tracked {len(entity_namespaces)} touched entities for partial semantic indexing")
            
        # Track processed records for later queue purging
        # We'll purge the queue at the end of the indexing cycle
            
    def acquire_lock(self, lock_path: Path, timeout: int = 60) -> typing.Optional[int]:
        """Acquire a file lock with timeout
        
        Args:
            lock_path: Path to the lock file
            timeout: Maximum time to wait for the lock in seconds
            
        Returns:
            File descriptor if lock was acquired, None if timeout
        """
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        lock_fd = None
        
        # Try to acquire the lock with timeout
        while (time.time() - start_time) < timeout:
            try:
                lock_fd = open(str(lock_path), 'w+')
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lock_fd
            except (IOError, OSError):
                # Lock is held by another process, wait and retry
                time.sleep(1)
        
        return None
        
    def release_lock(self, lock_fd):
        """Release a file lock
        
        Args:
            lock_fd: File descriptor returned by acquire_lock
        """
        if lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
            
    def _remove_temp_file(self, namespace=None, table=None):
        """
        Remove the temporary parquet file after rebuilding the full data.parquet index
        
        Args:
            namespace: The namespace of the entity (defaults to model's namespace)
            table: The table/entity name (defaults to model's name)
        """
        namespace = namespace or self.model_logic.namespace
        table = table or self.model_logic.name
        
        temp_parquet_path = self.get_path("temp_parquet", namespace, table)
        if Path(temp_parquet_path).exists():
            lock_path = self.get_path("parquet_lock", namespace, table)
            lock_fd = self.acquire_lock(lock_path)
            
            if not lock_fd:
                print(f"⚠️ Could not acquire lock for {temp_parquet_path}, skipping temp file removal")
                return False
            
            try:
                Path(temp_parquet_path).unlink()
                print(f"✔️ Removed temporary parquet file: {temp_parquet_path}")
                return True
            except Exception as e:
                print(f"❌ Error removing temporary file {temp_parquet_path}: {e}")
                return False
            finally:
                self.release_lock(lock_fd)
                
        return True  # No file to remove is a success case
                
    def _add_partial_predicate_index(self):
        """Build partial predicate indexes (temp Parquet files) for SQL querying
        
        - Collects records in memory and exports them to temporary Parquet files
        - Uses file locks to ensure concurrent safety
        - DuckDB can then query both the main and temp Parquet files using wildcards
        - Clears processed items from queue after indexing
        - Tracks touched entities for efficient full indexing
        """
        import polars as pl
        
        # Group records by namespace and entity type
        grouped_data = defaultdict(list)
        # Keep track of processed record IDs to remove from queue
        processed_records = []
        # Track which entities are being processed for efficient full indexing
        entity_namespaces = set()
             
        """the implementation pretends to be heterogenous but really the indexer is model bound - but easier to change in future if we want"""
        for obj in self.records:
            namespace = obj.get_model_namespace()
            table = obj.get_model_name()
            # Serialize the record to dict and add to grouped data
            record_data = obj.model_dump()
            grouped_data[(namespace, table)].append(record_data)
            # Add to processed records list
            processed_records.append((namespace, table, obj.id))
            # Track the entity namespace pair
            entity_namespaces.add((namespace, table))
 
        for (namespace, table), records in grouped_data.items():
       
            lock_path = self.get_path("parquet_lock", namespace, table)
            lock_fd = self.acquire_lock(lock_path)
            if not lock_fd:
                print(f"⚠️ Could not acquire lock for {namespace}:{table}, skipping partial predicate indexing")
                continue
            try:
                # Get temp parquet file path
                temp_parquet_path = self.get_path("temp_parquet", namespace, table)
                
                # Create dataframe and write to temp parquet file
                df = pl.DataFrame(records)
                df.write_parquet(temp_parquet_path)
                print(f"✔️ Wrote partial predicate index to {temp_parquet_path} with {len(records)} records")
                
                # Store table stats for query planning
                stats = {
                    'partial_items': len(df),
                    'last_updated': datetime.utcnow().isoformat(),
                    'columns': df.columns
                }
                # Store with a consistent key pattern
                stats_key = f"table_stats_partial:{namespace}:{table}"
                self.repo._add_raw(stats_key, json.dumps(stats))
                
                # Update the touched entities tracking for this namespace/table
                current_time = time.time()
                self.touched_entities[(namespace, table)] = current_time
            except:
                raise
            finally:
                # Always release the lock
                self.release_lock(lock_fd)
        
        if entity_namespaces:
            print(f"✔️ Tracked {len(entity_namespaces)} touched entities for partial predicate indexing")
                
        # Track processed records for later queue purging
        # We'll purge the queue at the end of the indexing cycle
    
    def get_partial_vectors(self, namespace: str, entity: str) -> typing.List[tuple]:
        """Retrieve the list of vector keys and their actual vectors from the partial index
        
        Args:
            namespace: Entity namespace
            entity: Entity name
            
        Returns:
            List of (obj_id, vector) tuples for the partial index
        """
        partial_key = f"partial-vindex:{namespace}:{entity}:all"
        data = self.repo._get_raw(partial_key)
        
        if not data:
            return []
            
        # Get the list of keys from the partial index
        vector_keys = json.loads(data.decode("utf-8"))
        
        # Resolve each key to get the actual vector and extract the ID
        result = []
        for key in vector_keys:
            # Get the vector data
            vec_data = self.repo._get_raw(key)
            if vec_data:
                # Extract the ID from the key
                parts = key.split(":")
                if len(parts) >= 4:
                    obj_id = parts[3]
                    # Decode the vector
                    vector = self.repo.decode_vector(vec_data)
                    result.append((obj_id, vector))
                    
        return result
        
    def get_top_neighbors(self, 
                        model: BaseModel, 
                        input_vector: typing.List[float], 
                        n: int = 5) -> typing.List[tuple]:
        """
        Find the top n vectors most similar to input_vector for a specific model type
        
        Args:
            model: Model instance to determine namespace/entity type
            input_vector: Query vector to compare against
            m: Maximum number of vectors to load (not currently used, kept for API compatibility)
            n: Number of results to return
            
        Returns:
            List of (key, similarity_score) tuples for the most similar items
        """
        # This method can now delegate to the repository's vector search
        # We generate the right model logic based on the model instance
        temp_model_logic = self.repo.model_logic.__class__(model.__class__)
        
        # Get the namespace and name from the model
        namespace = model.get_model_namespace()
        entity = model.get_model_name()
        
        # Use repo's direct vector search which already handles the vector retrieval and comparison
        # But we need to modify it for our return format
        results = self.repo._search_vectors_direct(
            query_vector=input_vector,
            top_k=n,
            ids_only=True  # Get IDs with scores
        )
        
        # Convert the (id, score) tuples to (key, score) tuples
        formatted_results = []
        for obj_id, score in results:
            # Create the full hash key using model logic
            hash_key = f"{namespace}:{entity}:hash:{obj_id}"
            formatted_results.append((hash_key, score))
            
        return formatted_results