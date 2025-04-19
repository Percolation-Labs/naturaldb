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
import os

from concurrent.futures import ThreadPoolExecutor, as_completed

class Indexer:
    def __init__(self, repo):
        self.repo = repo  # Keep reference to the parent repository
        self.db = repo.db
        self._path = repo._path
        self.model_logic = repo.model_logic  # Access to model logic for key handling
        self.records: list[BaseModel] = []
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures = []

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
            self._clean_up
        ]
        
        self.partial_index_tasks = [
            self._add_partial_semantic_index,
            self._add_partial_predicate_index,
            self._clean_up
        ]
        
        """setup partial index tasks for dumping the predicate and vector state for use with hybrid search
        the full indexers dont need to change since they do complete rebuilds by necessity
        """
        
    def _clean_up(self):
        self.records = []
        for key, value in self.repo.seek('partial-vindex:'):
            self[key] = set()

    def add_records(self, records: typing.List[BaseModel]):
        if not isinstance(records, list):
            records = [records]
        self.records += records

    def run_partial(self):
        """
        for now we are going to write a partial page 
        -- lock the partial parquet file access for writes and then merge
        -- write the vector index key stack that can be used on conjunction with the index blob - this creates a hybrid vector scan method
        
        full dont need to change because they build full indexes by necessity but we could combine them so we dont need to scan twice
        """
        if self.futures:
            print("Indexer already running")
            return

        for task in self.partial_index_tasks:
            future = self.executor.submit(self._safe_run, task)
            self.futures.append(future)
        #self.records = []
        
    def run(self):
        if self.futures:
            print("Indexer already running")
            return

        for task in self.index_tasks:
            future = self.executor.submit(self._safe_run, task)
            self.futures.append(future)
        #self.records = []
        
    def _safe_run(self, task_fn):
        try:
            task_fn()
            print(f"✅ Finished {task_fn.__name__} ")
       
        except Exception as e:
            print(f"❌ Error in {task_fn.__name__}: {traceback.format_exc()}")

    def wait_for_completion(self):
        for future in as_completed(self.futures):
            future.result()  # re-raises exceptions if any
        self.futures.clear()
        self.records = []
        print("✅ All tasks done.")

 
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
        """
        grouped_vecs = defaultdict(list)
        grouped_ids = defaultdict(list)

        """this repo should only iterate over the model"""
        for key, val in self.repo.iter_model_items():
            
            parts = key.split(":")
            if parts[0] == "vindex":
                """TODO: dangerous depending on id orde"""
                _, ns, ent, oid = parts
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
                # Use model logic to generate the proper key
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

            print(f"✅ Saved HNSW index: {index_path}")


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
        """
        import polars as pl
        # Group records by namespace and entity type
        grouped_data = defaultdict(list)
        
        """decodes by default"""
        for key, val in self.repo.iter_model_items():
            parts = key.split(":")
            
            # Check for hash pattern (namespace:entity:hash:id)
            if len(parts) == 4 and parts[0] == "hash":
                """todo this is dangerous because we need to use some sort of model logic parsing"""
                _, namespace, table, obj_id = parts
                
                record_data = json.loads(val.decode("utf-8")) 
                grouped_data[(namespace, table)].append(record_data)
 
        for (namespace, table), records in grouped_data.items(): 
            df = pl.DataFrame(records)
            parquet_path = self.get_path("parquet", namespace, table)
            df.write_parquet(parquet_path)
            print(f"✅ Wrote index to {parquet_path}")
            
            """TODO do this with retries and backoff to acquire a lock"""
            self._remove_temp_file()
            # Store table stats for query planning
            stats = {
                'items': len(df),
                'last_updated': datetime.utcnow().isoformat(),
                'columns': df.columns
            }
            # Store with a consistent key pattern
            stats_key = f"table_stats:{namespace}:{table}"
            self.repo._add_raw(stats_key, json.dumps(stats))

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
        """
        partial_vectors = set()
        
        for obj in self.records:
            key = self.model_logic.get_vindex_for_model_instance(obj)
            partial_vectors.add(key)
        
        # Store the set of vector keys
        if partial_vectors:
            # Create a key for the partial vector index
            partial_key = f"partial-vindex:{self.model_logic.namespace}:{self.model_logic.name}:all"
            # Store the set as set because our provider will auto pickle but test interface
            # TODO: broken abstractions with direct db access
            existing = self.db.get(partial_key)
            if existing:
                partial_vectors &= existing
            self.db[partial_key] = partial_vectors
            
            print(f"✅ Saved partial vector index: {partial_key} with {len(partial_vectors)} keys")
            
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
            
    def _remove_temp_file(self):
        """
        remove the temporary parquet file - this should be done just as we rebuild the full data.parquet index
        """
        
        temp_parquet_path = self.get_path("temp_parquet")
        if Path(temp_parquet_path).exists():
            
            lock_path = self.get_path("parquet_lock")
            lock_fd = self.acquire_lock(lock_path)
            if not lock_fd:
                print(f"⚠️ Could not acquire lock for {temp_parquet_path}, skipping partial predicate indexing")
                raise Exception('implement backoff')
                """"""
            Path(temp_parquet_path).unlink()
                    
                
    def _add_partial_predicate_index(self):
        """Build partial predicate indexes (temp Parquet files) for SQL querying
        
        - Collects records in memory and exports them to temporary Parquet files
        - Uses file locks to ensure concurrent safety
        - DuckDB can then query both the main and temp Parquet files using wildcards
        """
        import polars as pl
        
        # Group records by namespace and entity type
        grouped_data = defaultdict(list)
             
        """the implementation pretends to be heterogenous but really the indexer is model bound - but easier to change in future if we want"""
        for obj in self.records:
            namespace = obj.get_model_namespace()
            table = obj.get_model_name()
            # Serialize the record to dict and add to grouped data
            record_data = obj.model_dump()
            grouped_data[(namespace, table)].append(record_data)
 
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
                print(f"✅ Wrote partial predicate index to {temp_parquet_path} with {len(records)} records")
                
                # Store table stats for query planning
                stats = {
                    'partial_items': len(df),
                    'last_updated': datetime.utcnow().isoformat(),
                    'columns': df.columns
                }
                # Store with a consistent key pattern
                stats_key = f"table_stats_partial:{namespace}:{table}"
                self.repo._add_raw(stats_key, json.dumps(stats))
            except:
                raise
            finally:
                # Always release the lock
                self.release_lock(lock_fd)
    
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