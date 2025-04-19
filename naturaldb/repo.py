import os
import hashlib
import json
import requests
from typing import List, Optional, Type, TypeVar, Dict, Any, Union, Tuple, Iterator
from pydantic import BaseModel
from collections import defaultdict
import numpy as np
from pathlib import Path
import hnswlib
from rocksdict import Rdict, Options, SliceTransform
import duckdb
from datetime import datetime
from .index.Backgroundindexer import Indexer
import typing

T = TypeVar("T", bound=BaseModel)

OPENAI_SQL_MODEL = 'gpt-4.1-mini'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-ada-002"
DB_PATH = str(Path.home() / '.percolate' / 'natural-db')

def sha256_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

def _ensure_list(items: Any) -> List[Any]:
    """Ensure input is a list, converting single items to a list"""
    if not isinstance(items, list):
        items = [items]
    return items
    
class ModelLogic:
    """Conventions for key names and serialization to interface with RocksDB"""
    def __init__(self, model_cls: Type[T]):
        """Initialize model logic with the given model class"""
        self.model_cls = model_cls
        
    @property
    def namespace(self) -> str:
        """Get the model namespace"""
        return self.model_cls.get_model_namespace()
    
    @property
    def name(self) -> str:
        """Get the model name"""
        return self.model_cls.get_model_name()
    
    @property
    def hash_prefix(self) -> str:
        """Get the prefix for hash keys"""
        return f"hash:{self.namespace}:{self.name}:"    

    @classmethod
    def get_vindex_for_model_instance(cls, obj: BaseModel):
        """in reality the basemodel is abstracted"""
        ns = obj.get_model_namespace()
        name = obj.get_model_name()
        return cls.get_vindex_from_namespace_and_name(namespace=ns, name=name, id=obj.id)

    @classmethod
    def get_semmap_from_namespace_and_name(cls, namespace, name,id,encode:bool=True):
        k = f"sem_index_map:{namespace}:{name}:{id}" 
        return k if not encode else k.encode("utf-8")
          
    @classmethod
    def get_id_from_namespace_and_name(cls, namespace, name,id,encode:bool=True):
        k = f"hash:{namespace}:{name}:{id}" 
        return k if not encode else k.encode("utf-8")
    
    @classmethod
    def get_vindex_from_namespace_and_name(cls, namespace, name,id,encode:bool=True):
        k = f"vindex:{namespace}:{name}:{id}" 
        return k if not encode else k.encode("utf-8")
    
    @property
    def vector_prefix(self) -> str:
        """Get the prefix for vector keys"""
        return f"vindex:{self.namespace}:{self.name}:"
    
    @property
    def sem_index_map_prefix(self) -> str:
        """Get the prefix for semantic index mapping keys"""
        return f"sem_index_map:{self.namespace}:{self.name}:"
    
    @property
    def global_node_prefix(self) -> str:
        """Get the prefix for global node keys"""
        return "global:node:"
    
    def get_hash_id(self, id: str, encode: bool = True) -> Union[str, bytes]:
        """Get the hash key for an object ID, optionally encoding to bytes"""
        k = f"{self.hash_prefix}{id}"
        return k if not encode else k.encode("utf-8")
    
    def get_vector_id(self, id: str, encode: bool = True) -> Union[str, bytes]:
        """Get the vector key for an object ID, optionally encoding to bytes"""
        k = f"{self.vector_prefix}{id}"
        return k if not encode else k.encode("utf-8")
    
    def get_global_id(self, name: str, encode: bool = True) -> Union[str, bytes]:
        """Get the global node key for a name, optionally encoding to bytes"""
        k = f"{self.global_node_prefix}{name}"
        return k if not encode else k.encode("utf-8")
    
    def get_sem_index_map_id(self, hash_id: str, encode: bool = True) -> Union[str, bytes]:
        """Get the semantic index mapping key, optionally encoding to bytes"""
        k = f"{self.sem_index_map_prefix}{hash_id}"
        return k if not encode else k.encode("utf-8")
    
    def wrap_system(self, obj: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
        """Add system metadata to a model or dict before storage
        
        Currently adds:
        - updated_at timestamp
        """
        if isinstance(obj, BaseModel):
            data = obj.model_dump()
        else:
            data = obj.copy()
            
        data["updated_at"] = datetime.utcnow().isoformat()
        return data
    
    def serialize(self, obj: BaseModel) -> bytes:
        """Serialize a model to JSON bytes for storage"""
        wrapped = self.wrap_system(obj)
        return json.dumps(wrapped).encode("utf-8")
    
    def deserialize(self, data: bytes) -> T:
        """Deserialize bytes to a model instance"""
        json_data = json.loads(data.decode("utf-8"))
        return self.model_cls.model_validate(json_data)

class GraphLogic:
    """Helper to traverse edges using key conventions"""
    def __init__(self, repo: 'RocksRepo', namespace: str = "concept"):
        """Initialize with repository and default namespace"""
        self.repo = repo
        self.namespace = namespace
    
    def edge_prefix(self, node_id: str) -> bytes:
        """Get prefix for all edges from a node"""
        return f"{self.namespace}:edge:{node_id}:".encode("utf-8")
    
    def node_key(self, node_id: str) -> bytes:
        """Get key for a node"""
        return f"{self.namespace}:node:{node_id}".encode("utf-8")
    
    def edge_key(self, src_id: str, dst_id: str) -> bytes:
        """Get key for an edge between two nodes"""
        return f"{self.namespace}:edge:{src_id}:{dst_id}".encode("utf-8")
        
    def _get_edges_from_node(self, node_id: str) -> List[str]:
        """Get all edges from a node"""
        prefix = self.edge_prefix(node_id)
        edges = []
        
        for key, _ in self.repo.seek(prefix):
            parts = key.decode("utf-8").split(":")
            if len(parts) == 4:
                edges.append(parts[3])  # dst node
                
        return edges
    
    def _get_edges_from_nodes(self, node_ids: List[str]) -> Dict[str, List[str]]:
        """Get all edges from multiple nodes"""
        result = {}
        for node_id in node_ids:
            result[node_id] = self._get_edges_from_node(node_id)
        return result
    
    def get_neighbourhood(self, seed_keys: List[str], distance: int = 3) -> Dict[str, Any]:
        """Explore graph vicinity from seed keys up to given distance
        
        Returns a dict mapping node IDs to their metadata and connections
        """
        if distance <= 0:
            return {}
            
        visited = set(seed_keys)
        frontier = set(seed_keys)
        neighborhood = {}
        
        for _ in range(distance):
            new_frontier = set()
            for node_id in frontier:
                # Get node data
                node_key = self.node_key(node_id)
                node_data = self.repo._get_raw(node_key)
                if node_data:
                    neighborhood[node_id] = json.loads(node_data.decode("utf-8"))
                
                # Get connected nodes
                edges = self._get_edges_from_node(node_id)
                if node_id in neighborhood:
                    neighborhood[node_id]["edges"] = edges
                
                # Add new nodes to frontier
                for edge in edges:
                    if edge not in visited:
                        visited.add(edge)
                        new_frontier.add(edge)
            
            frontier = new_frontier
            if not frontier:
                break
                
        return neighborhood

class RocksRepo:
    def __init__(self, model_cls: Type[T], db_path: str = DB_PATH):
        """Initialize repository for a specific model class"""
        self._path = db_path
        self.model_cls = model_cls
        Path(f"{self._path}/data").parent.mkdir(parents=True, exist_ok=True)
        self.db = Rdict(f"{self._path}/data")
        self.model_logic = ModelLogic(model_cls)
        self.graph = GraphLogic(self)
        self.index = Indexer(self)

        ###                 ##
        #   ROCKS INTERFACE  #
        ###                 ##

    def _add_raw(self, key: Union[str, bytes], value: Union[str, bytes, dict]) -> None:
        """Add a raw key-value pair to the database"""
        encoded_key = key if isinstance(key, bytes) else key.encode("utf-8")
        
        if isinstance(value, dict):
            encoded_value = json.dumps(value).encode("utf-8")
        elif isinstance(value, str):
            encoded_value = value.encode("utf-8")
        else:
            encoded_value = value
            
        self.db[encoded_key] = encoded_value
    
    def _get_raw(self, key: Union[str, bytes]) -> Optional[bytes]:
        """Get raw value for a key"""
        encoded_key = key if isinstance(key, bytes) else key.encode("utf-8")
        return self.db.get(encoded_key)
    
    def _delete_raw(self, key: Union[str, bytes]) -> None:
        """Delete a key from the database"""
        encoded_key = key if isinstance(key, bytes) else key.encode("utf-8")
        if encoded_key in self.db:
            del self.db[encoded_key]
    
    def seek(self, prefix: Union[str, bytes]) -> Iterator[Tuple[str, bytes]]:
        """Efficiently seek keys by prefix"""
        encoded_prefix = prefix if isinstance(prefix, bytes) else prefix.encode("utf-8")
        
        for key, val in self.db.items(from_key=encoded_prefix):
            if not key.startswith(encoded_prefix):
                break
            yield key.decode("utf-8"), val
    
    def iter_keys(self, prefix: Optional[Union[str, bytes]] = None, decode_keys: bool = True) -> Iterator[Union[str, bytes]]:
        """Iterate over keys, optionally filtered by prefix"""
        if prefix:
            for key, _ in self.seek(prefix):
                yield key if decode_keys else key.encode("utf-8")
        else:
            for key in self.db.keys():
                yield key.decode("utf-8") if decode_keys else key
    
    def iter_model_items(self, decode_keys: bool = True):
        """iterate only over the model items"""
        for key, value in self.iter_items(prefix=self.model_logic.hash_prefix, decode_keys=decode_keys):
            yield key, value
    
    def iter_items(self, prefix: Optional[Union[str, bytes]] = None, decode_keys: bool = True) -> Iterator[Tuple[Union[str, bytes], bytes]]:
        """Iterate over key-value pairs, optionally filtered by prefix"""
        if prefix:
            for key, val in self.seek(prefix):
                yield (key if decode_keys else key.encode("utf-8")), val
        else:
            for key, val in self.db.items():
                yield (key.decode("utf-8") if decode_keys else key), val

        ###                 ##
        #   REPO INTERFACE   #
        ###                 ##
            
    def add_records(self, objects: List[BaseModel], run_index:bool=True):
        """add many nodes with embeddings - True run index for testing only"""
        if not isinstance(objects,list):
            objects = [objects]
            
        self.index.add_records(objects)
        
        grouped = defaultdict(list)
        for obj in objects:
            grouped[(obj.get_model_namespace(), obj.get_model_name())].append(obj)

        for (namespace, entity), objs in grouped.items():
            for obj in objs:
                """TODO: we have an opportunity to think about how we want to wrap entities here and on the get entity abstraction"""
                self.db[self.model_logic.get_hash_id(obj.id)] = obj.model_dump_json().encode("utf-8")

            # Store embeddings if enabled
            """TODO: hard coded on description of testing - model.get_embedding_info"""
            for embedding_field in ['description']:
                texts = [getattr(o,embedding_field) for o in objs]
                obj_ids = [str(o.id) for o in objs]
                """project against empty strings"""
                vectors = self._get_embeddings(texts)
                for obj_id, vector in zip(obj_ids, vectors):
                    vkey = self.model_logic.get_vector_id(obj_id)
                    self.db[vkey] = self.encode_vector(vector)
        
        if run_index:
            print("Indexing records - will add partitions for new vectors and relational predicates (hot)")
            #self.index.run_partial()
            self.index.run()
            
    def get_by_id(self, id: str) -> Optional[T]:
        """Get a record by ID - the id is our business id not the rocks KEY"""
        key = self.model_logic.get_hash_id(id)
        data = self._get_raw(key)
        if data:
            return self.model_logic.deserialize(data)
        return None
    
    def get_entities(self, names: Union[str, List[str]]) -> Dict[str, List[Dict[str, Any]]]:
        """Get entities by their names from the global index"""
        names = _ensure_list(names)
        result = defaultdict(list)
        
        for name in names:
            key = self.model_logic.get_global_id(name)
            data = self._get_raw(key)
            if not data:
                continue
                
            try:
                mapping = json.loads(data.decode("utf-8"))
                for entity_type, meta in mapping.items():
                    entity_key = meta["key"]
                    entity_data = self._get_raw(entity_key.encode("utf-8"))
                    if entity_data:
                        result[entity_type].append(json.loads(entity_data.decode("utf-8")))
            except Exception as e:
                print(f"Error processing entity name '{name}': {e}")
                
        return dict(result)
    
    def _search_single_question(self, 
                          question: str, 
                          strategies: List[str],
                          ids_only: bool = False) -> Dict[str, Any]:
        """Search entities using multiple strategies for a single question
        
        This internal function handles searching with a single question across
        multiple strategies, providing a foundation for future parallel execution.
        
        Args:
            question: Natural language question to search for
            strategies: List of strategies to use
            ids_only: If True, return only IDs instead of full records
                        
        Returns:
            Dict with results from each strategy for this specific question
        """
        question_results = {}
        
        # Execute each strategy (these will be parallelized in future)
        if 'semantic' in strategies:
            try:
                question_results['semantic'] = self._search_semantic_index_single(question, top_k=5, ids_only=ids_only)
            except Exception as e:
                print(f"Error in semantic search for '{question}': {e}")
                question_results['semantic'] = f"Error: {e}"
        
        if 'predicate' in strategies:
            try:
                question_results['predicate'] = self._search_predicate_index_single(question, sample_size=5, ids_only=ids_only)
            except Exception as e:
                print(f"Error in predicate search for '{question}': {e}")
                question_results['predicate'] = f"Error: {e}"
        
        if 'graph' in strategies:
            seed_keys = []
            # If we have semantic results, use them as seed keys
            if 'semantic' in question_results and isinstance(question_results['semantic'], list) and question_results['semantic']:
                if ids_only:
                    # Extract IDs from ID tuples
                    seed_keys = [id_tuple[0] for id_tuple in question_results['semantic']]
                else:
                    # Extract IDs from full records
                    seed_keys = [record['id'] for record in question_results['semantic'] if 'id' in record]
            
            if seed_keys:
                try:
                    question_results['graph'] = self._search_graph_index(seed_keys, ids_only=ids_only)
                except Exception as e:
                    print(f"Error in graph search using seed keys from '{question}': {e}")
                    question_results['graph'] = f"Error: {e}"
                    
        return question_results
    
    def hybrid_semantic_search(self, 
                          query: str, 
                          top_k: int = 5, 
                          ids_only: bool = False) -> Union[List[Dict], List[Tuple[str, float]]]:
        """Search using both the HNSW index and partial vector index
        
        This combines the results from both the persisted HNSW index and hot partial vectors
        that haven't yet been incorporated into the index. This hybrid approach ensures
        that newly added vectors are included in search results without waiting for the
        full index to be rebuilt.
        
        The function:
        1. Gets results from the HNSW index (older, persisted data)
        2. Gets results from partial vectors (newer, not yet indexed data)
        3. Combines and ranks results by similarity score
        4. Returns the top_k results
        
        This approach is designed for a system where full index updates happen periodically
        (e.g., every 12 hours), but where new data needs to be searchable immediately.
        The partial vector store acts as a hot cache that will eventually be folded into
        the persistent HNSW index during the next scheduled reindexing.
        
        Args:
            query: Natural language query to search for
            top_k: Number of results to return
            ids_only: If True, return only IDs with similarity scores instead of full records
            
        Returns:
            List of matching records (dicts) or (id, score) tuples if ids_only=True
        """
        # Get embedding for the query
        embedding = self._get_embeddings([query])[0]
        embedding_array = np.array(embedding, dtype=np.float32)
        
        # Combined results dictionary (obj_id -> similarity_score)
        combined_results = {}
        
        # 1. Get results from the HNSW index first
        # Check if HNSW index exists
        idx_path = self.index.get_path("semantic", self.model_logic.namespace, self.model_logic.name)
        
        if idx_path.exists():
            # Use HNSW index for efficient search
            idx = hnswlib.Index(space='cosine', dim=embedding_array.shape[0])
            idx.load_index(str(idx_path))
            
            # Query the index
            labels, distances = idx.knn_query(embedding_array.reshape(1, -1), k=top_k*2)  # Get more results to combine
            
            # Convert distances to similarity scores and add to combined results
            for i, lbl in enumerate(labels[0]):
                # Get the original ID from the mapping
                map_key = self.model_logic.get_sem_index_map_id(str(lbl))
                oid_bytes = self._get_raw(map_key)
                
                if oid_bytes:
                    oid = oid_bytes.decode()
                    similarity = 1.0 - distances[0][i]  # Convert distance to similarity score
                    combined_results[oid] = float(similarity)
        
        # 2. Get results from partial vectors (hot vectors not yet in HNSW index)
        partial_vectors = self.index.get_partial_vectors(
            namespace=self.model_logic.namespace,
            entity=self.model_logic.name
        )
        
        # Process each partial vector using cosine similarity
        if partial_vectors:
            from sklearn.metrics.pairwise import cosine_similarity
            
            for obj_id, vector in partial_vectors:
                # Skip if already in combined results (HNSW index takes precedence)
                if obj_id in combined_results:
                    continue
                    
                # Calculate similarity score
                vec_array = vector.reshape(1, -1)
                similarity = cosine_similarity(embedding_array.reshape(1, -1), vec_array)[0][0]
                
                # Add to combined results
                combined_results[obj_id] = float(similarity)
        
        # If no results found and no HNSW index exists, fall back to direct vector search
        if not combined_results and not idx_path.exists():
            direct_results = self._search_vectors_direct(
                query_vector=embedding,
                top_k=top_k,
                ids_only=True
            )
            
            for obj_id, score in direct_results:
                combined_results[obj_id] = score
        
        # Return the top_k results sorted by similarity score
        sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        if ids_only:
            return sorted_results  # List of (id, score) tuples
        else:
            # Convert IDs to full records
            records = []
            for obj_id, score in sorted_results:
                record = self.get_by_id(obj_id)
                if record:
                    result_dict = record.model_dump()
                    result_dict["_similarity"] = float(score)
                    records.append(result_dict)
            return records  # List of record dictionaries with similarity scores
            
    def hybrid_semantic_search_keys_only(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search using both the HNSW index and partial vector index, returning only keys
        
        This is a convenience method that calls hybrid_semantic_search with ids_only=True.
        It returns only the IDs and similarity scores without retrieving the full records.
        
        Args:
            query: Natural language query to search for
            top_k: Number of results to return
            
        Returns:
            List of (id, score) tuples for the most similar items
        """
        return self.hybrid_semantic_search(query, top_k, ids_only=True)
    
    def search(self, 
              questions: Union[str, List[str]], 
              strategies: List[str] = None,
              ids_only: bool = False,
              combine_results: bool = False) -> Dict[str, Any]:
        """Search entities using multiple strategies
        
        Args:
            questions: Natural language question(s) to search for
            strategies: List of strategies to use. If None, uses all available
                        Valid values: 'semantic', 'predicate', 'graph'
            ids_only: If True, return only IDs instead of full records
            combine_results: If True and ids_only=True, combine results from
                            different strategies into a single set of IDs
                        
        Returns:
            Dict with results from each strategy, or combined results if requested
        """
        questions = _ensure_list(questions)
        
        if strategies is None:
            strategies = ['semantic', 'predicate']  # Default strategies
        
        # Process each question and collect results
        results = {}
        all_ids = set()  # For combining results
        
        # Process each question individually (future: parallelize this loop)
        for question in questions:
            question_results = self._search_single_question(question, strategies, ids_only)
            
            # Organize results by strategy for client compatibility
            for strategy, strategy_results in question_results.items():
                if strategy not in results:
                    results[strategy] = {}
                results[strategy][question] = strategy_results
                
                # Collect IDs for combining if requested
                if ids_only and combine_results:
                    if strategy == 'semantic' and isinstance(strategy_results, list):
                        all_ids.update(id_tuple[0] for id_tuple in strategy_results)
                    elif strategy == 'predicate' and isinstance(strategy_results, list):
                        all_ids.update(strategy_results)
                    elif strategy == 'graph' and isinstance(strategy_results, list):
                        all_ids.update(strategy_results)
        
        # If combining results, replace individual strategy results with combined set
        if ids_only and combine_results:
            if all_ids:
                # Get full records if requested
                if not ids_only:
                    records = []
                    for id in set(all_ids):
                        record = self.get_by_id(id)
                        if record:
                            records.append(record.model_dump())
                    return {"combined": records}
                else:
                    return {"combined": list(all_ids)}
                
        return results
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for text using OpenAI API
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        
        Raises:
            EnvironmentError: If OpenAI API key is missing
            requests.RequestException: If API call fails
        """
        if not OPENAI_API_KEY:
            raise EnvironmentError("Missing OPENAI_API_KEY in environment variables")
            
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {"input": texts, "model": EMBEDDING_MODEL}
        
        response = requests.post(OPENAI_EMBEDDING_URL, headers=headers, json=data)
        response.raise_for_status()
        
        return [d["embedding"] for d in response.json()["data"]]
    
    def _get_sql_from_openai(self, prompt: str) -> str:
        """Get SQL clause from OpenAI
        
        Args:
            prompt: The prompt to send to OpenAI
            
        Returns:
            SQL WHERE clause as string
        """
        if not OPENAI_API_KEY:
            raise EnvironmentError("Missing OPENAI_API_KEY in environment variables")
            
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}", 
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": OPENAI_SQL_MODEL,
            "messages": [
                {"role": "system", "content": "You are an SQL assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
        
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        
        return resp.json()["choices"][0]["message"]["content"].strip().rstrip(';')
    
    def _search_predicate_index_single(self, question: str, sample_size: int = 5, ids_only: bool = False):
        """Search using SQL predicates on the DuckDB-backed Parquet index for a single question
        
        Uses OpenAI to convert natural language to SQL WHERE clauses
        
        Args:
            question: Natural language question to convert to SQL
            sample_size: Maximum number of results to return
            ids_only: If True, return only IDs instead of full records
            
        Returns:
            Either a list of IDs, a Polars DataFrame, or an error string
        """
       
        # Generate SQL WHERE clause using OpenAI
        schema_json = self.model_cls.model_json_schema()
        prompt = (
            f"Given a table with schema from the Parquet file, "
            f"write a SQL WHERE clause (no SELECT) to answer the question: '{question}'. "
            f"Do not fence the answer simply return WHERE [VALID PREDICATES] and use the agent schema here ```json{schema_json}```"
        )
        
        where_clause = self._get_sql_from_openai(prompt)
        
        # Execute the query via DuckDB
        conn = duckdb.connect(database=":memory:")
        
        # Select only ID column if ids_only is True
        select_clause = "id" if ids_only else "*"
        
        # Use a wildcard to read both the main parquet file and any temp parquet files
        # DuckDB will scan all matching files and combine the results
        parquet_dir = self.index.get_path("parquet_dir", self.model_logic.namespace)
        query = f"""
        SELECT {select_clause} 
        FROM read_parquet('{parquet_dir}/{self.model_logic.name}/*.parquet') 
        {where_clause} 
        LIMIT {sample_size};
        """
        
        print(f"Executing query: {query}")
        
        try:
            df = conn.execute(query).pl()
            
            if ids_only:
                # Return just list of IDs
                return df['id'].to_list()
            else:
                return df
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return f"Error: {e}"

    def _search_predicate_index(self, questions: Union[str, List[str]], sample_size: int = 5, ids_only: bool = False) -> Dict[str, Any]:
        """Search using SQL predicates on the DuckDB-backed Parquet index
        
        Uses OpenAI to convert natural language to SQL WHERE clauses
        
        Args:
            questions: Natural language question(s) to convert to SQL
            sample_size: Maximum number of results to return
            ids_only: If True, return only IDs instead of full records
            
        Returns:
            Dict mapping questions to Polars DataFrames or ID lists
        """
        questions = _ensure_list(questions)
        results = {}
        
        # Future: This loop can be parallelized
        for question in questions:
            results[question] = self._search_predicate_index_single(question, sample_size, ids_only)
            
        return results
    
    def _search_semantic_index_single(self, question: str, top_k: int = 5, ids_only: bool = False) -> List[Any]:
        """Search using semantic vectors for a single question
        
        Embeds question and finds nearest vectors in the semantic index
        
        Args:
            question: Natural language question to search for
            top_k: Number of results to return
            ids_only: If True, return only IDs with similarity scores instead of full records
            
        Returns:
            List of matching records or ID tuples
        """
        # Get embedding for the question
        embedding = self._get_embeddings([question])[0]
        embedding_array = np.array(embedding, dtype=np.float32)
        
        # Check if we should use the HNSW index or direct comparison
        idx_path = self.index.get_path("semantic", self.model_logic.namespace, self.model_logic.name)
        
        if idx_path.exists():
            # Use HNSW index for efficient search
            idx = hnswlib.Index(space='cosine', dim=embedding_array.shape[0])
            idx.load_index(str(idx_path))
            
            # Query the index
            labels, distances = idx.knn_query(embedding_array.reshape(1, -1), k=top_k)
            
            # Get actual records or just IDs
            question_results = []
            for i, lbl in enumerate(labels[0]):
                # Get the original ID from the mapping
                map_key = self.model_logic.get_sem_index_map_id(str(lbl))
                oid_bytes = self._get_raw(map_key)
                
                if oid_bytes:
                    oid = oid_bytes.decode()
                    similarity = 1.0 - distances[0][i]  # Convert distance to similarity score
                    
                    if ids_only:
                        # Return ID and similarity score as tuple
                        question_results.append((oid, float(similarity)))
                    else:
                        # Get full record
                        record = self.get_by_id(oid)
                        if record:
                            result_dict = record.model_dump()
                            result_dict["_similarity"] = float(similarity)
                            question_results.append(result_dict)
            
            return question_results
        else:
            # Fall back to direct vector comparison
            if ids_only:
                return self._search_vectors_direct(embedding, top_k, ids_only=True)
            else:
                records = self._search_vectors_direct(embedding, top_k)
                return [r.model_dump() for r in records]
    
    def _search_semantic_index(self, questions: Union[str, List[str]], top_k: int = 5, ids_only: bool = False) -> Dict[str, Any]:
        """Search using semantic vectors for multiple questions
        
        Embeds questions and finds nearest vectors in the semantic index
        
        Args:
            questions: Natural language question(s) to search for
            top_k: Number of results to return per question
            ids_only: If True, return only IDs with similarity scores instead of full records
            
        Returns:
            Dict mapping questions to lists of matching records or ID tuples
        """
        questions = _ensure_list(questions)
        results = {}
        
        # Future: This loop can be parallelized
        for question in questions:
            results[question] = self._search_semantic_index_single(question, top_k, ids_only)
        
        return results
    
    def _get_vector_data(self) -> Tuple[List[np.ndarray], List[str], List[str]]:
        """Retrieves all vectors and their associated IDs and keys
        
        This function extracts all vectors for the current entity type along with
        their identifying information, making it reusable for various search needs.
        
        Returns:
            Tuple containing:
            - List of vector arrays
            - List of object IDs corresponding to each vector
            - List of key strings for retrieving full records
        """
        vectors = []
        keys = []
        ids = []
        
        # Get all vectors for this entity type
        prefix = self.model_logic.vector_prefix
        
        for key, val in self.iter_items(prefix):
            try:
                vec = self.decode_vector(val)
                vectors.append(vec)
                
                # Extract ID from key
                parts = key.split(':') if isinstance(key, str) else key.decode('utf-8').split(':')
                if len(parts) >= 4:
                    obj_id = parts[3]
                    ids.append(obj_id)
                
                # Convert vector key to hash key for fetching full records
                hash_key = key.replace('vindex', 'hash')
                keys.append(hash_key)
            except Exception as e:
                print(f"Error decoding vector at {key}: {e}")
                
        return vectors, ids, keys
    
    def _search_vectors_direct(self, query_vector: List[float], top_k: int = 5, ids_only: bool = False) -> Union[List[T], List[Tuple[str, float]]]:
        """Search vectors directly without using the HNSW index
        
        Used as a fallback when the index doesn't exist or for small datasets
        
        Args:
            query_vector: The vector to search for
            top_k: Number of results to return
            ids_only: If True, return only IDs with similarity scores instead of full records
            
        Returns:
            List of matching records or (id, score) tuples if ids_only=True
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get all vectors and their metadata
        vectors, ids, keys = self._get_vector_data()
        
        if not vectors:
            return []
            
        # Stack vectors and compute similarities
        vec_matrix = np.stack(vectors)
        input_vec = np.array(query_vector).reshape(1, -1)
        scores = cosine_similarity(input_vec, vec_matrix)[0]
        
        # Get top k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        if ids_only:
            # Return (id, score) tuples
            return [(ids[i], float(scores[i])) for i in top_indices]
        else:
            # Return full records
            results = []
            for i in top_indices:
                record_key = keys[i] if isinstance(keys[i], bytes) else keys[i].encode("utf-8")
                record_data = self._get_raw(record_key)
                if record_data:
                    record = self.model_logic.deserialize(record_data)
                    # Could add similarity score to record if needed
                    results.append(record)
                    
            return results
    
    def _search_graph_index(self, seed_keys: Union[str, List[str]], distance: int = 2, ids_only: bool = False) -> Union[Dict[str, Any], List[str]]:
        """Search using graph traversal
        
        Explores the graph neighborhood of seed keys
        
        Args:
            seed_keys: Starting node IDs
            distance: Maximum distance to explore
            ids_only: If True, return only connected node IDs instead of full graph
            
        Returns:
            Dict containing the neighborhood graph or just a list of connected IDs
        """
        seed_keys = _ensure_list(seed_keys)
        neighborhood = self.graph.get_neighbourhood(seed_keys, distance=distance)
        
        if ids_only:
            # Return just the discovered node IDs (excluding seed keys if desired)
            connected_ids = set(neighborhood.keys())
            seed_key_set = set(seed_keys)
            # Optionally remove seed keys from result
            # connected_ids = connected_ids - seed_key_set
            return list(connected_ids)
        else:
            return neighborhood
    
        ###                 ##
        #   UTILS   #
        ###                 ##
        
    def encode_vector(self, vector: Union[List[float], np.ndarray]) -> bytes:
        """Encode a vector to bytes"""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        return vector.tobytes()
    
    def decode_vector(self, data: bytes) -> np.ndarray:
        """Decode bytes to a vector"""
        return np.frombuffer(data, dtype=np.float32)
        
