import os
import hashlib
import json
import requests
from typing import List, Optional, Type, TypeVar
from pydantic import BaseModel
import polars as pl
from collections import defaultdict
import numpy as np
from pathlib import Path
import hnswlib
from rocksdict import Rdict  
import duckdb
from datetime import datetime
from .index.Backgroundindexer import Indexer
T = TypeVar("T", bound=BaseModel)

OPENAI_SQL_MODEL = 'gpt-4.1-mini'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-ada-002"
DB_PATH = str(Path.home() / '.percolate' / 'natural-db')

def sha256_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class ModelLogic:
    """conventions for key names mainly plus some encoding (serialize) stuff to interface with rocks"""
    def __init__(self, model:  Type[T]):
        """"""
        self.model = model
        
    @property
    def namespace(self):
        return self.model_cls.get_model_name()
    @property
    def name(self):
        return self.model_cls.get_model_namespace()
        
    def get_hash_id(self,id, encode=True):
        k = f"{self.namespace}:{self.name}:hash:{id}"
        return k if not encode else k.encode("utf-8")
    
    def get_vector_id(self,id, encode=True):
        k = f"{self.namespace}:{self.name}:vindex:{id}"
        return k if not encode else k.encode("utf-8")
    
    def wrap_system(self, model:BaseModel|dict):
        """
        wrap in system fields 
        """
        pass
    
    """add prefixes - VINDEX, HINDEX, EDGE"""
    
    @property
    def encoded_model(self):
        """the encoded model that we store
        in future we may used a wrapped model
        """
        return self.model.model_dump_json().encode("utf-8")
    
class RocksPydanticRepository:
    def __init__(self, db_path: str = DB_PATH, embedding_enabled: bool = True):
        self._path = db_path
        self.db = Rdict(f"{self._path}/data") #maybe but rocks inside ./data and the index adjacently   
        self.embedding_enabled = embedding_enabled
        self.index = Indexer(self)
        
    def _key_prefix(self, obj: BaseModel) -> str:
        return f"{obj.get_model_namespace()}:{obj.get_model_name()}"

    def _hash_key(self, obj: BaseModel) -> bytes:
        return f"{self._key_prefix(obj)}:hash:{obj.id}".encode("utf-8")

    def _vector_key(self, obj: BaseModel) -> bytes:
        return f"{self._key_prefix(obj)}:vindex:{obj.id}".encode("utf-8")

    def _serialize(self, obj: BaseModel) -> bytes:
        return obj.model_dump_json().encode("utf-8")

    def _deserialize(self, data: bytes, model_cls: Type[T]) -> T:
        return model_cls.model_validate_json(data.decode("utf-8"))
    
    def _list_keys(self):
        for key, val in self.db.items():
            print(key)

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        """use the default embedding provider to index content"""
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

    def encode_vector(self, vector) -> bytes:
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        return vector.tobytes()

    def add_records(self, objects: List[BaseModel], run_index:bool=True):
        """add many nodes with embeddings - True run index for testing only"""
        if not isinstance(objects,list):
            objects = [objects]
        self.index.add_records(objects)
        
        grouped = defaultdict(list)
        for obj in objects:
            grouped[(obj.get_model_namespace(), obj.get_model_name())].append(obj)

        for (namespace, entity), objs in grouped.items():
            # Store each model
            for obj in objs:
                key = f"{namespace}:{entity}:hash:{obj.id}".encode("utf-8")
                """TODO: we have an opportunity to think about how we want to wrap entities here and on the get entity abstraction"""
                self.db[key] = obj.model_dump_json().encode("utf-8")

            # Store embeddings if enabled
            if self.embedding_enabled:
                """TODO: hard coded on description of testing"""
                texts = [o.description for o in objs]
                obj_ids = [str(o.id) for o in objs]
                vectors = self._get_embedding(texts)
                for obj_id, vector in zip(obj_ids, vectors):
                    vkey = f"{namespace}:{entity}:vindex:{obj_id}".encode("utf-8")
                    self.db[vkey] = self.encode_vector(vector)
        
        if run_index:
            print("Indexing records")
            """for testing this is fine but we should schedule the indexer"""
            self.index.run()

    def get_by_id(self, id: str, model_cls: Type[T]) -> Optional[T]:
        """ket lookup"""
        model_name = model_cls.get_model_name()
        namespace = model_cls.get_model_namespace()
        
        key = f"{namespace}:{model_name}:hash:{id}".encode("utf-8")
        value = self.db.get(key)
        return self._deserialize(value, model_cls) if value else None

    def predicate_search(self, model_cls: Type[T], predicate: str) -> List[T]:
        """
        scan items with templating matching - its better to use the sql predicate index of course and this is just for testing        
        """
        results = []
        """TODO this will have to be replaced with a prefix scan"""
        for key, val in self.db.items():
            skey = key.decode("utf-8")
            if skey.startswith(f"{model_cls.get_model_namespace()}:{model_cls.get_model_name()}:hash:"):
                obj = json.loads(val.decode("utf-8"))
                if eval(predicate, {}, {"item": obj}):
                    results.append(model_cls.model_validate(obj))
        return results

 
    def get_entities(self, names: List[str]) -> dict:
        """we store nodes as connected nodes to the global grouped into typed entities"""
        from collections import defaultdict

        result = defaultdict(list)
        if not isinstance(names,list):
            names = [names]
        for name in names:
            key = f"global:node:{name}".encode("utf-8")
            data = self.db.get(key)
            if not data:
                continue 

            try:
                mapping = json.loads(data.decode("utf-8"))
                for entity_type, meta in mapping.items():
                    """load the raw entity by the raw key"""
                   
                    entity = self.db.get(meta["key"].encode("utf-8"))
                    if entity:
                        result[entity_type].append(json.loads(entity.decode("utf-8")))
            except Exception as e:
                print(f"Error processing entity name '{name}': {e}")

        return dict(result)
        
 
    def get_edges(self, node_id: str, namespace: str = "concept") -> List[str]:
        """
        rocks db will allow prefix matching so we can traverse edges originating from a node
        A tree algorithms classes will use this to walk the neighbourhood using concept nodes
        """
        prefix = f"{namespace}:edge:{node_id}:".encode("utf-8")
        it = self.db.iteritems()
        it.seek(prefix)

        edges = []
        """TODO: this will have to be replaced with a prefix scan"""
        for key, _ in it:
            if not key.startswith(prefix):
                break
            # key format: namespace:edge:src:dst
            try:
                parts = key.decode("utf-8").split(":")
                if len(parts) == 4:
                    edges.append(parts[3])  # dst
            except Exception as e:
                print(f"Error parsing edge key: {e}")
        return edges

    def get_entity_from_local_key(self, encoded_key:str):
        """
        load the entity safely as json
        """
        
        e = self.db.get(encoded_key)
        if e:
            return json.loads(e.decode('utf-8'))
        
    def _search_similar_by_question(self, 
                                 question: str, 
                                 model_cls: Type[T], 
                                 m: int = 1000, 
                                 n: int = 5) -> List[tuple]:
        """
        Embed the question and return top `n` nearest entity vector matches.
        This specifically uses a scan of vectors for the entity and not the stored index
        initial delta when getting an embedding for the question
        """
   
        embedding = self._get_embedding([question])[0]
                
        """gets the keys for this entity e.g.
        b'p8:Agent:vindex:149df01e-0cbc-5521-91d2-e879383274cd'
        """
        results = self.index.get_top_neighbors(model=model_cls, input_vector=embedding, m=m,n=n)
        """^ returns a tuple of entity local keys and distances - we can load the actual entities"""
        results = [(self.get_entity_from_local_key(r[0].encode('utf-8')), r[1]) for r in results]
        
        return results

    def semantic_search(self, query: str, model_cls: Type[T], top_k: int = 3) -> List[T]:
        """search a type by a natural language query - we load an index from disk for now and them recover the mapping to rocks db keys
        
        TODO: in future we will combine searching over recent vectors and index. 
        table statistics can tell us how many records we have and we can use that to decode what to do
        """
        model_name = model_cls.get_model_name()
        namespace = model_cls.get_model_namespace()

        qvec = np.array(self._get_embedding([query])[0], dtype=np.float32)

        idx_path = Path(f"{self._path}/index/semantic/{namespace}/{model_name}.bin")
        if not idx_path.exists():
            raise FileNotFoundError(f"Index not found: {idx_path}")

        idx = hnswlib.Index(space='cosine', dim=qvec.shape[0])
        idx.load_index(str(idx_path))

        labels, _ = idx.knn_query(qvec, k=top_k)

        results = []
        for lbl in labels[0]:
            key = f"{namespace}:{model_name}:sem_index_map:{lbl}".encode()
            if (oid := self.db.get(key)):
                if (obj := self.get_by_id(oid.decode(), model_cls)):
                    results.append(obj)

        return results
    
    def sql_predicate_search(self,
                             question: str,
                             model_cls: Type[T], 
                             sample_size: int = 5) -> List[T]:
        """
        Uses OpenAI agent to convert a natural-language question into a SQL WHERE clause,
        then queries the Parquet file for <namespace>.<entity> via DuckDB.
        
        the prompt will be evolved to use examples and enums per entity model
        
        This will be used to get the keys which we will then use to get the items
        """
        # 1. Build Parquet path
        namespace = model_cls.get_model_namespace()
        entity = model_cls.get_model_name()
        
        parquet_path = f"{self._path}/index/structures/{namespace}/{entity}.parquet"
        if not Path(parquet_path).exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

      
        prompt = (
            f"Given a table with schema from the Parquet file at '{parquet_path}', "
            f"write a SQL WHERE clause (no SELECT) to answer the question: '{question}'. Do not fence the answer simply return WHERE [VALID PREDICATES] and use the agent schema here ```json{model_cls.model_json_schema()}````"
        )
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
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
        where_clause = resp.json()["choices"][0]["message"]["content"].strip().rstrip(';')

        conn = duckdb.connect(database=":memory:")
        query = f"SELECT * FROM read_parquet('{parquet_path}') {where_clause} LIMIT {sample_size};"
        
        print(query)
        
        df = conn.execute(query).pl()

        return df
    
    def search(self, question:str, model=None):
        """perform a multimodal search on multiple threads
        -- get the sql predicates if match
        -- get the semantic search if match (merge keys)
        -- options: allow entity extraction -> would also merge in those keys on its own thread
        -- options: allow second pass graph ngh walker
        -- options: return key information only i.e. defer decision or apply inline deterministically (like just using EXPLAIN modes)
        -- options: allow question decompositions or is there some way to detect that its worth doing?
        
        the model we have options for example sometimes a predicate search makes no sense for the model and if there are no embeddings then we should not do vector search or sometimes the user may just want to decide
        
        - trick things
        -- how would we know if a non predicate search returns records that are not temporally relevant - the semantic and entity nodes could have mod dates
        """
        
        pass