"""first sketch of the minimal database, we will break it up and make it better in the library"""

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


T = TypeVar("T", bound=BaseModel)

OPENAI_SQL_MODEL = 'gpt-4.1-mini'
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-ada-002"
DB_PATH = '/Users/sirsh/.percolate-fabric/db'

def sha256_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

class RocksPydanticRepository:
    def __init__(self, db_path: str = DB_PATH, embedding_enabled: bool = True):
        self._path = db_path
        self.db = Rdict(f"{self._path}/data") #maybe but rocks inside ./data and the index adjacently   
        self.embedding_enabled = embedding_enabled

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

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
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

    def add(self, obj: BaseModel):
        # Store model JSON
        key = self._hash_key(obj)
        self.db[key] = self._serialize(obj)

        # Optionally store embeddings
        if self.embedding_enabled:
            vector = self._get_embedding([json.dumps(obj.model_dump())])[0]
            vkey = self._vector_key(obj)
            self.db[vkey] = self.encode_vector(vector)

    def add_many(self, objects: List[BaseModel]):
        grouped = defaultdict(list)
        for obj in objects:
            grouped[(obj.get_model_namespace(), obj.get_model_name())].append(obj)

        for (namespace, entity), objs in grouped.items():
            # Store each model
            for obj in objs:
                key = f"{namespace}:{entity}:hash:{obj.id}".encode("utf-8")
                self.db[key] = obj.model_dump_json().encode("utf-8")

            # Store embeddings if enabled
            if self.embedding_enabled:
                """hard coded on description of testing"""
                texts = [o.description for o in objs]
                obj_ids = [str(o.id) for o in objs]
                vectors = self._get_embedding(texts)
                for obj_id, vector in zip(obj_ids, vectors):
                    vkey = f"{namespace}:{entity}:vindex:{obj_id}".encode("utf-8")
                    self.db[vkey] = self.encode_vector(vector)

    def get_by_id(self, id: str, model_cls: Type[T]) -> Optional[T]:
        model_name = model_cls.get_model_name()
        namespace = model_cls.get_model_namespace()
        
        key = f"{namespace}:{model_name}:hash:{id}".encode("utf-8")
        value = self.db.get(key)
        return self._deserialize(value, model_cls) if value else None

    def predicate_search(self, model_cls: Type[T], predicate: str) -> List[T]:
        results = []
        for key, val in self.db.items():
            skey = key.decode("utf-8")
            if skey.startswith(f"{model_cls.get_model_namespace()}:{model_cls.get_model_name()}:hash:"):
                obj = json.loads(val.decode("utf-8"))
                if eval(predicate, {}, {"item": obj}):
                    results.append(model_cls.model_validate(obj))
        return results

    def build_predicate_indexes(self):
        """the predicate index can include the enums and other query helper metadata"""
        grouped_data = defaultdict(list)
        for key, val in self.db.items():
            skey = key.decode("utf-8")
            parts = skey.split(":")
            if len(parts) == 4 and parts[2] == "hash":
                namespace, table, _, obj_id = parts
                grouped_data[(namespace, table)].append(json.loads(val.decode("utf-8")))

        for (namespace, table), records in grouped_data.items():
            df = pl.DataFrame(records)
          
            out_dir = Path(f"{self._path}/index/structures/{namespace}")
            out_dir.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_dir / f"{table}.parquet")
            print(f"✅ Wrote index to {out_dir / f'{table}.parquet'}")


    def build_semantic_indexes(self, dim: int):
        """save index - we should store the index per dim
        - need to understand how long this takes at scale because tradeoff would be to keep a hot list of new vectors and compare them to the result set
        - the hot vector list can be a parameter and we can save this using some sort of numpy like vector format and do an exact NN match - when that thing gets over threshold,
        we build the index and the replace the old file with the new index (there may be new x.index added but we should lock the temp one for writing but still read)
        """
        grouped_vecs = defaultdict(list)
        grouped_ids = defaultdict(list)

        for key, val in self.db.items():
            parts = key.decode().split(":")
            if len(parts) == 4 and parts[2] == "vindex":
                ns, ent, _, oid = parts
                vec = np.frombuffer(val, dtype=np.float32)
                if vec.shape[0] != dim:
                    print(f"❌ Dimension mismatch {key}")
                    continue
                grouped_vecs[(ns, ent)].append(vec)
                grouped_ids[(ns, ent)].append(oid)

        for (ns, ent), vecs in grouped_vecs.items():
            arr = np.vstack(vecs)
            ids = []
            for oid in grouped_ids[(ns, ent)]:
                hash_id = str(int(hashlib.sha1(oid.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF)
                key = f"{ns}:{ent}:sem_index_map:{hash_id}".encode()
                self.db[key] = oid.encode()
                ids.append(int(hash_id))

            idx = hnswlib.Index(space='cosine', dim=dim)
            idx.init_index(max_elements=arr.shape[0], ef_construction=200, M=16)
            idx.add_items(arr, np.array(ids, dtype=np.int64))
            idx.set_ef(50)

            out = Path(f"{self._path}/index/semantic/{ns}")
            out.mkdir(parents=True, exist_ok=True)
            idx.save_index(str(out / f"{ent}.bin"))

            print(f"✅ Saved HNSW index: {out / f'{ent}.bin'}")

    def semantic_search(self, query: str, model_cls: Type[T], top_k: int = 4) -> List[T]:
        """search a type by a natural language query - we load an index from disk for now and them recover the mapping to rocks db keys"""
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
        
        the model we have options for example sometimes a predicate search makes no sense for the model and if there are no embeddings then we should not do vector search or sometimes the user may just want to decide
        
        - trick things
        -- how would we know if a non predicate search returns records that are not temporally relevant - the semantic and entity nodes could have mod dates
        """
        
        pass