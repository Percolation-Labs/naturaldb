# Percolate-fabric: A lightweight personal AI-ready database that runs anywhere

A database needs to run anywhere and we are building for a world where users own their data. Internet scale is not important but efficiency and portability are. For example [this](https://github.com/marykdb/rocksdb-multiplatform) is a KMP Rocks project.
From working with generative AI, the need for a variety of indexes has become important. We need all of:
- key-value lookup
- graph path finding
- semantic search
- SQL predicate search

The Percolate-fabric database aims to push down a composite index to the lowest level to return documents that match keys by a number of criteria.

## Features

- Automatic background indexing
- Multi-modal search (semantic, predicate, graph)
- Pydantic integration for schema validation
- RocksDB for efficient key-value storage
- HNSW for vector similarity search
- DuckDB for SQL predicate querying

There are two types of queries; ones that evaluate to row level entities in which case a weight key matrix is used to retrieve documents based on a multiple conditions.
Secondly we have aggregate queries. We take an OLAP approach to this which reduces these to the first type.

Metadata is associated with each key space. We use the idea of namespaces and tables to describe key spaces and these are given rich "prompt like" descriptions that show their purpose. We also provide rich metadata on key space attributes which are like columns.
This means tables are in some sense "agentic" and can map 1:1 with agents.

Key spaces naturally have user id, created, modified, deleted at system attributes which can be used in different ways. keys and labels are required on all entities but they can be hidden from the user.

## Portability

- It should be possible to install percolate-fabric anywhere even via WebAssembly in the browser which might require a different approach to rocks
- 

## Setup
brew install rocks db and clone and make https://github.com/nmslib/hnswlib.git

```bash
git clone https://github.com/nmslib/hnswlib.git
cd hnswlib
# Build hnswlib
mkdir build && cd build
cmake ..
make
```

Python bindings

```bash
pip install hnswlib
```


Install rocks python

```bash
pip install git+https://github.com/gau-nernst/rocksdb-python.git
```

### Duck DB

When we start up the rocks database, we register all the namespaces and tables in the Duck Connection to allow for querying naturally

we connect to duck and 
```python
con.execute("CREATE SCHEMA IF NOT EXISTS my_schema")
con.register("my_schema.my_table", "path/to/parquet/folder/*.parquet")
```

We could probably expose and ODBC or some sort of interface like that so folks can query the database using the SQL dialect - we should add some system tables too

## Fix

01. Add the global key index i.e. global:key->(unique_key,entity) -> this indexing can be generated to add graph paths but by default it just adds the node
03. it may be useful to literally "index" predicate fields and not just store big chunky data that cannot be queried anyway e.g. big json files and the like. that way we can reduce space and still be useful - drop complex types and stuff that is indexed
04. should we wrap objects in system metadata or update the interface - for example we need timestamps and user id - we want them on dicts but also predicate indexes  

## Features

00. The duck client should register all the types so we can query them all e..g select * from p8Agent 
01. Multi-question, Multi-index search 
02. Graph and path finding
03. Entity registration flow
04. Tool registration flow
05. Worker process to build indexes/ there will be a need for enums in the index builder too: index(object).add_records(records), .run() ->runs the index according to the spec on a thread after inserting record batch
06. Simple interface add/update keys or search
07. Testing limits / C++ implementation and re-interface
08. Schema migration on the parquet re-index 
09. Distributed synchronization of keys and index rebuilds
10. Easy package and install anywhere
11. index images
12. daily snapshots 
13. scheduler is an important task for ingesting data or doing things for the user


# Examples

- conversation modeling
- function planning
- email, image, doc and site ingestion
- tasks and goals


# Is it a feature?
01. You can insert whatever you want and indexing and schema migration happens in the background| you dont need to register any types - risk of bad states
02. At least once you have to post the object with the entire schema - this is like having the pydantic object with all the metadata as opposed to just the dict of model dump. its fine to just write the dicts but we cannot build indexes without the metadata so they need to be added at least once



# Background Indexing Options

Percolate-fabric provides two approaches for background indexing:

## 1. Separate Process Mode (Original Approach)

This approach uses a separate process for the indexer:

```python
# Client process
client_repo = RocksRepo(model_cls=Agent, db_path="./data", client_mode=True)
client_repo.add_records(objects)

# Server process (in a different Python process)
server_repo = RocksRepo(model_cls=Agent, db_path="./data", client_mode=False)
server_repo.index.process_queue_continuously()
```

**Pros**: Clear separation of concerns  
**Cons**: Cannot share RocksDB instances between processes

## 2. Background Thread Mode (New Approach)

This approach uses a background thread within the same process:

```python
# Single process with background thread
repo = RocksRepo(model_cls=Agent, db_path="./data", use_background_thread=True)

# Add records - they will be automatically processed by the background thread
repo.add_records(objects)

# Later, when shutting down
repo.index._stop_background_thread()
```

**Pros**: 
- Shares the same RocksDB instance
- No need for separate processes
- Simpler implementation for most use cases

**Cons**:
- Thread safety concerns if directly accessing the database while indexing

# Goals

Sunday April 21: Test that the hybrid queries work for semantic or predicate index. Test get entities work.
- agents/entities and functions are added for agentic orchestration purposes
- planner can suggest plan
- session works - plan executes, runs functions and saves sessions and AIResponses

Sunday April 28: Test integration with percolate

- files and images can be added


Sunday May 5:
- email integration and docs integration works
- I can search the web and create tasks for research


--------------------

define interface and add tests before building the C++ version

--------------------


Build C++ version and package for desktop / consider mobile versions + DB should run and need vector indexing