# Percolate-Fabric Project Guidance

## Build & Development
```bash
python -m pip install -e .  # Install in dev mode
python main.py              # Run main sample
python try_prefix.py        # Test prefix functionality
```

## Testing
No testing commands found in codebase. Add pytest for testing.

## Project Structure
- `naturaldb/`: Core database implementation
  - `db.py`: Main database class (RocksPydanticRepository)
  - `repo.py`: Repository interface abstraction
  - `index/`: Background indexing system

## Code Style
- Use type hints for all functions and variables
- Follow PEP 8 style guidelines
- Import order: standard lib → third party → local
- Prefer explicit naming over abbreviations
- Error handling: Use try/except with specific exceptions
- Document all classes and public methods with docstrings
- Use Pydantic for data modeling and validation

## Dependencies
- RocksDB backend for key-value storage
- HNSW for vector indexing
- DuckDB for SQL predicate queries
- Pydantic for data modeling
- OpenAI for embeddings generation