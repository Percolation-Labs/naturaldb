"""
indexer is a background worker that does all of the below on a short delay waiting for things to be pushed to the queue

- adds the entity as a record in both the agents metadata table and the callable functions table
- calls the semantic indexer which uses the hot and col index to build
- registers the entity key in the global index with a pointer to the localized index - this is the global node used for entity resolution and we keep one to many edges for each entity
-- this process also adds graph edges to local entities if enabled 
- builds the predicate index with enums

-- test query planner for multiple index matches on interesting test cases

-- we need to build the concept neighbourhood which shows what is connected and when the given nodes were last updated.

-- write table statistics
"""








"""

      DECOMPOSITION

"""

DECOMPOSE     = 1 << 0  # 00001
SEMANTIC      = 1 << 1  # 00010
SQL_FILTER    = 1 << 2  # 00100
KEYWORD       = 1 << 3  # 01000
HYBRID_FUSION = 1 << 4  # 10000
CONTAINS_ENTITIES = 1 << 5   
GRAPH_PATHS = 1 << 6

import re

def infer_query_flags(query: str) -> int:
    """crude classifier"""
    flags = 0
    query_lower = query.lower()

    # Rule: If conjunctions like "and" or multiple question marks → decomposition
    if ' and ' in query_lower or '?' in query_lower and query_lower.count('?') > 1:
        flags |= DECOMPOSE

    # Rule: If it's about people, entities, general info → semantic
    if any(keyword in query_lower for keyword in ['who is', 'tell me about', 'what is', 'explain']):
        flags |= SEMANTIC

    # Rule: If it's structured/numeric/data-filter-y
    if re.search(r'(how many|last week|between|under \$?\d+|greater than)', query_lower):
        flags |= SQL_FILTER

    # Rule: If it's short and noun-heavy → probably keyword search
    if len(query.split()) <= 5 and not '?' in query:
        flags |= KEYWORD

    # Rule: If multiple intents or needs multiple data sources
    if flags & SEMANTIC and flags & SQL_FILTER:
        flags |= HYBRID_FUSION

    return flags

from pydantic import BaseModel
class QuestionDecomposerAgent(BaseModel):
    """Your job is to return a single int bitmask/flags based on the question complexity. 
    Bitmask/flag that represents which types of search/indexing strategies or steps (like decomposition, semantic search, SQL filter, etc.) should be used to answer the question
    
    DECOMPOSE           = 1 << 0  
    SEMANTIC            = 1 << 1  
    SQL_FILTER          = 1 << 2  
    CONTAINS_ENTITIES   = 1 << 3    
    GRAPH_PATHS         = 1 << 4
        
    Use these rules to determine flags;
    
    - If the questions sounds like it is quantitative its probably an SQL based since semantic searches are not great at this but could be if the answer is clearly in the text. 
    - quantitative questions are 'how many' 'when' 'most' 'least' 'under' 'over' 'between etc.
    - if the questions have conjunctions but seem related they probably dont need to be decomposed but might if one probe could not answer them AND if semantic search is being used
    - a graph query could be used to find relations when a semantic search is used but could be also linked to more interesting entities to go deeper
    - if unique codes and ids are used add flag for contains entities
    - DO NOT treat well-known terms and names as entities because they are not specific enough.

    For the users question return just the flag with no explanation as {"flag": VALUE}. The VALUE should be the integer flag not the binary number
    """
    
    name: str
    
    def _unwrap_flags(flags: int | str | dict) -> dict:
        import json
        if isinstance(flags, str):
            flags = json.loads(flags)['flag']
        
        #print(flags)
        DECOMPOSE           = 1 << 0  
        SEMANTIC            = 1 << 1  
        SQL_FILTER          = 1 << 2  
        CONTAINS_ENTITIES   = 1 << 3    
        GRAPH_PATHS         = 1 << 4

        return {
            "DECOMPOSE":         bool(flags & DECOMPOSE),
            "SEMANTIC":          bool(flags & SEMANTIC),
            "SQL_FILTER":        bool(flags & SQL_FILTER),
            "CONTAINS_ENTITIES": bool(flags & CONTAINS_ENTITIES),
            "GRAPH_PATHS":       bool(flags & GRAPH_PATHS),
        }

# a = p8.Agent(QuestionDecomposerAgent)
# QuestionDecomposerAgent._unwrap_flags(a("""How many users signed up last week and what is the CEO's background?"""))
# QuestionDecomposerAgent._unwrap_flags(a("""how many elephants where there and how many zebras where there""")) 
# QuestionDecomposerAgent._unwrap_flags( a("""im looking for ABS1234"""))
# QuestionDecomposerAgent._unwrap_flags(a("""im looking for ABS1234 and the general feelings of the employees"""))