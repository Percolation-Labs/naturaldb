from naturaldb.repo import RocksRepo

"""percolate will not be a dep"""
from percolate.models import Agent
from percolate.utils import make_uuid
import random

repo = RocksRepo(db_path="./test",model_cls=Agent)

def get_meaningful_sentences():
    return [
        "Helps users troubleshoot complex technical issues in real-time.",
        "Generates Python code snippets from plain English descriptions.",
        "Provides creative writing assistance for short stories and poetry.",
        "Translates between more than 30 languages using advanced models.",
        "Acts as a virtual therapist for journaling and cognitive reflection.",
        "Summarizes long documents into key bullet points.",
        "Searches legal documents for relevant case law and references.",
        "Classifies incoming support tickets by urgency and topic.",
        "Provides math step-by-step explanations for algebra and calculus.",
        "Helps users plan meals and recipes based on dietary preferences.",
        "Guides software engineers through debugging strategies.",
        "Analyzes customer reviews and extracts sentiment insights.",
        "Simulates conversational AI for training customer support agents.",
        "Provides voice-to-text conversion with punctuation and cleanup.",
        "Writes email drafts based on subject lines and context.",
        "Fetches and summarizes the latest scientific research articles.",
        "Generates SQL queries based on natural language prompts.",
        "Advises on marketing strategies using current trend analysis.",
        "Interprets medical symptoms using a diagnostic decision tree.",
        "Answers trivia and general knowledge questions with citations.",
         "Capital of ireland is dublin",
        "Saint patricks day in new york",
        "Cork is a nice city",
        "what do you know about celtic and emerald isle"
    ]

def create_sample_agents():
    sentences = get_meaningful_sentences()
    categories = ["chat", "code", "support", "ops", "vision"]
    agents = []

    for i, sentence in enumerate(sentences):
        agent = Agent(
            id=str(make_uuid(f'Agent-{i}')),
            name=f"Agent-{i}",
            category=random.choice(categories),
            description=sentence,
            spec={"input": {"type": "text"}, "output": {"type": "text"}},
            functions={
                "run": {
                    "params": ["input"],
                    "description": f"Run the agent on user input"
                }
            }
        )
        agents.append(agent)
    
    return agents

def main():
    agents = create_sample_agents()
 
    # e = repo._search_semantic_index_single("Something about ireland")
    # print(e) 
    # e = repo.get_entities('Agent-1')
    # print(e)
    
    # Adding records in batches to demonstrate the current implementation
    # Note: Currently the background indexer doesn't fully support requeuing items
    # during the lifetime of the indexer - when repo.add_records() is called, the indexer
    # adds to its internal queue (self.records list) but sets a self.futures flag that prevents
    # new runs until completion. To support continuous queueing:
    # 1. Modify run() and run_partial() to work with concurrent additions to self.records list
    # 2. Add a flag to track if the background worker is already processing (not just queued)
    # 3. Clear self.futures only when truly empty, not at the end of each batch
    # 4. Implement a proper producer/consumer pattern with thread-safe queues
    # 5. Consider adding a background thread that periodically processes any new records
    repo.add_records(agents[:10])
    repo.add_records(agents[10:])
    
    repo.index.wait_for_completion()
 
    # Test the hybrid semantic search
    # print("\nTesting hybrid_semantic_search:")
    # results = repo.hybrid_semantic_search("Help with troubleshooting and tech support", top_k=3)
    # print(f"Found {len(results)} results")
    # for result in results:
    #     print(f"ID: {result['id']}, Similarity: {result['_similarity']:.4f}")
    #     print(f"Description: {result['description']}")
    #     print()
    
    # Test the keys-only version
    # print("\nTesting hybrid_semantic_search_keys_only:")
    # key_results = repo.hybrid_semantic_search_keys_only("Help with troubleshooting and tech support", top_k=3)
    # print(f"Found {len(key_results)} key results")
    # for obj_id, score in key_results:
    #     print(f"ID: {obj_id}, Similarity: {score:.4f}")
    
    # Also test the partial vector indexing
    # print("\nTesting partial vector indexing:")
    # partial_vectors = repo.index.get_partial_vectors(namespace="p8", entity="Agent")
    # print(f"Found {len(partial_vectors)} partial vectors")
    
    # Test predicate search with the wildcard approach
    print("\nTesting predicate search with wildcards:")
    pred_results = repo._search_predicate_index_single("Find agents for customer support", sample_size=3)
    if not isinstance(pred_results, str):  # Check if it's not an error message
        print(f"Found {len(pred_results)} results from predicate search")
        if len(pred_results) > 0:
            for i, row in enumerate(pred_results.iter_rows(named=True)):
                print(f"Result {i+1}: {row['name']} - {row['category']}")
    else:
        print(f"Predicate search error: {pred_results}")
    
    #check all keys in db
    #keys = list(repo.iter_keys())
    # for k in keys:
    #     print(k)
    
    #check seek only provides the match from prefix
    #print(list([ k for k,v in repo.seek('vindex:p8:Agent')]))
    
    print(repo.db.get(b'table_stats:p8:Agent'))

if __name__ == "__main__":
    main()
