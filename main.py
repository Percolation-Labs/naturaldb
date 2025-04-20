from naturaldb.repo import RocksRepo

"""percolate will not be a dep"""
from percolate.models import Agent
from percolate.utils import make_uuid
import random


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
    repo = RocksRepo(db_path="./test",model_cls=Agent)

    agents = create_sample_agents()
 
    # Client mode (default): adds to in-memory queue and persistent DB queue
    # also processes immediately -> showing how this would fail because we are adding twice in a client mode but the index on the background thread cannot deal with that 
    # below there are examples of how client server works so we can safely add items to a task queue and expect the indexes to be built quickly in the background
    repo.add_records(agents[:10])
    repo.add_records(agents[10:])
    
    repo.index.wait_for_completion()
    
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
    
    print(repo.db.get(b'table_stats:p8:Agent'))
    
def demo_server_mode():
    """
    Demonstrates how to use the server mode for background processing.
    
    In a real application, this would typically be run in a separate process
    or thread that continuously monitors the queue and processes items.
    """
    from percolate.models import Agent
    
    # Create a repo in server mode
    server_repo = RocksRepo(model_cls=Agent, db_path="./test", client_mode=False)
    
    print("Starting server mode indexer...")
    
    # Process the queue continuously
    # This will run indefinitely until interrupted
    try:
        # You can adjust the interval (in seconds) between processing attempts
        server_repo.index.process_queue_continuously(interval=5)
    except KeyboardInterrupt:
        print("Server mode processing stopped")
        
def test_dual_mode():
    """
    Test both client and server modes working together.
    
    This simulates:
    1. Client adding records to the queue
    2. Server processing those records in the background
    """
    import threading
    import time
    from percolate.models import Agent
    
    # Create a client repo
    client_repo = RocksRepo(model_cls=Agent, db_path="./test", client_mode=True)
    
    # Create a server repo (same db_path, different mode)
    server_repo = RocksRepo(model_cls=Agent, db_path="./test", client_mode=False)
    
    # Start the server in a background thread
    def server_thread():
        print("Server thread started")
        # Process once every 2 seconds
        try:
            for _ in range(10):  # Process for 10 iterations then exit
                server_repo.index.run_partial()
                time.sleep(2)
        except Exception as e:
            print(f"Server thread error: {e}")
        print("Server thread finished")
    
    server = threading.Thread(target=server_thread)
    server.daemon = True
    server.start()
    
    # Client adds records in batches with delays
    agents = create_sample_agents()
    
    print("Client adding first batch of records...")
    client_repo.add_records(agents[:5])
    time.sleep(3)  # Give server time to process
    
    print("Client adding second batch of records...")
    client_repo.add_records(agents[5:10])
    time.sleep(3)
    
    print("Client adding third batch of records...")
    client_repo.add_records(agents[10:])
    
    # Wait for server to finish
    server.join()
    
    print("All processing completed")
    
    # Verify results
    pred_results = client_repo._search_predicate_index_single("Find agents for customer support", sample_size=3)
    if not isinstance(pred_results, str):
        print(f"Found {len(pred_results)} results from predicate search")
    
def test_background_thread_mode():
    """
    Test the new background thread mode for indexing.
    
    This mode uses a single RocksRepo instance with a background thread
    for processing the queue, allowing shared access to the database.
    """
    import time
    from percolate.models import Agent
    
    # Create a repo with background thread enabled and custom intervals
    repo = RocksRepo(
        model_cls=Agent, 
        db_path="./test", 
        use_background_thread=True
    )
    
    # Give the background thread a moment to start
    time.sleep(5)
    
    print("Main thread: Creating sample agents...")
    agents = create_sample_agents()
    
    print("Main thread: Adding first batch of records...")
    repo.add_records(agents[:10])
    
    # Wait a bit to let the background thread process
    time.sleep(3)
    
    print("Main thread: Adding second batch of records...")
    repo.add_records(agents[10:])
    
    # Wait for processing to complete
    time.sleep(15)  # Wait longer to ensure full indexing cycle runs
    
    # Test searching
    print("\nTesting predicate search:")
    pred_results = repo._search_predicate_index_single("Find agents for customer support", sample_size=3)
    if not isinstance(pred_results, str):
        print(f"Found {len(pred_results)} results from predicate search")
        if len(pred_results) > 0:
            for i, row in enumerate(pred_results.iter_rows(named=True)):
                print(f"Result {i+1}: {row['name']} - {row['category']}")
    else:
        print(f"Predicate search error: {pred_results}")
    
    print('Waiting for final background processing to complete...')
    time.sleep(30)  # Reduced waiting time
    
    # Clean up by stopping the background thread
    print("Stopping background indexer thread...")
    repo.index._stop_background_thread()
    
if __name__ == "__main__":
    # Regular client mode demo
    # main()
    
    # Uncomment to test server mode
    # Note: This will run indefinitely until interrupted
    # demo_server_mode()
    
    # Uncomment to test both modes working together
    # test_dual_mode()
    
    # Test the new background thread mode
    test_background_thread_mode()
