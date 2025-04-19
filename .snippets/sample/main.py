

import uuid
import random
from RocksPydanticRepository import RocksPydanticRepository
from percolate.models import Agent
from percolate.utils import make_uuid
import polars as pl
    
    
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
        "what do you know about celtic and emerald isle",
        
        
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
    repo = RocksPydanticRepository()

    agents = create_sample_agents()

    repo.add_many(agents)

    repo.build_predicate_indexes()

    repo.build_semantic_indexes(dim=1536)

    print("✅ Sample agents added and indexed")
    
    print(f"sample agent {pl.DataFrame([repo.get_by_id(str(make_uuid(f'Agent-{1}')), Agent)])}")

    print('*******************')
    res = repo.semantic_search('on ireland', Agent)

    res = pl.DataFrame([d.model_dump() for d in res])
    print(f"semantic search agent: 'on ireland'- {res}")
    
    # print('*******************')
    # res = repo.sql_predicate_search("Show me Agent-2 and Agent-9", Agent)    
    # print(f"predicate result  for searching by two names {res}")
    
    print('*******************')
    res = repo.sql_predicate_search("Show me support items", Agent)    
    print(f"predicate result for searching by by `support` {res}")
    
if __name__ == "__main__":
    main()
