import os
import sys
from typing import List, Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from query_inspector import GraphRAGQueryInspector
from judgebot_workflow.extract_cards import extract_and_fetch_cards

class GraphRAGWorkflow:
    def __init__(self, input_dir: str, lancedb_uri: str, database_path: str):
        """
        Initialize the workflow with GraphRAG
        
        Args:
            input_dir: Directory containing input files for GraphRAG
            lancedb_uri: URI for LanceDB
            database_path: Path to the SQLite database containing card data
        """
        self.query_inspector = GraphRAGQueryInspector(input_dir, lancedb_uri)
        self.database_path = database_path

    async def process_query(self, user_question: str) -> dict:
        """
        Process a user query through GraphRAG. First extracts any card references,
        then includes the card data in the context for the LLM.
        """
        # Extract any card references from the question
        cards = extract_and_fetch_cards(self.database_path, user_question)
        
        # Format card information for context
        card_context = ""
        if cards:
            card_context = "Referenced cards:\n"
            for card in cards:
                card_context += f"\n{card['name']}:\n"
                card_context += f"Type: {card['type_line']}\n"
                if card['oracle_text']:
                    card_context += f"Text: {card['oracle_text']}\n"
                if card['rulings']:
                    card_context += "Rulings:\n"
                    for ruling in card['rulings']:
                        card_context += f"- {ruling['comment']}\n"
                card_context += "\n"

        # Combine the original question with card context
        enhanced_query = f"{user_question}\n\nContext:\n{card_context}" if cards else user_question
        
        # Process through GraphRAG
        graphrag_result = await self.query_inspector.execute_query(enhanced_query)
        
        return {
            "response": graphrag_result.response,
            "context_data": graphrag_result.context_data,
            "referenced_cards": cards
        }

async def main():
    # Use absolute path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    workflow = GraphRAGWorkflow(
        input_dir=os.path.join(project_root, "output"),
        lancedb_uri=os.path.join(project_root, "output", "lancedb"),
        database_path=os.path.join(project_root, "db", "sqlite", "mtg_cards.sqlite")
    )
    
    question = "How does [[Lightning Bolt]] interact with [[Spellskite]]?"
    result = await workflow.process_query(question)
    
    print("\n=== GraphRAG Response ===")
    print(result["response"])
    
    if result["referenced_cards"]:
        print("\n=== Referenced Cards ===")
        for card in result["referenced_cards"]:
            print(f"\n{card['name']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())