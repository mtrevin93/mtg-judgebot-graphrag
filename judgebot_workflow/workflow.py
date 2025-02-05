import os
import sys
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from judgebot_workflow.local_search import MTGSearchEngine
from judgebot_workflow.extract_cards import extract_and_fetch_cards

JUDGE_SYSTEM_PROMPT = """You are a certified MTG judge from the MTG judge program. Using only the knowledge provided and WITHOUT using your general knowledge of Magic: The Gathering, you will answer the user's query.

You will provide:
1. A clear answer based only on the provided rules and sources
2. A list of the specific rule numbers used
3. A BRIEF explanation of your ruling

Information is in the following order of importance:
1. Card text and the formal rulings, if any.
2. The rules and sources.

Format your response as:
RULING: (your answer)
RULES USED: (list of rule numbers)
EXPLANATION: (brief explanation)"""

class GraphRAGWorkflow:
    def __init__(self, input_dir: str, database_path: str):
        """Initialize the workflow with GraphRAG and LangChain"""
        self.search_engine = MTGSearchEngine(
            input_dir=os.path.join(project_root, "output"),
            lancedb_uri=os.path.join(project_root, "output", "lancedb")
        )
        self.database_path = database_path
        
        # Initialize LangChain components
        self.judge_llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0
        )
        
        self.judge_prompt = ChatPromptTemplate.from_messages([
            ("system", JUDGE_SYSTEM_PROMPT),
            ("user", """Question: {question}

Card Information:
{card_context}

Relevant Rules and Sources:
{sources}""")
        ])

    async def process_query(self, user_question: str) -> dict:
        """Process a user query through GraphRAG and get a judge ruling"""
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

        # Get relevant rules and sources from GraphRAG
        graphrag_result = await self.search_engine.search(user_question)
        
        # Format sources for the judge
        sources_text = ""
        if "sources" in graphrag_result["context_data"]:
            for idx, source in graphrag_result["context_data"]["sources"].iterrows():
                sources_text += f"\nSource {idx + 1}:\n"
                if "title" in source:
                    sources_text += f"Title: {source['title']}\n"
                if "text" in source:
                    sources_text += f"Text: {source['text']}\n"
                sources_text += "-" * 40 + "\n"
        
        # Get the judge's ruling
        judge_chain = self.judge_prompt | self.judge_llm
        judge_response = judge_chain.invoke({
            "question": user_question,
            "card_context": card_context,
            "sources": sources_text
        })
        
        return {
            "graphrag_response": graphrag_result["response"],
            "judge_response": judge_response.content,
            "context_data": graphrag_result.get("context_data", {}),
            "context_text": graphrag_result.get("context_text", ""),
            "referenced_cards": cards
        }

async def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    workflow = GraphRAGWorkflow(
        input_dir=os.path.join(project_root, "output"),
        database_path=os.path.join(project_root, "db", "sqlite", "mtg_cards.sqlite")
    )
    
    question = "[[Austere command]] resolves differently than [[prismari command]] So [[austere command]] destroys any artifacts/enchantments before it would any creatures so those cards wouldnt see your creatures dying. [[Prismari command]] doesnt follow the same rule? Say it deals 2 damage to [[orcish bowmasters]] and draw two discard two. Why does the [[orcish bowmasters]] see the draw two?"
    result = await workflow.process_query(question)
    
    print("\n" + "="*80)
    print("QUESTION:")
    print("="*80)
    print(question)
    
    print("\n" + "="*80)
    print("GRAPHRAG RESPONSE:")
    print("="*80)
    print(result["graphrag_response"])
    
    print("\n" + "="*80)
    print("JUDGE RESPONSE:")
    print("="*80)
    print(result["judge_response"])
    
    print("\n" + "="*80)
    print("SOURCES USED:")
    print("="*80)
    if "sources" in result["context_data"]:
        for idx, source in result["context_data"]["sources"].iterrows():
            print(f"\nSource {idx + 1}:")
            print(f"Title: {source.get('title', 'N/A')}")
            print(f"Text: {source.get('text', 'N/A')}")
            print("-" * 40)
    else:
        print("No sources found in context data")
    
    print("\n" + "="*80)
    print("REFERENCED CARDS:")
    print("="*80)
    if result["referenced_cards"]:
        for card in result["referenced_cards"]:
            print(f"\n{card['name']}:")
            print(f"Type: {card['type_line']}")
            if card['oracle_text']:
                print(f"Text: {card['oracle_text']}")
            if card['rulings']:
                print("\nRulings:")
                for ruling in card['rulings']:
                    print(f"- {ruling['comment']}")
    else:
        print("No cards referenced")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())