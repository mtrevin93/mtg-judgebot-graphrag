import os
from dotenv import load_dotenv
from typing import List, Dict
from anthropic import Anthropic
from extract_cards import extract_and_fetch_cards

# Load environment variables from .env file
load_dotenv()

class MTGJudgeWorkflow:
    def __init__(self):
        api_key = os.getenv("ANTROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTROPIC_API_KEY not found in environment variables")
        
        self.client = Anthropic(api_key=api_key)
        # Define paths relative to the current file
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.rules_path = os.path.join(base_dir, "documents", "rules.txt")
        self.database_path = os.path.join(base_dir, "db", "sqlite", "mtg_cards.sqlite")
        self.cached_rules = self._prepare_cached_rules()

    def _prepare_cached_rules(self) -> Dict[str, List[dict]]:
        """Prepare the system messages with cached rules content"""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                rules_content = f.read()
        except Exception as e:
            print(f"Error reading rules file: {e}")
            rules_content = "Error: Could not load comprehensive rules."

        return {
            "system": [
                {
                    "type": "text",
                    "text": "You are an expert Magic: The Gathering judge. You provide accurate rulings based on comprehensive rules and card interactions."
                },
                {
                    "type": "text",
                    "text": rules_content,
                    # "cache_control": {"type": "ephemeral"}
                },
                {
                    "type": "text",
                    "text": "When providing rulings: Always cite relevant rules sections, explain clearly, and address all parts of the question."
                }
            ]
        }

    def _format_card_info(self, cards: List[dict]) -> str:
        """Format card information with double brackets"""
        if not cards:
            return ""
        
        formatted = "Referenced cards:\n"
        for card in cards:
            formatted += f"\n[[{card['name']}]]:\n"
            formatted += f"Type: {card['type_line']}\n"
            if card['oracle_text']:
                formatted += f"Text: {card['oracle_text']}\n"
            if card['rulings']:
                formatted += "Rulings:\n"
                for ruling in card['rulings']:
                    formatted += f"- {ruling['comment']}\n"
        return formatted

    async def process_query(self, question: str) -> dict:
        """Process a user query and get a judge ruling"""
        # Extract card references and get their info
        cards = extract_and_fetch_cards(self.database_path, question)
        
        # Format card information
        card_info = self._format_card_info(cards)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{card_info}\n\n{question}"
                    }
                ]
            }
        ]
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            system=self.cached_rules["system"],
            messages=messages
        )
        
        return {
            "question": question,
            "response": response.content[0].text,
            "usage": response.usage,
            "cards": cards
        }

async def main():
    workflow = MTGJudgeWorkflow()
    
    questions = [
        "if i have [[thrasios, triton hero]] and [[galadriel of Lothlórien]] and i use thrasios' ability and reveal a land ontop what would happen?\n\nmy guess is thrasios resolves and then puts the land in play then galadriel reveals the next top card and I can put that one into play if its a land"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}:")
        print(f"{'='*80}")
        print(question)
        
        result = await workflow.process_query(question)
        
        print(f"\nRESPONSE:")
        print(f"{'='*80}")
        print(result["response"])
        print(f"\nUsage stats:")
        print(f"{'='*80}")
        print(result["usage"])
        print("\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())