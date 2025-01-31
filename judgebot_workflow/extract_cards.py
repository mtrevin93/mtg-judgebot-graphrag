from typing import List, Dict, Any
import re
from db.mtg_cards_api import fetch_card_by_name

def extract_and_fetch_cards(database_path: str, text: str) -> List[Dict[str, Any]]:
    """
    Extract card names from text containing [[cardname]] patterns and fetch their details.
    Uses exact name matching for card lookups.
    
    Args:
        database_path: Path to the SQLite database
        text: Text containing card names in [[cardname]] format
        
    Returns:
        List of card objects with their details and rulings
    """
    # Find all card names enclosed in double brackets
    card_pattern = r'\[\[(.*?)\]\]'
    card_names = re.findall(card_pattern, text)
    
    # Remove duplicates while preserving order
    unique_card_names = list(dict.fromkeys(card_names))
    
    # Fetch details for each card using exact matching
    cards = []
    for card_name in unique_card_names:
        # Add exact match parameter to fetch_card_by_name call
        card_results = fetch_card_by_name(database_path, card_name, exact_match=True)
        # Add any found cards to our results
        cards.extend(card_results)
    
    return cards