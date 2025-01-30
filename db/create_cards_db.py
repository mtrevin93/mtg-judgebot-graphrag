import os
import json
from mtg_cards_api import setup_card_database, insert_card_into_db, insert_ruling_into_db

def create_cards_db(cards_file_path: str, rulings_file_path: str, db_path: str):
    """
    Create and populate the MTG cards database from JSON files.

    Args:
        cards_file_path (str): Path to the JSON file containing card data
        rulings_file_path (str): Path to the JSON file containing rulings data
        db_path (str): Path where the SQLite database should be created
    """
    # Delete the existing database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database: {db_path}")

    # Create new database and tables
    setup_card_database(db_path)
    print("Created new database with tables")

    # Load and insert cards
    with open(cards_file_path, 'r', encoding='utf-8') as f:
        cards = json.load(f)
        total_cards = len(cards)

        print(f"Loading {total_cards} cards...")
        for i, card in enumerate(cards, 1):
            insert_card_into_db(db_path, card)
            if i % 100 == 0:  # Progress update every 100 cards
                print(f"Processed {i}/{total_cards} cards")

    print("Finished loading cards")

    # Load and insert rulings
    with open(rulings_file_path, 'r', encoding='utf-8') as f:
        rulings = json.load(f)
        total_rulings = len(rulings)

        print(f"Loading {total_rulings} rulings...")
        for i, ruling in enumerate(rulings, 1):
            # Add a default value for source if it's missing
            if 'source' not in ruling:
                ruling['source'] = None
            insert_ruling_into_db(db_path, ruling)
            if i % 100 == 0:  # Progress update every 100 rulings
                print(f"Processed {i}/{total_rulings} rulings")

    print("Finished loading rulings")
    print("Database creation completed successfully")

if __name__ == "__main__":
    cards_file_path = '../documents/rulings-20241130100030.json'
    rulings_file_path = '../documents/oracle-cards-20241130100203.json'
    db_path = './mtg_cards.sqlite'

    # Ensure the db directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    create_cards_db(cards_file_path, rulings_file_path, db_path)
