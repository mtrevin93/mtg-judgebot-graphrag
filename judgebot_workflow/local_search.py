from dotenv import load_dotenv
import os
import pandas as pd
import tiktoken
from typing import Optional

from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType

_ = load_dotenv()

class MTGSearchEngine:
    def __init__(self, input_dir: str, lancedb_uri: str):
        self.input_dir = input_dir
        self.lancedb_uri = lancedb_uri
        self.setup_models()
        self.load_data()
        self.setup_search_engine()

    def setup_models(self):
        """Initialize language and embedding models"""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Set OPENAI_API_KEY in environment")
        
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-4",
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )

        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        self.text_embedder = OpenAIEmbedding(
            api_key=self.api_key,
            api_type=OpenaiApiType.OpenAI,
            model="text-embedding-3-small",
            deployment_name="text-embedding-3-small",
            max_retries=20,
        )

    def load_data(self):
        """Load all necessary data from parquet files"""
        # Load entities
        entity_df = pd.read_parquet(f"{self.input_dir}/create_final_nodes.parquet")
        entity_embedding_df = pd.read_parquet(f"{self.input_dir}/create_final_entities.parquet")
        self.entities = read_indexer_entities(entity_df, entity_embedding_df, community_level=2)

        # Setup vector store
        self.description_embedding_store = LanceDBVectorStore(
            collection_name="default-entity-description",
        )
        self.description_embedding_store.connect(db_uri=self.lancedb_uri)

        # Load relationships
        relationship_df = pd.read_parquet(f"{self.input_dir}/create_final_relationships.parquet")
        self.relationships = read_indexer_relationships(relationship_df)

        # Load reports
        report_df = pd.read_parquet(f"{self.input_dir}/create_final_community_reports.parquet")
        self.reports = read_indexer_reports(report_df, entity_df, community_level=2)

        # Load text units
        text_unit_df = pd.read_parquet(f"{self.input_dir}/create_final_text_units.parquet")
        self.text_units = read_indexer_text_units(text_unit_df)

    def setup_search_engine(self):
        """Configure the search context and engine"""
        context_builder = LocalSearchMixedContext(
            community_reports=self.reports,
            text_units=self.text_units,
            entities=self.entities,
            relationships=self.relationships,
            covariates=None,  # No claims needed for MTG
            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )

        SYSTEM_INSTRUCTIONS = """Please always answer by citing exact rules numbers. For example, rule 101.2a states 'text of rule 101.2a'.
Never give commentary about how a rule is important to gameplay, your job is only to answer the question as factually as possible.
Never mention generics like tabletop gaming.
Provide your understanding of the user's question, then the rules and citations, then a summary that is your understanding of the application of the rules to the problem."""

        self.search_engine = LocalSearch(
            llm=self.llm,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            llm_params={
                "max_tokens": 2_000,
                "temperature": 0.0,
            },
            context_builder_params={
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 10,
                "top_k_relationships": 10,
                "include_entity_rank": True,
                "include_relationship_weight": True,
                "include_community_rank": False,
                "return_candidate_context": False,
                "embedding_vectorstore_key": EntityVectorStoreKey.ID,
                "max_tokens": 6000,
            },
            response_type="multiple paragraphs",
            system_prompt=SYSTEM_INSTRUCTIONS,
        )

    async def search(self, query: str) -> dict:
        """Execute a search query"""
        result = await self.search_engine.asearch(query)
        return {
            "response": result.response,
            "context_data": result.context_data,
            "context_text": result.context_text,
        }

async def main():
    # Use absolute path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    search_engine = MTGSearchEngine(
        input_dir=os.path.join(project_root, "output"),
        lancedb_uri=os.path.join(project_root, "output", "lancedb")
    )
    
    question = "How does [[Lightning Bolt]] interact with [[Spellskite]]?"
    result = await search_engine.search(question)
    
    print("\n=== Response ===")
    print(result["response"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
