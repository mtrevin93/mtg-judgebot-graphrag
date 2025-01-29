import os
import argparse
import pandas as pd
import tiktoken
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_covariates
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore


class GraphRAGQueryInspector:
    def __init__(self, input_dir: str, lancedb_uri: str):
        self.input_dir = input_dir
        self.lancedb_uri = lancedb_uri
        self.setup_models()
        self.load_data()
        self.setup_search_engine()

    def setup_models(self):
        """Initialize language and embedding models"""
        # Debug prints
        print("Environment variables:")
        print(f"GRAPHRAG_API_KEY exists: {'GRAPHRAG_API_KEY' in os.environ}")
        print(f"OPENAI_API_KEY exists: {'OPENAI_API_KEY' in os.environ}")
        
        # Use OPENAI_API_KEY if GRAPHRAG_API_KEY is not set
        self.api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key found. Set either GRAPHRAG_API_KEY or OPENAI_API_KEY in environment")
        
        # Set default models if not specified
        self.llm_model = os.environ.get("GRAPHRAG_LLM_MODEL", "gpt-4o")
        self.embedding_model = os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "text-embedding-3-small")

        print(f"Using API key: {self.api_key[:6]}...")  # Only print first 6 chars for security
        print(f"Using LLM model: {self.llm_model}")
        print(f"Using embedding model: {self.embedding_model}")

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.llm_model,
            api_type=OpenaiApiType.OpenAI,
            max_retries=20,
        )

        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        self.text_embedder = OpenAIEmbedding(
            api_key=self.api_key,
            api_type=OpenaiApiType.OpenAI,
            model=self.embedding_model,
            deployment_name=self.embedding_model,
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
            relationships=self.relationships,            entity_text_embeddings=self.description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=self.text_embedder,
            token_encoder=self.token_encoder,
        )

        local_context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": True,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            "max_tokens": 12_000,
        }

        llm_params = {
            "max_tokens": 2_000,
            "temperature": 0.0,
        }

        self.search_engine = LocalSearch(
            llm=self.llm,
            context_builder=context_builder,
            token_encoder=self.token_encoder,
            llm_params=llm_params,
            context_builder_params=local_context_params,
            response_type="multiple paragraphs",
        )

    async def execute_query(self, query: str):
        """Execute a query and display all results"""
        import pdb; pdb.set_trace()  # Debugger will pause here
        print(f"\n=== Executing Query: {query} ===\n")
        
        result = await self.search_engine.asearch(query)
        
        print("=== Response ===")
        print(result.response)
        print("\n=== Context Data ===")
        
        # Display entities
        if "entities" in result.context_data:
            print("\nEntities:")
            print(result.context_data["entities"].head())
            
        # Display relationships
        if "relationships" in result.context_data:
            print("\nRelationships:")
            print(result.context_data["relationships"].head())
            
        # Display reports
        if "reports" in result.context_data:
            print("\nReports:")
            print(result.context_data["reports"].head())
            
        # Display sources
        if "sources" in result.context_data:
            print("\nSources:")
            print(result.context_data["sources"].head())
            
        # Display claims
        if "claims" in result.context_data:
            print("\nClaims:")
            print(result.context_data["claims"].head())

def main():
    parser = argparse.ArgumentParser(description='Execute and inspect GraphRAG queries')
    parser.add_argument('--query', type=str, required=True, help='The query to execute')
    parser.add_argument('--input-dir', type=str, default="./output",
                      help='Directory containing input parquet files')
    parser.add_argument('--lancedb-uri', type=str, help='URI for LanceDB')
    
    args = parser.parse_args()
    
    # If lancedb_uri not provided, use input_dir/lancedb
    lancedb_uri = args.lancedb_uri or f"{args.input_dir}/lancedb"
    
    inspector = GraphRAGQueryInspector(args.input_dir, lancedb_uri)
    
    import asyncio
    asyncio.run(inspector.execute_query(args.query))

if __name__ == "__main__":
    main()