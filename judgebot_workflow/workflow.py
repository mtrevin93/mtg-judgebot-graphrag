import os
from typing import List
from query_inspector import GraphRAGQueryInspector

class GraphRAGWorkflow:
    def __init__(self, input_dir: str, lancedb_uri: str):
        """
        Initialize the workflow with GraphRAG
        """
        self.query_inspector = GraphRAGQueryInspector(input_dir, lancedb_uri)

    async def process_query(self, user_question: str) -> dict:
        """
        Process a user query through GraphRAG
        """
        # Send the original question without modifications
        graphrag_result = await self.query_inspector.execute_query(user_question)
        
        return {
            "response": graphrag_result.response,
            "context_data": graphrag_result.context_data
        }

async def main():
    # Use absolute path relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    workflow = GraphRAGWorkflow(
        input_dir=os.path.join(project_root, "output"),
        lancedb_uri=os.path.join(project_root, "output", "lancedb")
    )
    
    question = "What are all of the combat steps?"
    result = await workflow.process_query(question)
    
    print("\n=== GraphRAG Response ===")
    print(result["response"])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())