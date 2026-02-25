import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from query_parser import parse_user_query

# Import types and DB
from custom_types import RAGUpsertResult, RAGSearchResult, RAQQueryResult
from vector_db import QdrantStorage
# Import our new powerful loader
from data_loader import load_and_embed_csv


inngest_client = inngest.Inngest(
    app_id="test_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


# NOTE: The helper `embed_text` is gone! We don't need it here anymore.

@inngest_client.create_function(
    fn_id="RAG: Ingest CSV",
    trigger=inngest.TriggerEvent(event="rag/ingest_csv"),
)
async def rag_ingest_csv(ctx: inngest.Context):
    # We combine Load + Embed into one step because data_loader does both now.
    def _run_ingest(ctx: inngest.Context) -> RAGUpsertResult:
        csv_path = ctx.event.data["csv_path"]

        # 1. The Heavy Lifting (Load + Embed) happens here
        ids, vectors, payloads = load_and_embed_csv(csv_path)
        print(f"✅ Loaded {len(ids)} records with embeddings")
        print(f"Sample payload: {payloads[0] if payloads else 'None'}")

        # 2. Push to Database
        storage = QdrantStorage()  # Use default collection like main.py
        storage.upsert(ids, vectors, payloads)

        return RAGUpsertResult(ingested_count=len(ids))

    # Single step execution
    result = await ctx.step.run("load-csv-and-embed-database", lambda: _run_ingest(ctx), output_type=RAGUpsertResult)
    return result.model_dump()

app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_csv])