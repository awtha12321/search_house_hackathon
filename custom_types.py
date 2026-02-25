import pydantic
from typing import List, Dict, Any, Optional

# --- INGESTION TYPES ---

class RAGIngestData(pydantic.BaseModel):
    # We use 'Any' for the values because a CSV row might have
    # ints (price), strings (address), or lists (transportation)
    rows: List[Dict[str, Any]]
    source_filename: str

class RAGUpsertResult(pydantic.BaseModel):
    ingested_count: int


# --- SEARCH TYPES ---

class RAGSearchResult(pydantic.BaseModel):
    # Instead of just "contexts" (text), we return the full objects
    # so the AI knows the price and location of the house.
    results: List[Dict[str, Any]]


class RAQQueryResult(pydantic.BaseModel):
    answer: str
    # We might want to list the IDs of the houses used for the answer
    used_house_ids: List[str]
    num_matches: int
    all_results: List[Dict[str, Any]]