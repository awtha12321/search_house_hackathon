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
    app_id="rag_app",
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

        # 2. Push to Database
        storage = QdrantStorage() # TODO: have to change this collection name.
        storage.upsert(ids, vectors, payloads)

        return RAGUpsertResult(ingested_count=len(ids))

    # Single step execution
    result = await ctx.step.run("load-csv-and-embed-database", lambda: _run_ingest(ctx), output_type=RAGUpsertResult)
    return result.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query Real Estate",
    trigger=inngest.TriggerEvent(event="rag/query_ai")
)

async def rag_query_ai(ctx: inngest.Context):
    # 1. Extract inputs from the event
    question = ctx.event.data["question"]

    # STEP 0: Parse user query to get intent and filters (async LLM call)
    async def _parse_query(user_question: str):
        return await parse_user_query(user_question)
    
    parsed_question = await ctx.step.run(
        "parse-user-query",
        lambda: _parse_query(question)
    )
    
    user_intent = parsed_question["search_query"]
    user_filters = parsed_question["filters"]

    # ✅ Clean filters - remove None and empty lists
    cleaned_filters = {
        key: value 
        for key, value in user_filters.items() 
        if value is not None and (not isinstance(value, list) or len(value) > 0)
    }

    # STEP 1: SEARCH
    def _search(search_query: str, filters: dict) -> RAGSearchResult:
        """Search using the query string (not vector directly)"""
        store = QdrantStorage()
        
        # ✅ Pass search_query (string), not query_vector
        found_results = store.search_rentals(
            search_query=search_query,  # ✅ Correct parameter name
            filters=filters,
            top_k=10 # 
        )
        
        return RAGSearchResult(results=found_results)

    # Execute search step
    found = await ctx.step.run(
        "search-db",
        lambda: _search(user_intent, cleaned_filters),
        output_type=RAGSearchResult
    )
    

    # STEP 2: FORMAT CONTEXT
    context_lines = []
    used_ids = []
    top_5_context = []
    count = 0
    
    

    for item in found.results:
        # ✅ Access nested payload
        payload = item.get('payload', {})
        score = item.get('score', 0)
        
        # Extract ID first
        apt_id = None
        if 'id' in payload:
            apt_id = str(payload['id'])
        elif 'id' in item:
            apt_id = str(item['id'])
        
        ## this is important cuz AI will use this to make inference 
        line = (
            f"[Match Score: {score:.3f}] "
            f"**Apartment ID: {apt_id}**\n"
            f"Price: {payload.get('price_huf', 'N/A')} HUF | "
            f"Utilities: {payload.get('avg_util_cost_huf', 'N/A')} HUF | "
            f"Bedrooms: {payload.get('num_bedrooms', 'N/A')} | "
            f"Size: {payload.get('size_m2', 'N/A')} m² | "
            f"Furnished: {payload.get('furnished', 'N/A')} | "
            f"Near Transport: {payload.get('close_to_public_transportation', 'N/A')} | "
            f"Campus: {', '.join(payload.get('close_to_campus', []))} | "
            f"Transport Types: {', '.join(payload.get('transport_types', []))} | "
            f"Description: {payload.get('description', '')}"
        )
        if count < 5: 
            top_5_context.append(line)
            count += 1
        context_lines.append(line)
        
        # ✅ Store the actual rental ID from payload
        if apt_id:
            used_ids.append(apt_id)

    context_block = "\n\n".join(context_lines)
    top_5_context_block = "\n\n".join(top_5_context)
    answer = context_block  # For now, we just return the context block as the answer for testing
    print(answer)
    
    # ctx.logger.info(f"All Apartments Context:\n{context_block}")

    # # Format parsed filters for LLM context || this is crucial for the LLM to understand the user's requirements and explain the matches effectively cuz sometimes the LLM hallucinates and forgets the user's requirements when explaining the matches. So we need to remind it of the user's requirements in a clear and structured way.
    # filter_info = []
    # if user_filters.get("price_huf"):
    #     filter_info.append(f"Max Budget: {user_filters['price_huf']:,} HUF")
    # if user_filters.get("avg_util_cost_huf"):
    #     filter_info.append(f"Max Utilities: {user_filters['avg_util_cost_huf']:,} HUF")
    # if user_filters.get("num_bedrooms"):
    #     filter_info.append(f"Bedrooms: {user_filters['num_bedrooms']}")
    # if user_filters.get("size_m2"):
    #     filter_info.append(f"Min Size: {user_filters['size_m2']} m²")
    # if user_filters.get("furnished") is not None:
    #     filter_info.append(f"Furnished: {user_filters['furnished']}")
    # if user_filters.get("close_to_campus"):
    #     filter_info.append(f"Near: {', '.join(user_filters['close_to_campus'])}")
    # if user_filters.get("transport_types"):
    #     filter_info.append(f"Transport: {', '.join(user_filters['transport_types'])}")
    
    # filter_summary = " | ".join(filter_info) if filter_info else "No specific filters applied"
    
    # user_content = (
    #     f"Student Query: \"{question}\"\n\n"
    #     f"**Student's Requirements:**\n{filter_summary}\n\n"
        
    #     "**All Matching Apartments (ranked by our scoring system):**\n\n"
    #     f"{top_5_context_block}\n\n"
        
    #     "CRITICAL INSTRUCTIONS - READ CAREFULLY:\n"
    #     "1. The apartments above are ALREADY RANKED by our advanced scoring system - NEVER REORDER\n"
    #     "2. ONLY explain the TOP 5 apartments in THE EXACT ORDER shown above. Do not change the order.\n"
    #     "3. The FIRST apartment listed MUST be your 1st recommendation (copy its exact ID)\n"
    #     "4. The SECOND apartment listed MUST be your 2nd recommendation (copy its exact ID)\n\n"
        
    #     "FOR EACH APARTMENT, YOU MUST CHECK:\n"
    #     f"- PRICE: Compare actual price to student's max budget ({user_filters.get('price_huf', 'Not specified')} HUF)\n"
    #     f"- UTILITIES: Compare actual utilities to student's max budget ({user_filters.get('avg_util_cost_huf', 'Not specified')} HUF)\n"
    #     f"- BEDROOMS: Compare actual bedrooms to student's need ({user_filters.get('num_bedrooms', 'Not specified')})\n"
    #     f"- CAMPUS: State which campus(es) the apartment is near (from 'Campus:' field above)\n"
    #     "- FEATURES: Use ONLY features from the Description field - DO NOT INVENT\n\n"
        
    #     "BUDGET COMPARISON FORMAT:\n"
    #     "- If UNDER budget: 'Rent: 200,000 HUF (under your 220,000 budget by 20,000). Utilities: 35,000 HUF (under your 40,000 budget).'\n"
    #     "- If AT budget: 'Rent: 220,000 HUF (at your budget). Utilities: 40,000 HUF (at your budget).'\n"
    #     "- If OVER budget: 'Rent: 240,000 HUF (exceeds your 220,000 budget by 20,000 - trade-off for luxury). Utilities: 45,000 HUF (exceeds budget by 5,000).'\n\n"
        
    #     "BEDROOM COMPARISON FORMAT:\n"
    #     f"- If matches: 'Bedrooms: {user_filters.get('num_bedrooms', 'N/A')} (matches your needs)'\n"
    #     f"- If differs: 'Bedrooms: [actual] (you requested {user_filters.get('num_bedrooms', 'N/A')} - this is a trade-off)'\n\n"
        
    #     "LOCATION FORMAT:\n"
    #     f"- State: 'Location: Near [campus name from Campus field]'\n"
    #     f"- If different from student's preferred ({', '.join(user_filters.get('close_to_campus', [])) if user_filters.get('close_to_campus') else 'Not specified'}), add: '(not your preferred campus - consider commute)'\n\n"
        
    #     "Required format - maintain EXACT ORDER from apartment list above:\n"
    #     "**1st Recommendation: Apartment ID: [EXACT ID FROM FIRST APARTMENT]**\n"
    #     "[Brief description from Description field]. "
    #     "Price: [actual] HUF ([under/at/over] your [budget]). "
    #     "Utilities: [actual] HUF ([under/at/over] your [budget]). "
    #     "Bedrooms: [actual] ([matches/differs from] your needs). "
    #     "Location: Near [campus from data] ([add commute note if different from preferred]).\n\n"
    #     "(Continue format for 2nd, 3rd, 4th, 5th apartments IN ORDER)\n\n"
    # )

    # adapter = ai.openai.Adapter(
    #     auth_key="ollama",
    #     base_url="http://localhost:11434/v1",
    #     model="qwen2.5:7b"
    # )

    # # STEP 3: LLM GENERATION
    # res = await ctx.step.ai.infer(
    #     "llm-answer",
    #     adapter=adapter,
    #     body={
    #         "max_tokens": 1024,
    #         "temperature": 0.3,  # ✅ Slightly higher for more natural responses
    #         "messages": [
    #             {
    #                 "role": "system",
    #                 "content": (
    #                     "You are a factual housing data analyst for University of Debrecen students.\n\n"
    #                     "ABSOLUTE RULES - BREAKING THESE IS AN ERROR:\n"
    #                     "1. Apartments are PRE-RANKED - NEVER change their order. They are already ordered based on the score from database. Do not change the order\n"
    #                     "2. Copy apartment IDs in EXACT ORDER from the list\n"
    #                     "3. ONLY state facts from the data - NO creative descriptions\n"
    #                     "4. Compare ALL budget items: price, utilities, AND bedrooms\n"
    #                     "5. Use the 'Campus:' field to state location - NOT the Description field\n"
    #                     "6. If ANY value exceeds budget, state it as a trade-off\n"
    #                     "7. Do not say 'a bit of a walk' or similar vague terms\n"
    #                     "8. Do not say 'city center' unless it's in the Campus field\n"
    #                     "9. Write concise, data-driven explanations\n"
    #                     "10. Stop after the 5th apartment\n\n"
    #                     "Your task: Analyze the first 5 apartments IN ORDER and state facts about price, utilities, bedrooms, and campus location."
    #                 )
    #             },
    #             {"role": "user", "content": user_content}
    #         ]
    #     }
    # )

    # answer = res["choices"][0]["message"]["content"].strip()
    
    # # Add remaining apartment IDs (from 6th onwards) if available
    # if len(used_ids) > 5:
    #     remaining_ids = used_ids[5:]
    #     answer += "\n\n---\n\n**Other Options:**\n"
    #     answer += f"Additional apartments you might be interested in: {', '.join(remaining_ids)}"
    
    # print("LLM Answer:\n", answer)

    # ✅ Return with correct spelling
    return RAQQueryResult(
        answer=answer,
        used_house_ids=used_ids,
        num_matches=len(found.results),
        all_results=found.results
    ).model_dump()


# FastAPI app
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_csv, rag_query_ai])
