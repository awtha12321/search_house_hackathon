import ollama
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio
import time
import json


# 1. Define the structure you WANT (The "Schema")
class SearchFilters(BaseModel):
    price_huf: Optional[int] = Field(None, description="Maximum budget in HUF")
    close_to_public_transportation: Optional[bool] = Field(None, description="Must be near public transport")
    close_to_campus: Optional[List[str]] = Field(
        default_factory=list,
        description="Must be one of: ['Kassai Campus', 'Medical School', 'Main Campus', 'Faculty of Engineering', 'Böszörményi Campus']"
    )
    flexible_location: Optional[bool] = Field(False, description="Whether location is flexible (not a hard requirement)")
    size_m2: Optional[int] = Field(None, description="Minimum size in square meters")
    num_bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    avg_util_cost_huf: Optional[int] = Field(None, description="Maximum utility cost in HUF")
    furnished: Optional[bool] = Field(None, description="Must be furnished")
    transport_types: Optional[List[str]] = Field(
        default_factory=list,
        description="Specific transport types: ['Tram', 'Bus']"
    )


class UserIntent(BaseModel):
    search_query: str = Field(
        ...,
        description="Keywords describing the vibe, style, or atmosphere. (e.g. 'quiet', 'modern', 'bright', 'renovated')"
    )
    filters: SearchFilters


SYSTEM_PROMPTS = {
    "prices": """You are a structured data extraction assistant for a University of Debrecen housing platform.

Your ONLY job is to extract JSON data from user messages. Follow these rules EXACTLY.

■ price_huf (integer or null)
  - Convert: "150k" → 150000, "1.5 million" → 1500000, "220 000" → 220000, "125,000" → 125000
  - Extract the MAXIMUM budget mentioned
  - If no price mentioned → null
  
■ avg_util_cost_huf (integer or null)
  - Extract utility/bills cost: "utilities max 20k" → 20000
  - If not mentioned → null

EXAMPLES:

Input: "Hi! I'm a 4th year General Medicine student, so I need to be super close to the Clinics or Auguszta. I'm looking for a modern, renovated studio. My absolute max budget is 180,000 HUF for rent, and I'm hoping utilities stay under 25k. It must be fully furnished since I'm international. I don't drive, so being near a Tram stop is essential."
Output:
{
  "price_huf": 180000,
  "avg_util_cost_huf": 25000
}

Input: "Hello, we are two Civil Engineering students looking for a flat near Ótemető street. We need a 2-bedroom apartment, ideally around 60 m2. We are on a tight budget, so 140k HUF is our limit for rent, and maybe 30,000 for bills. We have our own beds, so an unfurnished place is fine. We take the bus to campus every day."
Output:
{
  "price_huf": 140000,
  "avg_util_cost_huf": 30000
}

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "price_huf": 220000,
  "avg_util_cost_huf": 40000
}
""",
    
    "basics": """You are a structured data extraction assistant for a University of Debrecen housing platform.

Your ONLY job is to extract JSON data from user messages. Follow these rules EXACTLY.

■ furnished (boolean or null)
  - "furnished", "fully furnished", "equipped" → true
  - "unfurnished", "empty", "I have my own furniture" → false
  - If not mentioned → null
  
■ size_m2 (integer or null)
  - Extract square meters: "45 m2", "45sqm", "45 square meters" → 45
  - If not mentioned → null
  
■ num_bedrooms (integer or null)
  - "studio" or "1 bedroom" → 1
  - "2 bedroom" → 2, "3 bedroom" → 3
  - If not mentioned → null
  
EXAMPLES:

Input: "Hi! I'm a 4th year General Medicine student, so I need to be super close to the Clinics or Auguszta. I'm looking for a modern, renovated studio. My absolute max budget is 180,000 HUF for rent, and I'm hoping utilities stay under 25k. It must be fully furnished since I'm international. I don't drive, so being near a Tram stop is essential."
Output:
{
  "furnished": true,
  "size_m2": null,
  "num_bedrooms": 1
}

Input: "Hello, we are two Civil Engineering students looking for a flat near Ótemető street. We need a 2-bedroom apartment, ideally around 60 m2. We are on a tight budget, so 140k HUF is our limit for rent, and maybe 30,000 for bills. We have our own beds, so an unfurnished place is fine. We take the bus to campus every day."
Output:
{
  "furnished": false,
  "size_m2": 60,
  "num_bedrooms": 2
}

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "furnished": null,
  "size_m2": null,
  "num_bedrooms": 3
}
""",
    
    "transport": """You are a structured data extraction assistant for a University of Debrecen housing platform.

Your ONLY job is to extract JSON data from user messages. Follow these rules EXACTLY.

■ close_to_public_transportation (boolean or null)
  - Set TRUE only if user mentions: "public transport", "tram", "bus", "transit"
  - Set null if not mentioned
  - Do NOT set true just because user wants to be "close" to something
  
■ transport_types (array)
  - ONLY ALLOWED VALUES: "Tram", "Bus"
  - If user says "tram", "tram line", "tram line 1" → ["Tram"]
  - If user says "bus", "bus line" → ["Bus"]
  - If user says both → ["Tram", "Bus"]
  - FORBIDDEN VALUES: "walking distance", "parking", "metro", "car"
  - If user only says "close to public transport" generally → ["Tram", "Bus"]
  - If not mentioned → []
  
EXAMPLES:

Input: "Hi! I'm a 4th year General Medicine student, so I need to be super close to the Clinics or Auguszta. I'm looking for a modern, renovated studio. My absolute max budget is 180,000 HUF for rent, and I'm hoping utilities stay under 25k. It must be fully furnished since I'm international. I don't drive, so being near a Tram stop is essential."
Output:
{
  "close_to_public_transportation": true,
  "transport_types": ["Tram"]
}

Input: "Hello, we are two Civil Engineering students looking for a flat near Ótemető street. We need a 2-bedroom apartment, ideally around 60 m2. We are on a tight budget, so 140k HUF is our limit for rent, and maybe 30,000 for bills. We have our own beds, so an unfurnished place is fine. We take the bus to campus every day."
Output:
{
  "close_to_public_transportation": true,
  "transport_types": ["Bus"]
}

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "close_to_public_transportation": null,
  "transport_types": []
}
""",
    
    "campus": """You are a structured data extraction assistant for a University of Debrecen housing platform.

Your ONLY job is to extract JSON data from user messages. Follow these rules EXACTLY.

YOU MUST ALWAYS CHECK THIS SECTION BEFORE RESPONDING 

■ close_to_campus (array)
  - ALWAYS extract the campus based on major/location keywords mentioned
  - This represents their PREFERRED campus location
  
■ flexible_location (boolean)
  - Set to TRUE if user uses ANY of these phrases:
    * "it's fine if not near [campus]"
    * "doesn't have to be near [campus]"
    * "location doesn't matter"
    * "location is flexible"
    * "as long as I get [requirement], location doesn't matter"
    * "I'm okay with anywhere"
    * "campus location is not important"
  - Set to FALSE if user emphasizes location importance:
    * "must be near"
    * "need to be close to"
    * "super close to"
    * No flexibility phrases mentioned

RULE: Scan the user's message for ANY of these keywords, then set close_to_campus accordingly.

──────────────────────────────────────────────────────────────────
MEDICAL SCHOOL KEYWORDS → close_to_campus: ["Medical School"]
──────────────────────────────────────────────────────────────────
- Medicine, General Medicine, Medical School
- Dentistry, Dental, Dental School
- Pharmacy, Pharmacist
- Nursing, Nurse
- Clinics, Hospital, Auguszta, Nagyerdei körút

EXAMPLE: "I study General Medicine" → ["Medical School"]
EXAMPLE: "close to the Clinics" → ["Medical School"]
EXAMPLE: "Dentistry student near Auguszta" → ["Medical School"]

──────────────────────────────────────────────────────────────────
KASSAI CAMPUS KEYWORDS → close_to_campus: ["Kassai Campus"]
──────────────────────────────────────────────────────────────────
- Computer Science, CS, IT, Informatics
- Business Informatics, Data Science
- Law, Legal Studies
- Kassai, Kassai road, Kassai út
- Főnix, Fonix, Főnix Arena
- Laktanya

──────────────────────────────────────────────────────────────────
FACULTY OF ENGINEERING → close_to_campus: ["Faculty of Engineering"]
──────────────────────────────────────────────────────────────────
- Civil Engineering, Mechanical Engineering
- Electrical Engineering, Mechatronics
- Architecture, Technical studies
- Ótemető, Otemeto, Ótemető street

──────────────────────────────────────────────────────────────────
BÖSZÖRMÉNYI CAMPUS → close_to_campus: ["Böszörményi Campus"]
──────────────────────────────────────────────────────────────────
- Business Administration, Economics
- International Business, Business School
- Agriculture, Food Science, Agrarian
- Böszörményi, Boszormenyi

──────────────────────────────────────────────────────────────────
MAIN CAMPUS → close_to_campus: ["Main Campus"]
──────────────────────────────────────────────────────────────────
- Biology, Chemistry, Physics
- Humanities, Psychology, History
- Life Sciences, Natural Sciences
- Music, Conservatory, Faculty of Music
- Main Building, Main Campus
- Egyetem tér, Egyetem ter, University Square
- Botanical Garden

──────────────────────────────────────────────────────────────────
EXAMPLES
──────────────────────────────────────────────────────────────────

Input: "Hi! I'm a 4th year General Medicine student, so I need to be super close to the Clinics or Auguszta. I'm looking for a modern, renovated studio. My absolute max budget is 180,000 HUF for rent, and I'm hoping utilities stay under 25k. It must be fully furnished since I'm international. I don't drive, so being near a Tram stop is essential."
Output:
{
  "close_to_campus": ["Medical School"],
  "flexible_location": false
}
REASON: User says "need to be super close" - location is a HARD requirement.

Input: "Hello, we are two Civil Engineering students looking for a flat near Ótemető street. We need a 2-bedroom apartment, ideally around 60 m2. We are on a tight budget, so 140k HUF is our limit for rent, and maybe 30,000 for bills. We have our own beds, so an unfurnished place is fine. We take the bus to campus every day."
Output:
{
  "close_to_campus": ["Faculty of Engineering"],
  "flexible_location": false
}
REASON: User wants place "near Ótemető street" with no flexibility mentioned - location is important.

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "close_to_campus": ["Kassai Campus"],
  "flexible_location": false
}
REASON: User mentions Kassai Campus and wants to be "near Főnix Arena" - location is important to them.

Input: "I'm a Law student at Kassai. I need to be super close to the campus, like walking distance. Budget is 150k, 2 bedrooms, furnished."
Output:
{
  "close_to_campus": ["Kassai Campus"],
  "flexible_location": false
}
REASON: User emphasizes "super close" and "walking distance" - location is a HARD requirement.

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. It's also fine if the house is not near the Kassai campus, as long as I get 3 bedrooms. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "close_to_campus": ["Kassai Campus"],
  "flexible_location": true
}
REASON: User says "it's fine if the house is not near the Kassai campus" - location is FLEXIBLE. We still extract preferred campus for scoring.

Input: "Engineering student looking for 3-bedroom apartment. Budget 200k. Location doesn't matter as long as it has good bus connections and is affordable."
Output:
{
  "close_to_campus": ["Faculty of Engineering"],
  "flexible_location": true
}
REASON: User says "location doesn't matter" - FLEXIBLE. But we still extract Engineering campus from their major.

──────────────────────────────────────────────────────────────────
MULTIPLE CAMPUSES
──────────────────────────────────────────────────────────────────
If user mentions TWO different majors/locations, include BOTH:

Input: "Me and my twin brother are looking for a flat. I am studying Dentistry so I need to be near the Clinics, but he is a Law student at Kassai. We need a place that is roughly in the middle so neither of us has to commute too far."
Output:
{
  "close_to_campus": ["Medical School", "Kassai Campus"]
}

Input: "Hi, I'm doing my PhD in Biology at the Main Campus. My roommate is an Engineering student at Ótemető street. We are looking for a 2-bedroom flat with good bus connections to both of our faculties."
Output:
{
  "close_to_campus": ["Main Campus", "Faculty of Engineering"]
}

Input: "We are three girls from Spain. Two of us are studying Business at Böszörményi, and one is doing General Medicine. We need a large apartment."
Output:
{
  "close_to_campus": ["Böszörményi Campus", "Medical School"]
}
""",
    
  "query": """You are a structured data extraction assistant for a University of Debrecen housing platform.

Your ONLY job is to extract JSON data from user messages. Follow these rules EXACTLY.

The search_query should contain ONLY descriptive adjectives that are EXPLICITLY mentioned in the user's message.

Categories to extract:
- Atmosphere: quiet, peaceful, lively, social
- Style: modern, renovated, minimalist, cozy
- Features: balcony, garden, parking, bright, spacious
- Condition: clean, new, well-maintained
- Location preferences: If user mentions campus names or areas (Kassai Campus, Főnix Arena, Main Campus, etc.) as preferences, include them

CRITICAL RULES:
1. ONLY include words that are directly stated by the user
2. DO NOT add related or assumed keywords (e.g., don't add "pet friendly" unless user mentions pets)
3. REMOVE: numbers, prices, "I want", "looking for", "apartment", "flat"
4. DO NOT add "fully furnished" or "furnished" - this is a filter, not a search keyword
5. If user mentions location/campus but says it's flexible (e.g., "fine if not near X"), STILL include it in search_query as a soft preference

TRANSLATION TABLE (only apply if user mentions these):
- "2-3 bedroom" → add "spacious" to query
- "studio" → add "cozy compact" to query
- "cheap/budget/tight budget" → add "affordable" to query
- "luxury/premium" → add "high end premium" to query
- "walking distance" or "don't have a car" → add "convenient location" to query

EXAMPLES:

Input: "Hi! I'm a 4th year General Medicine student, so I need to be super close to the Clinics or Auguszta. I'm looking for a modern, renovated studio. My absolute max budget is 180,000 HUF for rent, and I'm hoping utilities stay under 25k. It must be fully furnished since I'm international. I don't drive, so being near a Tram stop is essential."
Output:
{
  "search_query": "modern renovated cozy compact convenient location"
}

Input: "Hello, we are two Civil Engineering students looking for a flat near Ótemető street. We need a 2-bedroom apartment, ideally around 60 m2. We are on a tight budget, so 140k HUF is our limit for rent, and maybe 30,000 for bills. We have our own beds, so an unfurnished place is fine. We take the bus to campus every day. We prefer a quiet area to study."
Output:
{
  "search_query": "affordable spacious quiet"
}

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. It's also fine if the house is not near the Kassai campus, as long as I get 3 bedrooms. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "search_query": "cool social modern balcony spacious large Kassai Campus Főnix Arena"
}
REASON: Even though location is flexible (close_to_campus=[]), we still include "Kassai Campus" and "Főnix Arena" in the semantic search query as PREFERENCES to boost those results.

Input: "International Dentistry student here. This is my first time in Debrecen. I need a cozy, modern studio very close to the Clinics. Fully furnished is essential. My budget is tight, maximum 160k rent and 18k utilities. I don't have a car so tram access is critical."
Output:
{
  "search_query": "cozy modern compact affordable convenient location"
}
"""
}

# Available models to test

# Schema for individual prompt responses
class PricesResponse(BaseModel):
    price_huf: Optional[int] = None
    avg_util_cost_huf: Optional[int] = None

class BasicsResponse(BaseModel):
    furnished: Optional[bool] = None
    size_m2: Optional[int] = None
    num_bedrooms: Optional[int] = None

class TransportResponse(BaseModel):
    close_to_public_transportation: Optional[bool] = None
    transport_types: Optional[List[str]] = Field(default_factory=list)

class CampusResponse(BaseModel):
    close_to_campus: Optional[List[str]] = Field(default_factory=list)
    flexible_location: Optional[bool] = False

class QueryResponse(BaseModel):
    search_query: str = ""


async def parse_user_query(user_text: str) -> dict:
    """
    Parse user query by running it through all specialized prompts.
    Returns combined result matching UserIntent structure.
    """
    print(f"🧠 Parsing intent for: '{user_text}'")

    # Create async client
    client = ollama.AsyncClient()
    model = 'qwen2.5:7b'
    
    # Schema mapping for each prompt category
    schema_map = {
        "prices": PricesResponse.model_json_schema(),
        "basics": BasicsResponse.model_json_schema(),
        "transport": TransportResponse.model_json_schema(),
        "campus": CampusResponse.model_json_schema(),
        "query": QueryResponse.model_json_schema()
    }
    
    # Storage for combined results
    combined_data = {
        "search_query": "",
        "filters": {}
    }
    
    # Loop through each system prompt and extract data
    total_start_time = time.time()
    
    for prompt_name, system_prompt in SYSTEM_PROMPTS.items():
        prompt_start = time.time()
        
        try:
            response = await client.chat(
                model=model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_text}
                ],
                format=schema_map[prompt_name],
                options={
                    'temperature': 0.3,
                    'num_ctx': 8192
                }
            )
            
            prompt_time = time.time() - prompt_start
            parsed = json.loads(response.message.content)
            
            print(f"  ✅ {prompt_name:<12} ({prompt_time:.2f}s): {parsed}")
            
            # Merge results into combined_data
            if prompt_name == "query":
                combined_data["search_query"] = parsed.get("search_query", "")
            else:
                # Merge all other fields into filters
                for key, value in parsed.items():
                    combined_data["filters"][key] = value
                    
        except Exception as e:
            print(f"  ⚠️  {prompt_name:<12} ERROR: {str(e)}")
    
    total_time = time.time() - total_start_time
    print(f"⏱️  Total parsing time: {total_time:.2f}s")
    print(f"✅ Final combined result: {combined_data}")
    
    return combined_data


# Test queries for parse_user_query function
TEST_QUERIES_FOR_PARSER = [
    "I'm a Pharmacy student. Need a quiet, bright apartment with at least 50 m2. Fully furnished is a must. Budget is around 170,000 and bills should be under 20k. I prefer walking to the Medical School.",
    
    "Hey! Looking for a super cheap place, max 120k HUF. I'm a Computer Science student at Kassai Campus. Need something close to the tram, doesn't matter if it's furnished or not. Just needs to be clean and have WiFi.",
    
    "We are 4 Engineering students looking for a huge apartment near Ótemető. Budget is flexible, around 280,000 total. We need at least 80 sqm, 3-4 bedrooms, and it has to be on a bus line. Unfurnished is fine since we have furniture already.",
    
    "International Dentistry student here. This is my first time in Debrecen. I need a cozy, modern studio very close to the Clinics. Fully furnished is essential. My budget is tight, maximum 160k rent and 18k utilities. I don't have a car so tram access is critical.",
    
    "Looking for a peaceful place to focus on my PhD in Biology. Main Campus area preferred, near the Botanical Garden would be perfect. 1 bedroom, renovated, with a balcony. Budget around 180,000, utilities hopefully under 25k. Furnished would be nice but not required.",
    
    "Me and my girlfriend are Economics students at Böszörményi. We want a lively neighborhood with cafes and shops nearby. 2-bedroom apartment, modern style, around 70 m2. Budget is 200k max, utilities around 30k. Good public transport is a must since we both commute.",
    
    "Urgent! Starting Law school at Kassai next month. Need a furnished studio ASAP. Budget is 140,000 HUF max, bills should be cheap. I want something close to Főnix Arena, maybe walking distance. Prefer a quiet building since I study a lot.",
    
    "Hi, I'm a Music Conservatory student. I need excellent soundproofing for practicing piano. Looking for a 1-bedroom near Main Campus or Egyetem tér. Furnished, bright, and clean. My parents are helping so budget is around 190,000 rent and 20k bills. Parking would be a bonus.",
    
    "Three of us studying Business at Böszörményi Campus. We need a spacious 3-bedroom, furnished, with good bus connections. Our combined budget is 220,000 for rent and maybe 40k for utilities. We want a social area, not too far from downtown.",
    
    "Civil Engineering student, final year. Looking for a small, affordable studio or 1-bedroom near Ótemető street. Unfurnished is totally fine, I have everything. Budget is super tight - 110,000 HUF max, and bills under 15k. I bike everywhere so transport doesn't matter."
]


async def test_parser():
    """Test the parse_user_query function with sample queries"""
    print("=" * 80)
    print("🧪 TESTING PARSE_USER_QUERY FUNCTION")
    print("=" * 80)
    
    for i, query in enumerate(TEST_QUERIES_FOR_PARSER, 1):
        print(f"\n{'='*80}")
        print(f"TEST #{i}")
        print(f"{'='*80}")
        print(f"Query: {query}\n")
        
        result = await parse_user_query(query)
        
        print(f"\n📊 RESULT:")
        print(f"  search_query: {result['search_query']}")
        print(f"  filters:")
        for key, value in result['filters'].items():
            print(f"    {key}: {value}")
        print()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_parser())
