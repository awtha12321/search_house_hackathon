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

The close_to_campus field is ONE OF THE MOST IMPORTANT filters.
You MUST extract it whenever the user mentions their major, faculty, or location keywords.

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
  "close_to_campus": ["Medical School"]
}

Input: "Hello, we are two Civil Engineering students looking for a flat near Ótemető street. We need a 2-bedroom apartment, ideally around 60 m2. We are on a tight budget, so 140k HUF is our limit for rent, and maybe 30,000 for bills. We have our own beds, so an unfurnished place is fine. We take the bus to campus every day."
Output:
{
  "close_to_campus": ["Faculty of Engineering"]
}

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "close_to_campus": ["Kassai Campus"]
}

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

CRITICAL RULES:
1. ONLY include words that are directly stated by the user
2. DO NOT add related or assumed keywords (e.g., don't add "pet friendly" unless user mentions pets)
3. REMOVE: numbers, prices, "I want", "looking for", "apartment", "flat", campus names, location names
4. DO NOT add "fully furnished" or "furnished" - this is a filter, not a search keyword

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

Input: "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
Output:
{
  "search_query": "cool social modern balcony spacious large"
}

Input: "International Dentistry student here. This is my first time in Debrecen. I need a cozy, modern studio very close to the Clinics. Fully furnished is essential. My budget is tight, maximum 160k rent and 18k utilities. I don't have a car so tram access is critical."
Output:
{
  "search_query": "cozy modern compact affordable convenient location"
}
"""
}

# Available models to test
MODELS = [
    'qwen2.5:7b',
    'llama3.2:3b',
    'mistral:v0.3',
]



  # Test queries for validation
TEST_QUERIES = [
    {
        "query": "Hi! I'm a 4th year General Medicine student, so I need to be super close to the Clinics or Auguszta. I'm looking for a modern, renovated studio. My absolute max budget is 180,000 HUF for rent, and I'm hoping utilities stay under 25k. It must be fully furnished since I'm international. I don't drive, so being near a Tram stop is essential.",
        "expected": {
            "prices": {"price_huf": 180000, "avg_util_cost_huf": 25000},
            "basics": {"furnished": True, "size_m2": None, "num_bedrooms": 1},
            "transport": {"close_to_public_transportation": True, "transport_types": ["Tram"]},
            "campus": {"close_to_campus": ["Medical School"]},
        }
    },
    {
        "query": "Hello, we are two Civil Engineering students looking for a flat near Ótemető street. We need a 2-bedroom apartment, ideally around 60 m2. We are on a tight budget, so 140k HUF is our limit for rent, and maybe 30,000 for bills. We have our own beds, so an unfurnished place is fine. We take the bus to campus every day.",
        "expected": {
            "prices": {"price_huf": 140000, "avg_util_cost_huf": 30000},
            "basics": {"furnished": False, "size_m2": 60, "num_bedrooms": 2},
            "transport": {"close_to_public_transportation": True, "transport_types": ["Bus"]},
            "campus": {"close_to_campus": ["Faculty of Engineering"]},
        }
    },
    {
        "query": "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k.",
        "expected": {
            "prices": {"price_huf": 220000, "avg_util_cost_huf": 40000},
            "basics": {"furnished": None, "size_m2": None, "num_bedrooms": 3},
            "transport": {"close_to_public_transportation": None, "transport_types": []},
            "campus": {"close_to_campus": ["Kassai Campus"]},
        }
    },
    {
        "query": "Looking for a cheap studio near the university, max 100k HUF. I don't care about furniture, I have my own stuff. Needs to be close to tram line 1.",
        "expected": {
            "prices": {"price_huf": 100000, "avg_util_cost_huf": None},
            "basics": {"furnished": False, "size_m2": None, "num_bedrooms": 1},
            "transport": {"close_to_public_transportation": True, "transport_types": ["Tram"]},
            "campus": {"close_to_campus": []},
        }
    },
    {
        "query": "I'm a Pharmacy student. Need a quiet, bright apartment with at least 50 m2. Fully furnished is a must. Budget is around 170,000 and bills should be under 20k. I prefer walking to the Medical School.",
        "expected": {
            "prices": {"price_huf": 170000, "avg_util_cost_huf": 20000},
            "basics": {"furnished": True, "size_m2": 50, "num_bedrooms": None},
            "transport": {"close_to_public_transportation": None, "transport_types": []},
            "campus": {"close_to_campus": ["Medical School"]},
        }
    },
    {
        "query": "Me and my twin brother are looking for a flat. I am studying Dentistry so I need to be near the Clinics, but he is a Law student at Kassai. We need a place that is roughly in the middle so neither of us has to commute too far.",
        "expected": {
            "prices": {"price_huf": None, "avg_util_cost_huf": None},
            "basics": {"furnished": None, "size_m2": None, "num_bedrooms": None},
            "transport": {"close_to_public_transportation": None, "transport_types": []},
            "campus": {"close_to_campus": ["Medical School", "Kassai Campus"]},
        }
    },
    {
        "query": "We're three Economics students from Böszörményi Campus. Need a 3-bedroom place, furnished, with good public transport. Budget is 200k, utilities around 35k. We want a lively area with cafes nearby.",
        "expected": {
            "prices": {"price_huf": 200000, "avg_util_cost_huf": 35000},
            "basics": {"furnished": True, "size_m2": None, "num_bedrooms": 3},
            "transport": {"close_to_public_transportation": True, "transport_types": ["Tram", "Bus"]},
            "campus": {"close_to_campus": ["Böszörményi Campus"]},
        }
    },
    {
        "query": "Biology PhD student at Main Campus. I need a peaceful, renovated 1-bedroom near Egyetem tér. 150,000 HUF max. Must have a balcony for my plants. Bills should be cheap, maybe 15k.",
        "expected": {
            "prices": {"price_huf": 150000, "avg_util_cost_huf": 15000},
            "basics": {"furnished": None, "size_m2": None, "num_bedrooms": 1},
            "transport": {"close_to_public_transportation": None, "transport_types": []},
            "campus": {"close_to_campus": ["Main Campus"]},
        }
    },
    {
        "query": "Looking for a massive apartment for 4 people, all Mechanical Engineering students. We study at Ótemető street. Need 80+ sqm, unfurnished. Budget: 250,000 HUF rent + 50k utilities. Must be on a bus route.",
        "expected": {
            "prices": {"price_huf": 250000, "avg_util_cost_huf": 50000},
            "basics": {"furnished": False, "size_m2": 80, "num_bedrooms": None},
            "transport": {"close_to_public_transportation": True, "transport_types": ["Bus"]},
            "campus": {"close_to_campus": ["Faculty of Engineering"]},
        }
    },
    {
        "query": "International student, Music Conservatory. I need a cozy, quiet studio with excellent soundproofing. Furnished please. 180k rent, 20k bills. Near Main Campus would be ideal.",
        "expected": {
            "prices": {"price_huf": 180000, "avg_util_cost_huf": 20000},
            "basics": {"furnished": True, "size_m2": None, "num_bedrooms": 1},
            "transport": {"close_to_public_transportation": None, "transport_types": []},
            "campus": {"close_to_campus": ["Main Campus"]},
        }
    }
]


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

class QueryResponse(BaseModel):
    search_query: str = ""


async def test_model_prompt_performance():
    """
    Tests each model against each system prompt category with all test queries.
    Measures both speed and accuracy for each combination.
    """
    print("=" * 80)
    print("🧪 STARTING MODEL & PROMPT PERFORMANCE TEST")
    print("=" * 80)
    
    client = ollama.AsyncClient()
    
    # Schema mapping for each prompt category
    schema_map = {
        "prices": PricesResponse.model_json_schema(),
        "basics": BasicsResponse.model_json_schema(),
        "transport": TransportResponse.model_json_schema(),
        "campus": CampusResponse.model_json_schema(),
        "query": QueryResponse.model_json_schema()
    }
    
    # Results storage
    results = {
        model: {
            prompt_name: {
                "total_time": 0,
                "avg_time": 0,
                "correct": 0,
                "total": 0,
                "accuracy": 0,
                "errors": []
            }
            for prompt_name in SYSTEM_PROMPTS.keys()
        }
        for model in MODELS
    }
    
    # Run tests
    for model in MODELS:
        print(f"\n{'=' * 80}")
        print(f"🤖 Testing Model: {model}")
        print(f"{'=' * 80}")
        
        for prompt_name, system_prompt in SYSTEM_PROMPTS.items():
            print(f"\n  📝 Prompt Category: {prompt_name}")
            prompt_start_time = time.time()
            
            for i, test_case in enumerate(TEST_QUERIES):
                print(f"current testing prompt: {test_case['query']}")
                query = test_case["query"]
                expected = test_case["expected"].get(prompt_name, {})
                
                # Skip if no expected data for this prompt category
                if not expected:
                    continue
                
                try:
                    # Time individual query
                    query_start = time.time()
                    
                    response = await client.chat(
                        model=model,
                        messages=[
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': query}
                        ],
                        format=schema_map[prompt_name],
                        options={
                            'temperature': 0.3,  # Lower temperature for more consistent results
                            'num_ctx': 8192
                        }
                    )
                    
                    query_time = time.time() - query_start
                    results[model][prompt_name]["total_time"] += query_time
                    
                    # Parse response
                    parsed = json.loads(response.message.content)
                    
                    # Check accuracy
                    is_correct = True
                    mismatches = []
                    
                    for key, expected_value in expected.items():
                        actual_value = parsed.get(key)
                        
                        # Handle list comparison
                        if isinstance(expected_value, list):
                            if sorted(actual_value or []) != sorted(expected_value):
                                is_correct = False
                                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
                        else:
                            if actual_value != expected_value:
                                is_correct = False
                                mismatches.append(f"{key}: expected {expected_value}, got {actual_value}")
                    
                    if is_correct:
                        results[model][prompt_name]["correct"] += 1
                        print(f"    ✅ Query {i+1}: PASS ({query_time:.2f}s)")
                    else:
                        print(f"    ❌ Query {i+1}: FAIL ({query_time:.2f}s) - {mismatches}")
                        results[model][prompt_name]["errors"].append({
                            "query_num": i+1,
                            "mismatches": mismatches
                        })
                    
                    results[model][prompt_name]["total"] += 1
                    
                except Exception as e:
                    print(f"    ⚠️  Query {i+1}: ERROR - {str(e)}")
                    results[model][prompt_name]["errors"].append({
                        "query_num": i+1,
                        "error": str(e)
                    })
                    results[model][prompt_name]["total"] += 1
            
            # Calculate averages for this prompt
            total = results[model][prompt_name]["total"]
            if total > 0:
                results[model][prompt_name]["avg_time"] = results[model][prompt_name]["total_time"] / total
                results[model][prompt_name]["accuracy"] = (results[model][prompt_name]["correct"] / total) * 100
            
            prompt_total_time = time.time() - prompt_start_time
            print(f"  ⏱️  Total time for '{prompt_name}': {prompt_total_time:.2f}s")
            print(f"  🎯 Accuracy: {results[model][prompt_name]['accuracy']:.1f}% ({results[model][prompt_name]['correct']}/{total})")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 80)
    
    for prompt_name in SYSTEM_PROMPTS.keys():
        print(f"\n🔹 {prompt_name.upper()} Prompt:")
        print(f"  {'Model':<20} {'Avg Time':<12} {'Accuracy':<12} {'Correct/Total'}")
        print(f"  {'-'*60}")
        
        # Sort by accuracy, then by speed
        sorted_models = sorted(
            MODELS,
            key=lambda m: (-results[m][prompt_name]["accuracy"], results[m][prompt_name]["avg_time"])
        )
        
        for model in sorted_models:
            r = results[model][prompt_name]
            if r["total"] > 0:
                print(f"  {model:<20} {r['avg_time']:<12.3f} {r['accuracy']:<12.1f} {r['correct']}/{r['total']}")
    
    # Best model per category
    print("\n" + "=" * 80)
    print("🏆 BEST MODEL PER CATEGORY")
    print("=" * 80)
    
    for prompt_name in SYSTEM_PROMPTS.keys():
        best_model = max(
            MODELS,
            key=lambda m: (results[m][prompt_name]["accuracy"], -results[m][prompt_name]["avg_time"])
        )
        r = results[best_model][prompt_name]
        print(f"  {prompt_name:<15} → {best_model:<20} (Accuracy: {r['accuracy']:.1f}%, Avg Time: {r['avg_time']:.3f}s)")
    
    return results

async def test_query_generation_manual():
    """
    Test ONLY the 'query' prompt category across all models.
    Display all responses for manual evaluation since there's no right/wrong answer.
    """
    print("=" * 80)
    print("🔍 QUERY GENERATION TEST - MANUAL EVALUATION")
    print("=" * 80)
    print("This test shows how each model generates search_query strings.")
    print("Judge the quality based on: relevance, conciseness, and keyword extraction.")
    print("=" * 80)
    
    client = ollama.AsyncClient()
    query_prompt = SYSTEM_PROMPTS["query"]
    schema = QueryResponse.model_json_schema()
    
    # Results storage for timing
    model_timings = {model: [] for model in MODELS}
    
    for i, test_case in enumerate(TEST_QUERIES):
        query_text = test_case["query"]
        
        print(f"\n{'='*80}")
        print(f"📋 TEST CASE #{i+1}")
        print(f"{'='*80}")
        print(f"Input Query:")
        print(f"  {query_text}")
        print(f"\n{'-'*80}")
        print(f"Model Responses:")
        print(f"{'-'*80}\n")
        
        for model in MODELS:
            try:
                # Time the response
                start_time = time.time()
                
                response = await client.chat(
                    model=model,
                    messages=[
                        {'role': 'system', 'content': query_prompt},
                        {'role': 'user', 'content': query_text}
                    ],
                    format=schema,
                    options={
                        'temperature': 0.3,
                        'num_ctx': 8192
                    }
                )
                
                elapsed = time.time() - start_time
                model_timings[model].append(elapsed)
                
                # Parse and display
                parsed = json.loads(response.message.content)
                search_query = parsed.get("search_query", "")
                
                print(f"  🤖 {model:<20} ({elapsed:.2f}s)")
                print(f"     → \"{search_query}\"")
                print()
                
            except Exception as e:
                print(f"  ⚠️  {model:<20} ERROR")
                print(f"     → {str(e)}")
                print()
    
    # Summary statistics
    print(f"\n{'='*80}")
    print(f"⏱️  SPEED SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Avg Time':<12} {'Min Time':<12} {'Max Time'}")
    print(f"{'-'*80}")
    
    for model in MODELS:
        timings = model_timings[model]
        if timings:
            avg_time = sum(timings) / len(timings)
            min_time = min(timings)
            max_time = max(timings)
            print(f"{model:<20} {avg_time:<12.3f} {min_time:<12.3f} {max_time:.3f}")
    
    print(f"\n{'='*80}")
    print(f"✅ TEST COMPLETE - Review responses above to judge quality")
    print(f"{'='*80}")


async def main_test():
    """Run the performance test"""
    # Uncomment the test you want to run:
    
    # Full test with accuracy checking for all prompts:
    # await test_model_prompt_performance()
    
    # Manual evaluation test for query generation only:
    await test_query_generation_manual()


if __name__ == "__main__":
    # Run the test
    asyncio.run(main_test())
    
