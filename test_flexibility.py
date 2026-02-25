import asyncio
from query_parser import parse_user_query

async def test():
    # Test case 1: Original rigid query
    query1 = "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
    
    # Test case 2: Flexible query
    query2 = "I'm starting my Computer Science degree at Kassai Campus. I want a cool, social place near Főnix Arena. I have a budget of 220,000 HUF. I definitely need a modern style, maybe with a balcony. I need a large place, maybe 3 bedrooms to share. It is also fine if the house is not near the Kassai campus, cuz as long as I get 3 bedroom apartment, I am fine. A gym or parking nearby would be nice. Utilities aren't a huge concern, maybe max 40k."
    
    print("=" * 80)
    print("TEST 1: RIGID LOCATION REQUIREMENT")
    print("=" * 80)
    result1 = await parse_user_query(query1)
    print(f"\nSearch Query: {result1['search_query']}")
    print(f"Filters: {result1['filters']}")
    print(f"\nclose_to_campus: {result1['filters']['close_to_campus']}")
    print("Expected: ['Kassai Campus'] (hard filter)")
    
    print("\n" + "=" * 80)
    print("TEST 2: FLEXIBLE LOCATION (PREFERENCE ONLY)")
    print("=" * 80)
    result2 = await parse_user_query(query2)
    print(f"\nSearch Query: {result2['search_query']}")
    print(f"Filters: {result2['filters']}")
    print(f"\nclose_to_campus: {result2['filters']['close_to_campus']}")
    print("Expected: [] (empty = no hard filter, but still in search_query)")
    
    print("\n" + "=" * 80)
    print("EXPLANATION")
    print("=" * 80)
    print("Test 1: Location is a HARD FILTER - only show Kassai Campus apartments")
    print("Test 2: Location is a PREFERENCE - show ALL apartments matching other filters,")
    print("        but semantic search will still boost Kassai Campus results higher")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test())
