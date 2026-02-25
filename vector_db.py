from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, Range, MatchValue, MatchAny
from rental_rank import RentalRanker



class QdrantStorage:
    # UPDATED: dim=768 is the standard for nomic-embed-text
    def __init__(self, url="http://localhost:6333", collection="real_estate_docs", dim=768): # ollama uses 768 dim
        self.client = QdrantClient(url, timeout=30)

        # DEBUG: Check if 'search' exists now
        # print("Client methods:", dir(self.client))

        self.collection = collection

        # Check if collection exists
        if not self.client.collection_exists(self.collection):
            print(f"Creating collection '{self.collection}' with dim={dim}...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
        else:
            # OPTIONAL SAFETY CHECK:
            # If the collection exists, we should probably check if the dimensions match.
            # But for now, just knowing you need 768 is enough.
            pass

    """ 
    Sample payload: 
    {'id': 5001, 'price_huf': 120000, 
    'close_to_public_transportation': True, 'close_to_campus': ['Main Campus', 'Medical School'], 'size_m2': 45, 'num_bedrooms': 1, 'avg_util_cost_huf': 25000,
    'furnished': True, 'transport_types': ['Tram', 'Bus'],
    'description': 'Cozy renovated flat right next to the Main Building. Perfect for medical students who need quick access to the clinics. The windows face a quiet park.'}
    """

    def upsert(self, ids, vectors, payloads):
        # the vector we see is gonna be the spcial column already encoded from another file
        # payload is basically the whole CSV row but in JSON format
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    # NOTE: very important about the desciprtion column. it hsa to be portray the house accurately. the more accurate it is, the better the search result is gonna be.


    from qdrant_client.models import Range, Filter, FieldCondition, MatchValue, MatchAny
    
    def search_rentals(self, search_query: str, filters: dict, top_k: int = 10, price_buffer_rent: int = 30000, price_buffer_util: int = 20000, num_room_deduct: int = 1):
        import ollama
        
        query_embedding = ollama.embeddings(
            model='nomic-embed-text', 
            prompt=search_query
        )['embedding']
        
        
        must_conditions = []  # Required filters
        
        if filters:
            for key, value in filters.items():
                # Skip None values and empty lists
                if value is None or (isinstance(value, list) and len(value) == 0):
                    continue
                
                # Skip furnished filter - treat it as a preference, not a hard requirement
                # People who say "unfurnished" might still want to see furnished options
                # People who say "furnished" might fall for minimal/unfurnished places
                if key == "furnished":
                    continue
                
                # Skip flexible_location - this is just a control flag, not a filter
                if key == "flexible_location":
                    continue
                
                # Handle close_to_campus based on flexible_location flag
                if key == "close_to_campus":
                    is_flexible = filters.get("flexible_location", False)
                    if is_flexible:
                        # Location is flexible - skip hard filter, use for scoring only
                        continue
                    else:
                        # Location is required - apply as hard filter
                        if isinstance(value, list) and len(value) > 0:
                            must_conditions.append(
                                FieldCondition(key=key, match=MatchAny(any=value))
                            )
                        continue
                    
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    # Add buffer for price to allow slightly higher options
                    if key == "price_huf":
                        must_conditions.append(
                            FieldCondition(key=key, range=Range(lte=value + price_buffer_rent))
                        )
                    elif key == "avg_util_cost_huf":
                        must_conditions.append(
                            FieldCondition(key=key, range=Range(lte=value + price_buffer_util))
                        )
                    else:
                        # For num_bedrooms: search for (requested - 1) or more rooms
                        # This allows flexibility while still filtering out too-small options
                        must_conditions.append(
                            FieldCondition(key=key, range=Range(gte=value - num_room_deduct))
                        )
                elif isinstance(value, bool):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                elif isinstance(value, list): 
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchAny(any=value))
                    )
                else:
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                    
        query_filter = Filter(must=must_conditions) if must_conditions else None
        # print(query_filter)

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding, 
            query_filter=query_filter,
            with_payload=True,
            limit=top_k * 2  # Get more results before re-ranking
        ).points 
        
        # Apply custom scoring system     
        ranker = RentalRanker()    
        scored_results = ranker.rank_rentals(results, filters)
        
        # Return top_k results with combined score
        return scored_results[:top_k] # do i want to return evverything? 


    def list_all_collections(self):
        """List all collections in the Qdrant database"""
        collections = self.client.get_collections()
        print(f"\n📊 All collections in Qdrant:")
        print(f"{'='*60}")
        for collection in collections.collections:
            print(f"  - {collection.name}")
            # Get collection info
            info = self.client.get_collection(collection.name)
            print(f"    Points: {info.points_count}")
            print(f"    Vector size: {info.config.params.vectors.size}")
        print(f"{'='*60}\n")
        return collections

        
        
if __name__ == "__main__":
    storage = QdrantStorage()
    
    # List all collections
    print("Listing all collections:")
    storage.list_all_collections()
    
    # Test search
    print("\nTesting search:")
    results = storage.search_rentals('affordable compact convenient location', {'price_huf': 110000, 'avg_util_cost_huf': 15000, 'furnished': False, 'size_m2': None, 'num_bedrooms': 1, 'close_to_public_transportation': None, 'transport_types': [], 'close_to_campus': ['Faculty of Engineering']}, top_k=5)
    print(results)