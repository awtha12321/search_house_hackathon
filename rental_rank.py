import numpy as np
import pandas as pd

class RentalRanker:
    def __init__(self):
        # 1. Define Domain Knowledge Weights (Sum must be ~1.0)
        # We prioritize Price and Rooms as you suggested.
        self.weights = {
            "price_huf": 0.35,        # Adjusted down slightly to fit Transport
            "num_bedrooms": 0.20,     # High priority (Benefit)
            "campus_score": 0.15,     # High priority (Benefit)
            "avg_util_cost_huf": 0.10,# Low priority (Cost)
            "size_sqm": 0.05,         # Very low priority (Benefit)
            "furnished_score": 0.05,  # Low priority (Benefit)
            "transport_score": 0.05,  # NEW: General public transport
            "transport_type_score": 0.05 # NEW: Bus vs Tram preference
        }
        
        # Define which columns are "Costs" (Lower is better)
        self.cost_criteria = ["price_huf", "avg_util_cost_huf"]

    def rank_rentals(self, results, filters):
        """
        Applies TOPSIS algorithm to rank the vector search results.
        """
        if not results:
            return []

        # --- Step 1: Prepare the Data Matrix ---
        candidates = []
        for res in results:
            payload = res.payload
            
            # Calculate dynamic categorical scores
            campus_score = self._calculate_campus_score(payload, filters)
            furnished_score = self._calculate_furnished_score(payload, filters)
            transport_score = self._calculate_public_transport_score(payload, filters)
            transport_type_score = self._calculate_transport_type_score(payload, filters)
            
            # Create a row for this apartment
            row = {
                "id": res.id,
                "payload": payload,
                "semantic_score": res.score, # Keep the vector score to blend later
                
                # Numerical columns for TOPSIS
                "price_huf": float(payload.get("price_huf", 999999)),
                "num_bedrooms": float(payload.get("num_bedrooms", 0)),
                "avg_util_cost_huf": float(payload.get("avg_util_cost_huf", 50000)),
                "size_sqm": float(payload.get("size_sqm", 0)),
                "campus_score": campus_score,
                "furnished_score": furnished_score,
                "transport_score": transport_score,
                "transport_type_score": transport_type_score
            }
            candidates.append(row)

        df = pd.DataFrame(candidates)
        
        # --- Step 2: Vector Normalization ---
        # Formula: x / sqrt(sum(x^2))
        norm_df = df.copy()
        
        criteria_cols = ["price_huf", "num_bedrooms", "avg_util_cost_huf", 
                         "size_sqm", "campus_score", "furnished_score", 
                         "transport_score", "transport_type_score"]

        for col in criteria_cols:
            # Avoid division by zero
            denom = np.sqrt((df[col] ** 2).sum())
            if denom == 0:
                norm_df[col] = 0
            else:
                norm_df[col] = df[col] / denom

        # --- Step 3: Apply Weights ---
        weighted_df = norm_df.copy()
        for col, weight in self.weights.items():
            weighted_df[col] = weighted_df[col] * weight

        # --- Step 4: Determine Ideal (A+) and Negative Ideal (A-) ---
        ideal_solution = {}
        anti_ideal_solution = {}

        for col in criteria_cols:
            if col in self.cost_criteria:
                # For Cost: Ideal is MIN, Anti-Ideal is MAX
                ideal_solution[col] = weighted_df[col].min()
                anti_ideal_solution[col] = weighted_df[col].max()
            else:
                # For Benefit: Ideal is MAX, Anti-Ideal is MIN
                ideal_solution[col] = weighted_df[col].max()
                anti_ideal_solution[col] = weighted_df[col].min()

        # --- Step 5: Calculate Euclidean Distances ---
        # Distance to Best (S+)
        s_plus = np.sqrt(((weighted_df[criteria_cols] - pd.Series(ideal_solution)) ** 2).sum(axis=1))
        
        # Distance to Worst (S-)
        s_minus = np.sqrt(((weighted_df[criteria_cols] - pd.Series(anti_ideal_solution)) ** 2).sum(axis=1))

        # --- Step 6: Calculate Relative Closeness (The TOPSIS Score) ---
        # Score = S- / (S+ + S-)
        denominator = s_plus + s_minus
        topsis_scores = np.divide(s_minus, denominator, out=np.zeros_like(s_minus), where=denominator!=0)

        # --- Step 7: Final Hybrid Score ---
        # We blend the TOPSIS score (hard stats) with the Semantic Score (text match)
        final_scores = (topsis_scores * 0.8) + (df["semantic_score"] * 0.2)
        
        # Update original list
        final_results = []
        for idx, score in enumerate(final_scores):
            # Start with original payload
            item = candidates[idx]
            final_results.append({
                "payload": item["payload"],
                "score": score,
                "topsis_debug": topsis_scores[idx] # Keep for debugging!
            })

        # Sort descending
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results

    def _calculate_campus_score(self, payload, filters):
        """Converts categorical campus data to a 0-1 number."""
        target_campuses = filters.get("close_to_campus", [])
        
        # 1. User has no preference -> Neutral Score
        if not target_campuses:
            return 0.5 

        apt_campuses = payload.get("close_to_campus", [])
        
        # 2. Check for Match
        if isinstance(apt_campuses, list):
            # If the apartment is near ANY of the target campuses
            if any(campus in target_campuses for campus in apt_campuses):
                return 1.0 # Perfect match (Best possible score)

        # 3. No Match Found - Check Flexibility
        is_flexible = filters.get("flexible_location", False)

        if is_flexible:
            # User IS flexible: Give a "Neutral" score instead of a zero.
            return 0.5 
        else:
            # User is NOT flexible: strict penalty.
            return 0.0

    def _calculate_furnished_score(self, payload, filters):
        """Converts categorical furnished data to a 0-1 number."""
        target = filters.get("furnished")
        if target is None:
            return 0.5 # User doesn't care
        
        is_furnished = payload.get("furnished", False)
        if target == is_furnished:
            return 1.0 # Perfect match
        return 0.0 # Mismatch

    def _calculate_public_transport_score(self, payload, filters):
        """
        Handles general proximity to public transport.
        Filter=None -> 0.5
        Filter=True/False -> Strict Match (1.0) or Mismatch (0.0)
        """
        target = filters.get("close_to_public_transportation")
        
        # 1. User didn't specify -> Neutral
        if target is None:
            return 0.5
        
        # 2. Strict matching (True matches True, False matches False)
        has_transport = payload.get("close_to_public_transportation", False)
        
        if target == has_transport:
            return 1.0
        return 0.0

    def _calculate_transport_type_score(self, payload, filters):
        """
        Handles specific transport types (Bus, Tram).
        """
        target_types = filters.get("transport_types", [])
        
        # 1. User didn't specify or provided empty list -> Neutral
        if not target_types:
            return 0.5
            
        apt_types = payload.get("transport_types", [])
        
        # 2. Check for overlap
        if isinstance(apt_types, list):
            # If apartment has ANY of the requested types (Bus OR Tram)
            if any(t in target_types for t in apt_types):
                return 1.0
                
        return 0.0