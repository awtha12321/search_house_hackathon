import pandas as pd
import ollama
import uuid
import ast


def load_and_embed_csv(path: str, description_column: str = "description", id_column: str = "id"):
    """
    Reads CSV, handles ID types (int/string), fixes list strings,
    creates embeddings, and ensures data integrity.
    """

    # 1. Load Data
    df = pd.read_csv(path)

    # Validation
    if description_column not in df.columns:
        raise ValueError(f"Column '{description_column}' not found!")
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found!")

    ids = []
    vectors = []
    payloads = []

    print(f"Processing {len(df)} rows from {path}...")

    # 2. Iterate and Process
    for index, row in df.iterrows():
        text = row.get(description_column)

        # Skip rows with empty descriptions
        if not isinstance(text, str) or not text.strip():
            continue

        # --- STEP A: GET EMBEDDING ---
        # We do this FIRST. If this fails, we skip the whole row.
        try:
            response = ollama.embeddings(model='nomic-embed-text', prompt=text)
            embedding = response['embedding']
        except Exception as e:
            print(f"Error embedding row {index}: {e}")
            continue

            # --- STEP B: PREPARE ID ---
        # Logic: If it's a number, keep as int. If not, make sure it's a valid string.
        raw_id = row[id_column]
        try:
            # Try to convert to integer (e.g., 5001)
            row_id = int(raw_id)
        except ValueError:
            # If it's a string (e.g. "apt_1"), hash it into a UUID so Qdrant accepts it
            row_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(raw_id)))

        # --- STEP C: PREPARE PAYLOAD ---
        clean_row = row.fillna("").to_dict()

        # Fix the "List inside String" problem (e.g. "['Bus', 'Tram']")
        list_columns = ["transport_types", "close_to_campus"]
        for col in list_columns:
            if col in clean_row and isinstance(clean_row[col], str):
                try:
                    clean_row[col] = ast.literal_eval(clean_row[col])
                except (ValueError, SyntaxError):
                    clean_row[col] = []

        # --- STEP D: APPEND ALL TOGETHER ---
        # Only append if we made it this far successfully
        ids.append(row_id)
        vectors.append(embedding)
        payloads.append(clean_row)

    # 3. SAFETY CHECK
    # This ensures main.py never crashes with IndexError
    if not (len(ids) == len(vectors) == len(payloads)):
        raise RuntimeError(f"Data Mismatch! IDs: {len(ids)}, Vectors: {len(vectors)}, Payloads: {len(payloads)}")

    return ids, vectors, payloads