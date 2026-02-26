import json
import sqlite3
import os
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def create_rich_metadata_db(vps_json_path, results_json_path, output_db_path):
    """
    Creates a SQLite database mapping FAISS IDs to rich metadata (filename, text, vienna, year).
    """
    if not os.path.exists(vps_json_path):
        logger.error(f"VPS ID Map not found at {vps_json_path}")
        return
    
    if not os.path.exists(results_json_path):
        logger.error(f"Results metadata not found at {results_json_path}")
        return

    # 1. Load FAISS ID Order
    logger.info(f"Loading FAISS ID map from {vps_json_path}...")
    with open(vps_json_path, 'r') as f:
        vps_ids = json.load(f)  # List of filenames in FAISS order

    # 2. Load Metadata into a lookup map
    logger.info(f"Loading Metadata from {results_json_path}...")
    with open(results_json_path, 'r') as f:
        metadata_list = json.load(f)
    
    # Filename -> Metadata Object
    metadata_lookup = {item['file']: item for item in tqdm(metadata_list, desc="Indexing Metadata")}

    # 3. Create SQLite DB
    if os.path.exists(output_db_path):
        os.remove(output_db_path)
    
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE id_map (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            tm_text TEXT,
            vienna_codes TEXT,
            year INTEGER
        )
    ''')
    
    # 4. Preparing data for batch insert
    logger.info("Preparing database entries...")
    insert_data = []
    for i, filename in enumerate(tqdm(vps_ids, desc="Preparing DB")):
        meta = metadata_lookup.get(filename, {})
        
        # Serialize vienna_codes as a comma-separated string or JSON string
        vienna = meta.get('vienna_codes', "")
        if isinstance(vienna, list):
            vienna = ",".join(vienna)
            
        insert_data.append((
            i,
            filename,
            meta.get('text', ""),
            vienna,
            meta.get('year', None)
        ))
    
    # 5. Batch Insert
    logger.info(f"Inserting {len(insert_data):,} records into {output_db_path}...")
    cursor.executemany('INSERT INTO id_map VALUES (?, ?, ?, ?, ?)', insert_data)
    
    conn.commit()
    conn.close()
    logger.success("🚀 Rich Metadata Database Created!")

if __name__ == "__main__":
    # Paths for best_model
    vps_map = "models/onnx/best_model/vps_id_map.json"
    raw_results = os.path.join(os.getenv("RAW_DATASET_DIR"), "results.json")
    db_out = "models/onnx/best_model/vps_id_map.db"
    
    create_rich_metadata_db(vps_map, raw_results, db_out)
